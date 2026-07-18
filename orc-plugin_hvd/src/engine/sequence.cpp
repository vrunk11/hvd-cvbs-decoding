// SPDX-License-Identifier: GPL-3.0-or-later
//
// sequence.cpp — port of decode_sequence's per-window body (reference/hvd/
// decoder.py). Reference line references: prepare_field (1433), the driver
// loop (1560-1730), synth_reference (741), psi_closed_form (1292).

#include "engine/sequence.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <utility>

#include "engine/engine.h"
#include "engine/holographic_init.h"
#include "engine/lockin.h"
#include "engine/motion.h"
#include "engine/temporal.h"
#include "engine/variational.h"

namespace hvd {

namespace {

constexpr float kPi = 3.14159265358979323846F;

float MedianOf(std::vector<float> v) {
  if (v.empty()) return 0.0F;
  const size_t k = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + k, v.end());
  float m = v[k];
  if (v.size() % 2 == 0) {
    std::nth_element(v.begin(), v.begin() + k - 1, v.begin() + k);
    m = 0.5F * (m + v[k - 1]);
  }
  return m;
}

// ---- synth_reference (reference line 741) --------------------------------
// Build a denoised reference for field j by motion-compensated temporal
// blending of the *decoded* fields, then re-encode it to composite for an
// honesty check. NTSC geometry does the heavy lifting: separation leakage
// anti-correlates at +/-1 frame (180 deg carrier flip), so the blend cancels
// it; at +/-2 frames the carrier is back in phase, so those neighbours
// average noise without touching signal. Geman-McClure per-pixel weights
// drop motion residuals out. conf_ref (the anchor weight) is measured in
// the COMPOSITE domain: 1 where the re-encoded synthesis explains the raw
// measurement, -> 0 where it does not — the anchor never trusts a reference
// the data contradicts.
AnchorTerm SynthReference(int j, const std::vector<Plane>& Ys,
                          const std::vector<ComplexPlane>& chis,
                          const std::vector<FieldObs>& fields,
                          const HvdConfig& cfg,
                          const std::vector<int>& parities,
                          std::map<std::pair<int, int>, MotionField>* mcache,
                          std::mutex* mcache_mu) {
  const Plane& S = fields[j].s;
  const ComplexPlane& carrier = fields[j].carrier;
  const float eps_b = cfg.nr_eps;
  const int hh = Ys[j].height();
  const int ww = Ys[j].width();
  const int tile = cfg.mc_tile;
  const long n = static_cast<long>(Ys[j].size());

  Plane accY = Ys[j];
  ComplexPlane accC = chis[j];
  Plane accW(hh, ww, 1.0F);

  for (int k = j - cfg.nr_radius; k <= j + cfg.nr_radius; ++k) {
    if (k == j || k < 0 || k >= static_cast<int>(Ys.size())) continue;
    // The shared fast-mode cache is read/written under `mcache_mu` when the
    // colored parallel sweep is active (map iterators/references are not
    // stable across concurrent inserts) — the estimate itself runs outside
    // the lock; a rare duplicate estimate on a race is benign (same inputs,
    // same result).
    MotionField local;
    const MotionField* mo = &local;
    bool have = false;
    if (mcache) {
      std::unique_lock<std::mutex> lk;
      if (mcache_mu) lk = std::unique_lock<std::mutex>(*mcache_mu);
      auto it = mcache->find({j, k});
      if (it != mcache->end()) {
        local = it->second;
        have = true;
      }
    }
    if (!have) {
      local = EstimateMotion(Ys[k], Ys[j], tile, cfg.mc_search);
      if (mcache) {
        std::unique_lock<std::mutex> lk;
        if (mcache_mu) lk = std::unique_lock<std::mutex>(*mcache_mu);
        (*mcache)[{j, k}] = local;
      }
    }
    const Plane conf = UpsampleConfidence(mo->confidence, tile, hh, ww);
    // Decoded fields are BASEBAND: bilinear sub-pixel warp is legal, and
    // the half-line parity offset between opposite fields is compensated
    // exactly on the sampling grid.
    const float row_off =
        (static_cast<float>(parities[j]) - static_cast<float>(parities[k])) /
        2.0F;
    const PerPixelMotion vpix = VectorsPerPixel(mo->dy, mo->dx, tile, hh, ww);
    const Plane Yw = WarpBilinearTiles(Ys[k], vpix, row_off, hh, ww);
    Plane re(hh, ww), im(hh, ww);
    for (long i = 0; i < n; ++i) {
      re[i] = chis[k][i].real();
      im[i] = chis[k][i].imag();
    }
    const Plane wre = WarpBilinearTiles(re, vpix, row_off, hh, ww);
    const Plane wim = WarpBilinearTiles(im, vpix, row_off, hh, ww);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) {
      const Complex cw(wre[i], wim[i]);
      const float dy = Yw[i] - Ys[j][i];
      const float d2 =
          dy * dy + std::norm(cw - chis[j][i]);
      const float w = conf[i] * eps_b * eps_b / (d2 + eps_b * eps_b);
      accY[i] += w * Yw[i];
      accC[i] += w * cw;
      accW[i] += w;
    }
  }

  AnchorTerm out;
  out.luma = Plane(hh, ww);
  out.chroma = ComplexPlane(hh, ww);
  Plane resid(hh, ww);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long i = 0; i < n; ++i) {
    out.luma[i] = accY[i] / accW[i];
    out.chroma[i] = accC[i] / accW[i];
    const float s_hat = out.luma[i] + (out.chroma[i] * carrier[i]).real();
    resid[i] = std::fabs(S[i] - s_hat);
  }
  const Plane r = BoxBlur(std::move(resid), 1);
  out.weight = Plane(hh, ww);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long i = 0; i < n; ++i)
    out.weight[i] = eps_b * eps_b / (r[i] * r[i] + eps_b * eps_b);
  return out;
}

// ---- psi_closed_form (reference line 1292) --------------------------------
// N-step phase-shifting-interferometry init: every static pixel is observed
// under several KNOWN carrier phases (its own field plus each MC-warped
// neighbour). Per observation k: S_k = Y + p*Re[c_k] - q*Im[c_k], chi=p+iq —
// exactly an N-step interferogram set; the weighted normal equations are a
// 3x3 per-pixel system solved in closed form (Cramer). Where confidence is
// high it lands on the temporally exact solution before any spatial prior;
// where total confidence is low it blends back to the fallback init.
ComplexPlane PsiClosedForm(const Plane& S, const ComplexPlane& carrier,
                           const std::vector<NeighborTerm>& neighbors,
                           const ComplexPlane& chi_fallback) {
  const int hh = S.height();
  const int ww = S.width();
  const long n = static_cast<long>(S.size());
  ComplexPlane out(hh, ww);
  constexpr float kEpsT = 6.0F;  // the refine's robust gate AT THE INIT
                                 // POINT — without it the closed form
                                 // commits to motion-contaminated chi

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long i = 0; i < n; ++i) {
    // Accumulate weighted normal equations A^T W A over the observations:
    // basis [1, Re c, -Im c].
    double a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
    double b1 = 0, b2 = 0, b3 = 0;
    double wsum_nbr = 0.0;
    auto add = [&](float Sk, Complex ck, float wk) {
      const double rc = ck.real();
      const double ic = -ck.imag();
      a11 += wk;
      a12 += wk * rc;
      a13 += wk * ic;
      a22 += wk * rc * rc;
      a23 += wk * rc * ic;
      a33 += wk * ic * ic;
      b1 += wk * Sk;
      b2 += wk * Sk * rc;
      b3 += wk * Sk * ic;
    };
    add(S[i], carrier[i], 1.0F);
    for (const NeighborTerm& nb : neighbors) {
      const Complex dc = carrier[i] - nb.carrier[i];
      const float rt =
          (nb.composite[i] - S[i]) + (chi_fallback[i] * dc).real();
      const float wk =
          nb.confidence[i] * kEpsT * kEpsT / (rt * rt + kEpsT * kEpsT);
      add(nb.composite[i], nb.carrier[i], wk);
      wsum_nbr += wk;
    }
    const double det = a11 * (a22 * a33 - a23 * a23) -
                       a12 * (a12 * a33 - a23 * a13) +
                       a13 * (a12 * a23 - a22 * a13);
    // Phase diversity: with < 3 well-separated phases the system is
    // near-singular (det -> 0); blend to fallback there.
    const double a11c = std::max(a11, 1e-9);
    const bool ok = det > 1e-3 * a11c * a11c * a11c * 0.01;
    const double d = ok ? det : 1.0;
    const double q1 = (a11 * (b2 * a33 - b3 * a23) -
                       b1 * (a12 * a33 - a23 * a13) +
                       a13 * (a12 * b3 - b2 * a13)) /
                      d;  // p = Re chi
    const double q2 = (a11 * (a22 * b3 - a23 * b2) -
                       a12 * (a12 * b3 - b2 * a13) +
                       b1 * (a12 * a23 - a22 * a13)) /
                      d;  // q = Im chi
    const float t =
        ok ? std::clamp(static_cast<float>(wsum_nbr / 2.0), 0.0F, 1.0F) : 0.0F;
    const Complex psi(static_cast<float>(q1), static_cast<float>(q2));
    out[i] = t * psi + (1.0F - t) * chi_fallback[i];
  }
  return out;
}

}  // namespace

FieldObs PrepareFieldObs(const FieldInput& field, const FieldGeometry& g,
                         const HvdConfig& cfg, int parity) {
  std::vector<float> theta = BurstLockinPhase(field.samples, g);
  // Chroma phase correction on the burst-locked reference itself, same as
  // the frame path (WeaveAndBuildCarrier in engine.cpp).
  if (cfg.chroma_phase_deg != 0.0F) {
    const float offset = -cfg.chroma_phase_deg * kPi / 180.0F;
    for (float& t : theta) t += offset;
  }
  const int fal = g.first_active_field_line;
  const int lal = g.last_active_line();
  const int a0 = g.active_video_start;
  const int a1 = g.active_video_end;
  const int lines = std::max(0, lal - fal);
  const int width = std::max(0, a1 - a0);

  FieldObs obs;
  obs.parity = parity;
  obs.s = Plane(lines, width);
  for (int y = 0; y < lines; ++y)
    for (int x = 0; x < width; ++x)
      obs.s.at(y, x) = field.samples.at(fal + y, a0 + x);
  std::vector<float> theta_active(lines);
  for (int y = 0; y < lines; ++y) theta_active[y] = theta[fal + y];
  obs.carrier = MakeCarrier(theta_active, g);
  return obs;
}

std::vector<DecodedField> DecodeFieldWindow(const std::vector<FieldObs>& fields,
                                            const FieldGeometry& g,
                                            const HvdConfig& cfg, Fft2d* fft,
                                            SequenceDiagnostics* diag) {
  std::vector<DecodedField> inits(fields.size());
  for (size_t j = 0; j < fields.size(); ++j) {
    HoloInit hi = HolographicInit(fields[j].s, fields[j].carrier, g, cfg, fft);
    inits[j].luma = std::move(hi.luma);
    inits[j].chroma = std::move(hi.chroma);
  }
  return DecodeFieldWindowWithInits(fields, inits, g, cfg, diag);
}

DrizzleResult DrizzleFrame(int j0, const std::vector<Plane>& Ys,
                           const std::vector<ComplexPlane>& chis,
                           const std::vector<int>& parities,
                           const HvdConfig& cfg, int scale) {
  const int L = Ys[j0].height();
  const int W = Ys[j0].width();
  const int HF = 2 * L * scale;  // fine frame height
  const int n = static_cast<int>(Ys.size());
  const float eps_b = cfg.nr_eps > 0.0F ? cfg.nr_eps : 3.0F;

  Plane accY(HF, W, 0.0F);
  Plane accCr(HF, W, 0.0F), accCi(HF, W, 0.0F);
  Plane accW(HF, W, 0.0F);

  // Scatter accumulation is inherently write-conflicting across sources
  // (nearby samples land on the same fine rows), so the deposit loops stay
  // sequential per contributing field; the arithmetic per deposit is tiny
  // next to the motion estimation feeding it.
  for (int jt = j0; jt <= j0 + 1; ++jt) {
    for (int k = std::max(0, j0 - 2 * cfg.nr_radius);
         k < std::min(n, j0 + 2 + 2 * cfg.nr_radius); ++k) {
      const int pk = parities[k];
      if ((k == j0 || k == j0 + 1) && k != jt)
        continue;  // the sibling field deposits on its own turn
      MotionField mo;
      if (k == jt) {
        mo.dy = Plane(1, 1, 0.0F);
        mo.dx = Plane(1, 1, 0.0F);
        mo.confidence = Plane(1, 1, 1.0F);
        mo.tile = cfg.mc_tile;
      } else {
        mo = EstimateMotion(Ys[k], Ys[jt], cfg.mc_tile, cfg.mc_search);
      }
      const PerPixelMotion v =
          VectorsPerPixel(mo.dy, mo.dx, cfg.mc_tile, L, W);
      const int th = mo.confidence.height();
      const int tw = mo.confidence.width();
      for (int y = 0; y < L; ++y) {
        const int cty = std::min(y / cfg.mc_tile, th - 1);
        for (int x = 0; x < W; ++x) {
          const int ctx = std::min(x / cfg.mc_tile, tw - 1);
          const float cpx = mo.confidence.at(cty, ctx);
          const float vy = v.dy.at(y, x);
          const float vx = v.dx.at(y, x);
          // Robust agreement with the target field's own decode, evaluated
          // at the integer-rounded warp (cheap).
          const int syi = std::clamp(
              static_cast<int>(std::lround(y - vy)), 0, L - 1);
          const int sxi = std::clamp(
              static_cast<int>(std::lround(x - vx)), 0, W - 1);
          const float dY = Ys[k].at(y, x) - Ys[jt].at(syi, sxi);
          const float d2 =
              dY * dY + std::norm(chis[k].at(y, x) - chis[jt].at(syi, sxi));
          const float w = cpx * eps_b * eps_b / (d2 + eps_b * eps_b);
          // Source sample (y, x) of field k lands, in target-frame fine
          // rows, at (2*(y + vy) + pk) * scale; linear split between the
          // two nearest fine rows ('pixfrac=1, linear kernel' drizzle).
          const float yf = (2.0F * (y + vy) + static_cast<float>(pk)) *
                           static_cast<float>(scale);
          const int xs = std::clamp(
              static_cast<int>(std::lround(x + vx)), 0, W - 1);
          const int y0 = static_cast<int>(std::floor(yf));
          const float fy = yf - static_cast<float>(y0);
          const Complex ck = chis[k].at(y, x);
          const float sk = Ys[k].at(y, x);
          for (int off = 0; off < 2; ++off) {
            const int yt = std::clamp(y0 + off, 0, HF - 1);
            const float ww = w * (off == 0 ? (1.0F - fy) : fy);
            accY.at(yt, xs) += ww * sk;
            accCr.at(yt, xs) += ww * ck.real();
            accCi.at(yt, xs) += ww * ck.imag();
            accW.at(yt, xs) += ww;
          }
        }
      }
    }
  }

  // Fallback where coverage is thin: linear vertical interpolation of the
  // plain woven decode. (The reference computes an unused r0/fr/r1 block
  // here — dead code, not carried over.)
  const int p0 = parities[j0];
  const int p1 = parities[j0 + 1];
  Plane wovenY(2 * L, W);
  Plane wovenCr(2 * L, W), wovenCi(2 * L, W);
  for (int r = 0; r < L; ++r) {
    for (int x = 0; x < W; ++x) {
      wovenY.at(2 * r + p0, x) = Ys[j0].at(r, x);
      wovenY.at(2 * r + p1, x) = Ys[j0 + 1].at(r, x);
      wovenCr.at(2 * r + p0, x) = chis[j0].at(r, x).real();
      wovenCr.at(2 * r + p1, x) = chis[j0 + 1].at(r, x).real();
      wovenCi.at(2 * r + p0, x) = chis[j0].at(r, x).imag();
      wovenCi.at(2 * r + p1, x) = chis[j0 + 1].at(r, x).imag();
    }
  }

  DrizzleResult out;
  out.luma = Plane(HF, W);
  out.chroma = ComplexPlane(HF, W);
  constexpr float kLam = 0.35F;  // coverage confidence scale
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int r = 0; r < HF; ++r) {
    const float fine = static_cast<float>(r) / static_cast<float>(scale);
    const int f0 = std::clamp(static_cast<int>(std::floor(fine)), 0, 2 * L - 2);
    const float ff = fine - static_cast<float>(f0);
    for (int x = 0; x < W; ++x) {
      const float baseY =
          wovenY.at(f0, x) * (1.0F - ff) + wovenY.at(f0 + 1, x) * ff;
      const Complex baseC{
          wovenCr.at(f0, x) * (1.0F - ff) + wovenCr.at(f0 + 1, x) * ff,
          wovenCi.at(f0, x) * (1.0F - ff) + wovenCi.at(f0 + 1, x) * ff};
      const float aw = accW.at(r, x);
      const float mix = aw / (aw + kLam);
      const float dzY = aw > 0.0F ? accY.at(r, x) / std::max(aw, 1e-9F) : 0.0F;
      const Complex dzC =
          aw > 0.0F ? Complex{accCr.at(r, x) / std::max(aw, 1e-9F),
                              accCi.at(r, x) / std::max(aw, 1e-9F)}
                    : Complex{};
      out.luma.at(r, x) = mix * dzY + (1.0F - mix) * baseY;
      out.chroma.at(r, x) = mix * dzC + (1.0F - mix) * baseC;
    }
  }
  return out;
}

std::vector<DecodedField> DecodeFieldWindowWithInits(
    const std::vector<FieldObs>& fields,
    const std::vector<DecodedField>& inits, const FieldGeometry& g,
    const HvdConfig& cfg_in, SequenceDiagnostics* diag) {
  (void)g;
  const int nf = static_cast<int>(fields.size());
  std::vector<DecodedField> out(nf);
  if (nf == 0) return out;

  // --- self-calibration: measure the source's noise, scale the robust
  // gates accordingly (a fixed IRE gate is 5 sigma on a clean disc but
  // 1.4 sigma on a noisy one => over-gating exactly where 3D helps most)
  HvdConfig ccfg = cfg_in;
  {
    std::vector<float> sigmas;
    sigmas.reserve(fields.size());
    for (const FieldObs& f : fields) sigmas.push_back(EstimateNoiseIre(f.s));
    const float sigma = MedianOf(std::move(sigmas));
    if (diag) diag->sigma_ire = sigma;
    if (ccfg.temporal_eps <= 0.0F)
      ccfg.temporal_eps = std::clamp(7.0F * sigma, 4.0F, 20.0F);
    if (ccfg.nr_eps <= 0.0F)
      ccfg.nr_eps = std::clamp(3.0F * sigma, 3.0F, 12.0F);
  }

  std::vector<int> offs;
  if (ccfg.extended_temporal) {
    offs = ccfg.bidirectional ? std::vector<int>{-3, -2, -1, 1, 2, 3}
                              : std::vector<int>{-3, -2, -1};
  } else {
    offs = ccfg.bidirectional ? std::vector<int>{-2, -1, 1, 2}
                              : std::vector<int>{-2, -1};
  }

  std::vector<Plane> Ys(nf);
  std::vector<ComplexPlane> chis(nf);
  std::vector<int> parities(nf);
  for (int j = 0; j < nf; ++j) {
    chis[j] = inits[j].chroma;
    if (!inits[j].luma.empty()) {
      Ys[j] = inits[j].luma;
    } else {
      Ys[j] = Plane(fields[j].s.height(), fields[j].s.width());
      for (size_t i = 0; i < Ys[j].size(); ++i)
        Ys[j][i] =
            fields[j].s[i] - (chis[j][i] * fields[j].carrier[i]).real();
    }
    parities[j] = fields[j].parity;
  }

  // Per-window motion cache. Fast mode shares one estimate per field pair
  // across passes AND with the anchor blend (~4x fewer block-matching
  // runs); slow mode keys per pass (bit-exact preservation of the
  // reference's per-pass estimates). Key: (j, k) with pass folded in for
  // slow mode via a distinct map per pass.
  std::map<std::pair<int, int>, MotionField> mcache_fast;

  // ADAPTIVE temporal strength. Convention: strength < 0 => 3D OFF (the
  // stage passes -1 when the switch is off); == 0 => AUTO (measure the
  // content); > 0 => fixed (reference behaviour). The right strength is a
  // property of the CONTENT: on Y/C-ambiguous material (luma energy at
  // the subcarrier) the neighbour equations resolve what 2D cannot and
  // deserve to be strong; on unambiguous material they can only lift
  // chroma noise and should stay weak. The measurement uses the phase
  // physics directly: d_j = 5-tap demod of S by the field's own carrier;
  // for a same-parity pair (j, j+2) the carrier has flipped 180 degrees,
  // so true (static) chroma appears IDENTICALLY in both while luma
  // leaking through the chroma band flips sign — (d_j - d_{j+2}) / 2
  // isolates exactly the ambiguous energy. Noise is subtracted in
  // quadrature via the same sigma the robust gates calibrated from;
  // motion only biases the estimate upward, i.e. toward "more 3D", where
  // the per-pixel gates already protect the result.
  if (ccfg.temporal_strength == 0.0F && ccfg.cg_iterations > 0 && nf > 2) {
    std::vector<float> ambs;
    for (int j = 0; j + 2 < nf; ++j) {
      const Plane& Sj = fields[j].s;
      const Plane& Sk = fields[j + 2].s;
      const int hh = Sj.height();
      const int ww = Sj.width();
      // Crude demod, decimated: this is a scalar calibration, not a decode.
      double acc = 0.0;
      long n = 0;
      for (int y = 2; y < hh - 2; y += 3) {
        for (int x = 4; x + 7 < ww; x += 4) {
          // Triangle-7 (= box4 convolved with box4): a DOUBLE null at the
          // carrier frequency (pi/2 rad/px at 4fsc). The demod shifts
          // ordinary luma to ~pi/2; a single 4-tap null is exact only ON
          // the carrier and leaks ~9% a tenth of a radian away — enough
          // for a 20 IRE luma ramp to dwarf real ambiguity. The double
          // null rejects the whole neighbourhood (<1% at +/-0.12 rad), so
          // what survives is energy that genuinely sat NEAR fsc before
          // the demod: the actual Y/C ambiguity.
          auto demod = [&](const Plane& S, const ComplexPlane& car) {
            static constexpr float kW[7] = {1.0F, 2.0F, 3.0F, 4.0F,
                                            3.0F, 2.0F, 1.0F};
            Complex d{0.0F, 0.0F};
            for (int k = 0; k < 7; ++k)
              d += kW[k] * S.at(y, x + k) * std::conj(car.at(y, x + k));
            return d * (1.0F / 16.0F);
          };
          const Complex dj = demod(Sj, fields[j].carrier);
          const Complex dk = demod(Sk, fields[j + 2].carrier);
          acc += 0.25 * std::norm(dj - dk);
          ++n;
        }
      }
      if (n > 0) ambs.push_back(std::sqrt(static_cast<float>(acc / n)));
    }
    float amb = MedianOf(std::move(ambs));
    // Remove the noise contribution: the triangle-7 demod carries
    // sum(w^2)/16^2 = 44/256 of sigma^2 per field; the half-difference of
    // two fields halves that -> rms ~= 0.29*sigma. Use the same sigma the
    // robust gates calibrated from.
    const float sigma_d = ccfg.nr_eps > 0.0F ? ccfg.nr_eps / 3.0F : 1.0F;
    const float noise_floor = sigma_d * 0.29F;
    amb = std::sqrt(std::max(0.0F, amb * amb - noise_floor * noise_floor));
    // Map ambiguity (IRE of leaked luma in the chroma band) onto strength
    // around the reference's --3d value (~1 IRE of leak deserves 0.5;
    // heavy cross-colour saturates at 1.5). NO floor: measured on the
    // clean regression scene, even a small forced strength (plus the
    // anchor it drags in) costs ~3 dB where there is no ambiguity to
    // resolve — amb ~ 0 must mean genuinely OFF.
    ccfg.temporal_strength =
        std::clamp(0.5F * amb - 0.05F, 0.0F, 1.5F);
    if (diag) diag->ambiguity_ire = amb;
  }
  if (ccfg.temporal_strength < 0.0F) ccfg.temporal_strength = 0.0F;

  if (diag) {
    diag->resolved_strength = std::max(0.0F, ccfg.temporal_strength);
    diag->temporal_eps = ccfg.temporal_eps;
    diag->nr_eps = ccfg.nr_eps;
  }

  const bool has_temporal =
      ccfg.temporal_strength > 0.0F && ccfg.cg_iterations > 0;

  // decode_field semantics when the temporal terms are off: ONE pass, no
  // anchor. (The reference only runs passes/anchor inside decode_sequence,
  // i.e. WITH temporal terms; letting the anchor blend fields in "2D" both
  // diverged from the reference and made the supposedly-decoupled fields
  // couple through SynthReference — a data race under the stride-1
  // parallel sweep, and a semantics bug before it was a threading one.)
  const int n_passes = has_temporal ? std::max(1, ccfg.passes) : 1;

  std::mutex mcache_mu;  // guards the shared fast-mode cache during the
                         // colored parallel sweep (see below)

  for (int pass = 0; pass < n_passes; ++pass) {
    std::map<std::pair<int, int>, MotionField> mcache_pass;  // slow mode
    auto& mcache = ccfg.fast ? mcache_fast : mcache_pass;

    // Cache access helpers: lookups/inserts under the mutex, estimation
    // outside it (a duplicated estimate on a race is benign — identical
    // inputs give identical results; corrupting the map would not be).
    auto cache_get = [&](int a, int b, MotionField* out_mf) {
      std::lock_guard<std::mutex> lk(mcache_mu);
      auto it = mcache.find({a, b});
      if (it == mcache.end()) return false;
      *out_mf = it->second;
      return true;
    };
    auto cache_put = [&](int a, int b, const MotionField& mf) {
      std::lock_guard<std::mutex> lk(mcache_mu);
      mcache[{a, b}] = mf;
    };

    auto solve_field = [&](int j) {
      const Plane& S = fields[j].s;
      const ComplexPlane& carrier = fields[j].carrier;
      const int pj = parities[j];

      // ---- pairwise motion for this field's offsets ----------------------
      std::map<int, MotionField> pair_motion;
      if (has_temporal) {
        // Fast mode: full pyramid search only for {-1, +1, +2}; the rest
        // get trajectory-PREDICTED vectors audited by VerifyMotion.
        const std::vector<int> full_os =
            ccfg.fast ? std::vector<int>{-1, 1, 2} : offs;
        for (int o : offs) {
          const int k = j + o;
          if (k < 0 || k >= nf) continue;
          MotionField mf;
          if (cache_get(j, k, &mf)) {
            pair_motion[o] = std::move(mf);
          } else if (!ccfg.trajectory_fit ||
                     std::find(full_os.begin(), full_os.end(), o) !=
                         full_os.end()) {
            mf = EstimateMotion(Ys[k], Ys[j], ccfg.mc_tile, ccfg.mc_search);
            cache_put(j, k, mf);
            pair_motion[o] = std::move(mf);
          }
        }
        if (ccfg.fast && ccfg.trajectory_fit && !pair_motion.empty()) {
          // Per-tile velocity from the offsets we DID search (plain median,
          // parity term removed), used to PREDICT the remaining offsets.
          const int th = pair_motion.begin()->second.dy.height();
          const int tw = pair_motion.begin()->second.dy.width();
          Plane vym0(th, tw), vxm0(th, tw);
          {
            std::vector<float> sy, sx;
            for (int i = 0; i < th * tw; ++i) {
              sy.clear();
              sx.clear();
              for (const auto& [o, mf] : pair_motion) {
                const float ho =
                    (static_cast<float>(parities[j + o]) - pj) / 2.0F;
                sy.push_back((mf.dy[i] - ho) / static_cast<float>(o));
                sx.push_back(mf.dx[i] / static_cast<float>(o));
              }
              vym0[i] = MedianOf(sy);
              vxm0[i] = MedianOf(sx);
            }
          }
          for (int o : offs) {
            const int k = j + o;
            if (pair_motion.count(o) || k < 0 || k >= nf) continue;
            MotionField mf;
            if (cache_get(j, k, &mf)) {
              pair_motion[o] = std::move(mf);
              continue;
            }
            const float ho = (static_cast<float>(parities[k]) - pj) / 2.0F;
            Plane pdy(vym0.height(), vym0.width());
            Plane pdx(vxm0.height(), vxm0.width());
            for (size_t i = 0; i < pdy.size(); ++i) {
              pdy[i] = static_cast<float>(o) * vym0[i] + ho;
              pdx[i] = static_cast<float>(o) * vxm0[i];
            }
            mf = VerifyMotion(Ys[k], Ys[j], pdy, pdx, ccfg.mc_tile);
            cache_put(j, k, mf);
            pair_motion[o] = std::move(mf);
          }
        }
        if (ccfg.trajectory_fit && pair_motion.size() >= 2) {
          // Trajectory-coherent consensus snap (THEORY 9e), with the known
          // half-line parity term h_k = (p_k - p_j)/2 removed before the
          // fit and reinstated after.
          std::vector<OffsetMotion> pm;
          pm.reserve(pair_motion.size());
          for (const auto& [o, mf] : pair_motion) {
            OffsetMotion m;
            m.offset = o;
            m.parity_shift =
                (static_cast<float>(parities[j + o]) - pj) / 2.0F;
            m.field = mf;
            pm.push_back(std::move(m));
          }
          TrajectorySnap(&pm);
          for (OffsetMotion& m : pm) pair_motion[m.offset] = std::move(m.field);
        }
      }

      // ---- neighbour equations -------------------------------------------
      std::vector<NeighborTerm> neighbors;
      // Half-line validity envelope for ODD offsets (opposite parity).
      // An opposite-parity neighbour samples the scene BETWEEN this
      // field's lines: after the integer warp its equation claims the
      // content at y +/- 0.5 lines is the content at y. Wherever the
      // current field has vertical structure that claim is wrong by
      // ~0.5*|dS/dline| — and, crucially, on detail thinner than the
      // frame-line pitch (a 1-frame-line coloured ledge, thin blinds) the
      // opposite parity genuinely does not SEE the feature, so its
      // equation votes "background" with a SMALL residual that the
      // robust weight never gates: measured, 3D made 1-frame-line chroma
      // detail WORSE than 2D (5.2 vs 3.5 IRE chi error) through exactly
      // this hole. The bias is known geometry, so gate it
      // deterministically: wt *= eps^2 / (eps^2 + (0.5*dS/dline)^2),
      // horizontally smoothed. Even offsets are aligned and untouched.
      Plane vgrad;
      if (has_temporal) {
        bool any_odd = false;
        for (int o : offs) {
          const int k = j + o;
          if (k >= 0 && k < nf && ((o & 1) != 0)) any_odd = true;
        }
        if (any_odd) {
          // BASEBAND vertical envelope — NOT the composite's: within a
          // field, consecutive lines are consecutive in SCAN TIME, so the
          // carrier flips 180 deg per FIELD line and the composite of a
          // flat saturated colour alternates sign line-to-line (|dS| ~
          // 2|chi|) — a composite-based gate reads "vertical detail"
          // everywhere chroma is strong and strangles the odd equations
          // globally, chroma-correlated (measured: -8.6 dB on the
          // reference's saturated SMPTE scene). The decoded baseband
          // state (Ys, chis) measures the actual scene structure the
          // opposite parity cannot see. One-sided max diff (a central
          // diff is exactly zero ON a one-row feature), local RMS via
          // blurred squares.
          const int hh = S.height();
          const int ww = S.width();
          Plane gmag(hh, ww);
          for (int y = 0; y < hh; ++y) {
            const int ym = std::max(0, y - 1);
            const int yp = std::min(hh - 1, y + 1);
            for (int x = 0; x < ww; ++x) {
              const float dyu = Ys[j].at(y, x) - Ys[j].at(ym, x);
              const float dyd = Ys[j].at(yp, x) - Ys[j].at(y, x);
              const float dcu = std::abs(chis[j].at(y, x) - chis[j].at(ym, x));
              const float dcd = std::abs(chis[j].at(yp, x) - chis[j].at(y, x));
              const float mu = dyu * dyu + dcu * dcu;
              const float md = dyd * dyd + dcd * dcd;
              gmag.at(y, x) = std::max(mu, md);
            }
          }
          // HORIZONTAL-only smoothing          // HORIZONTAL-only smoothing: a 2D blur dilutes a one-row
          // feature's vertical footprint by ~2.5x and weakens the gate on
          // exactly the detail it protects; vertical coverage already
          // comes free from the one-sided max (the feature row and both
          // its neighbours each carry a full-amplitude diff).
          vgrad = Plane(hh, ww);
          for (int y = 0; y < hh; ++y) {
            float acc = 0.0F;
            for (int x = 0; x < std::min(5, ww); ++x) acc += gmag.at(y, x);
            for (int x = 0; x < ww; ++x) {
              const int lo = x - 2, hi = x + 2;
              if (x > 0) {
                if (hi < ww) acc += gmag.at(y, hi);
                if (lo - 1 >= 0) acc -= gmag.at(y, lo - 1);
              }
              const int n_taps = std::min(hi, ww - 1) - std::max(lo, 0) + 1;
              vgrad.at(y, x) = std::sqrt(acc / static_cast<float>(n_taps));
            }
          }
          // Noise-floor the envelope: at pass 0 the state is the
          // holographic INIT, whose vertical cross-colour/noise gives a
          // non-zero envelope everywhere — ungated that read as "detail"
          // and cost the reference chart 4 dB of 3D gain. The floor is
          // self-calibrated as the MEDIAN envelope (background dominates
          // by area): background -> gate ~ 1, true structure >> median
          // -> gate bites.
          {
            std::vector<float> sample;
            sample.reserve(vgrad.size() / 7 + 1);
            for (size_t i = 0; i < vgrad.size(); i += 7)
              sample.push_back(vgrad[i]);
            const float med = MedianOf(std::move(sample));
            const float fl2 = 2.25F * med * med;  // (1.5 * median)^2
            for (size_t i = 0; i < vgrad.size(); ++i)
              vgrad[i] = std::sqrt(
                  std::max(0.0F, vgrad[i] * vgrad[i] - fl2));
          }
        }
      }
      if (has_temporal) {
        for (int o : offs) {
          const int k = j + o;
          if (k < 0 || k >= nf) continue;
          // Raw integer warp for the equations: EXACT for aligned static
          // content; residual half-line bias on odd offsets is per-pixel
          // gated. (An envelope-resampled variant was tried in the
          // reference and rejected: its 1-line comb leaks vertical luma
          // gradients into chroma.)
          NeighborRawState state;
          state.luma = Ys[k];
          state.composite = fields[k].s;
          state.carrier = fields[k].carrier;
          const MotionField& mo = pair_motion.at(o);
          const bool gate_on =
              ccfg.coherence_gate > 0.0F && !(ccfg.fast && pass == 0);
          // Share ONE per-pixel vector interpolation between the raw warps
          // and the gate's chi warp (it used to be computed twice per
          // neighbour when the gate was active).
          PerPixelMotion vpx_shared;
          if (gate_on) {
            vpx_shared = VectorsPerPixel(mo.dy, mo.dx, ccfg.mc_tile,
                                         Ys[j].height(), Ys[j].width());
          }
          MotionCompensatedResult mc = MotionCompensatePrev(
              state, Ys[j], ccfg.mc_tile, ccfg.mc_search, &mo, ccfg.fast,
              gate_on ? &vpx_shared : nullptr);
          if (gate_on) {
            // Fast: coherence only from pass 1 — at pass 0 the chi fields
            // are still inits and the measurement is barely informative.
            // InSAR coherence between the current chroma and the warped
            // neighbour chroma, floored so grey content (|chi| ~ 0, gamma
            // = pure noise) keeps the equation's LUMA benefit.
            //
            // REFERENCE BUG, fixed here (see docs/PORTING.md section 11 for
            // the numeric demonstration): the bilinear-warp convention
            // assumes the motion carries NO half-line parity component
            // (row_offset supplies it), which held when the estimates were
            // margin-zeroed on static content — but the trajectory snap
            // (THEORY 9e) deliberately RE-INTRODUCES h_k = (p_k - p_j)/2
            // into the pairwise vectors, so adding row_offset on top
            // double-compensated: the gate compared chi fields shifted by
            // a FULL line on exactly the static consensus tiles it exists
            // to protect. With the snapped/subpixel motion already
            // carrying the total inter-grid displacement, the correct warp
            // is simply sy = y - dy: row_offset must be 0 here. (Worst
            // case moves from 1.0 line, on the dominant snapped case, to
            // 0.5 line on margin-zeroed non-consensus tiles only.)
            const float row_off = 0.0F;
            const int hh = Ys[j].height();
            const int ww = Ys[j].width();
            const PerPixelMotion& vpx = vpx_shared;
            Plane re(hh, ww), im(hh, ww);
            for (size_t i = 0; i < chis[k].size(); ++i) {
              re[i] = chis[k][i].real();
              im[i] = chis[k][i].imag();
            }
            const Plane wre = WarpBilinearTiles(re, vpx, row_off, hh, ww);
            const Plane wim = WarpBilinearTiles(im, vpx, row_off, hh, ww);
            ComplexPlane cw(hh, ww);
            for (size_t i = 0; i < cw.size(); ++i)
              cw[i] = Complex{wre[i], wim[i]};
            const Plane gamma = ComplexCoherence(chis[j], cw, 6);
            const float a = ccfg.coherence_gate;
            for (size_t i = 0; i < mc.confidence.size(); ++i)
              mc.confidence[i] *= (1.0F - a) + a * gamma[i];
          }
          if ((o & 1) != 0 && !vgrad.empty()) {
            // ENVELOPE gate, floored. The residual cannot do this job:
            // at a chroma feature the odd equation's residual OSCILLATES
            // with x (|dchi * cos|), so its robust weight gates the
            // equation at some columns and lets confident wrong votes
            // through at the cosine zeros — only the phase-independent
            // ENVELOPE catches all of them (this is why the residual-
            // dependent variant measured worse). The 0.25 floor keeps a
            // quarter of the weight alive everywhere: on step-edge-heavy
            // content (the reference's saturated chart) the odd
            // equations are biased but informative and hard-gating them
            // cost 4.4 dB of 3D gain; a floored gate keeps ~75% of the
            // thin-detail protection while leaving the solve enough
            // equation mass to keep absorbing edge bias.
            const float e2 = ccfg.temporal_eps * ccfg.temporal_eps;
            for (size_t i = 0; i < mc.confidence.size(); ++i) {
              const float b = vgrad[i];
              mc.confidence[i] *=
                  std::max(0.35F, e2 / (e2 + b * b));
            }
          }
          neighbors.push_back(
              NeighborTerm{std::move(mc.composite), std::move(mc.carrier),
                           std::move(mc.confidence)});
        }
      }

      if (pass == 0 && ccfg.psi_init && !neighbors.empty()) {
        chis[j] = PsiClosedForm(S, carrier, neighbors, chis[j]);
      }
      if (pass >= 1 && ccfg.nr_anchor > 0.0F && nf > 1) {
        const AnchorTerm anchor = SynthReference(
            j, Ys, chis, fields, ccfg, parities,
            ccfg.fast ? &mcache_fast : nullptr,
            ccfg.fast ? &mcache_mu : nullptr);
        RefineResult r = VariationalRefineJoint(S, carrier, Ys[j], chis[j],
                                                ccfg, neighbors, &anchor);
        Ys[j] = std::move(r.luma);
        chis[j] = std::move(r.chroma);
      } else {
        RefineResult r = VariationalRefine(S, carrier, chis[j], ccfg, neighbors);
        Ys[j] = std::move(r.luma);
        chis[j] = std::move(r.chroma);
      }
    };

    if (ccfg.fast || !has_temporal) {
      // COLORED Gauss-Seidel sweep (the fast-mode math change that unlocks
      // field-level parallelism): field j only READS fields within
      // `reach` = max(largest temporal offset, nr_radius) of itself (the
      // neighbour equations, the coherence gate, the anchor blend), so
      // fields spaced `reach + 1` apart share no read/write dependency and
      // can be solved CONCURRENTLY. Sweeping color-by-color keeps the
      // Gauss-Seidel character (each color reads the freshest available
      // state of the others) — same fixed point as the sequential sweep,
      // a different but convergent iteration trajectory; measured within
      // the fast-mode quality envelope on the regression scene. The
      // sequential sweep stays the slow-mode behaviour, bit-faithful to
      // the reference. Inside the parallel region the solvers' own OpenMP
      // loops deactivate (no nested parallelism), which is the intended
      // trade: one field per core beats one memory-bound loop across all
      // cores.
      int max_off = 0;
      for (int o : offs) max_off = std::max(max_off, std::abs(o));
      const int reach =
          std::max(max_off, ccfg.nr_anchor > 0.0F ? ccfg.nr_radius : 0);
      // With no temporal terms the fields are fully decoupled: one color,
      // every field concurrent (this is the per-field 2D decode path).
      const int stride = has_temporal ? reach + 1 : 1;
      for (int color = 0; color < stride; ++color) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int j = color; j < nf; j += stride) solve_field(j);
      }
    } else {
      for (int j = 0; j < nf; ++j) solve_field(j);
    }
  }

  for (int j = 0; j < nf; ++j) {
    out[j].chroma = std::move(chis[j]);
    if (ccfg.output_fidelity) {
      // Purist output: chroma keeps the fully guided separation, luma
      // reverts to exact data fidelity (no NR in the deliverable).
      out[j].luma = Plane(fields[j].s.height(), fields[j].s.width());
      for (size_t i = 0; i < out[j].luma.size(); ++i)
        out[j].luma[i] = fields[j].s[i] -
                         (out[j].chroma[i] * fields[j].carrier[i]).real();
    } else {
      out[j].luma = std::move(Ys[j]);
    }
  }
  return out;
}

}  // namespace hvd
