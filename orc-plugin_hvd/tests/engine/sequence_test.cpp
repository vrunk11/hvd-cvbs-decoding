// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests for engine/sequence.{h,cpp} — the decode_sequence port. Mirrors the
// reference's own 3D assertion ("3D must beat 2D on static content",
// run_tests.py) at the engine level, FFT-free via the WithInits seam, plus
// the invariants the module promises:
//   * output_fidelity: S - Y - Re[chi * carrier] == 0 exactly per field
//   * pass >= 2 anchored (synth-reference + joint) path runs and stays sane
//   * fast mode (shared cache, predicted+verified ME, deferred coherence)
//     stays close to the slow decode
//   * psi_init runs and does not degrade the static solve

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "check.h"
#include "engine/ntsc_geometry.h"
#include "engine/sequence.h"
#include "engine/temporal.h"

namespace {

using hvd::Complex;
using hvd::ComplexPlane;
using hvd::DecodedField;
using hvd::FieldGeometry;
using hvd::FieldObs;
using hvd::HvdConfig;
using hvd::Plane;

constexpr int kFrameLines = 96;  // frame-grid truth; fields see half each
constexpr int kW = 128;
constexpr int kFields = 8;  // 4 frames — enough for f+-3 and nr_radius 2
constexpr float kPi = 3.14159265358979323846F;

struct Scene {
  std::vector<FieldObs> fields;
  std::vector<ComplexPlane> chi_truth;  // per field, field grid
  std::vector<DecodedField> inits;
};

// Static scene on the FRAME grid, sampled by each field at its parity's
// lines (the half-line geometry the parity terms exist for). Carrier phase
// advances 3/4 cycle per field (adjacent |dc| = sqrt(2)) and flips 180 deg
// per frame, the NTSC relations the pipeline's comments rely on — here they
// are just the synthesis; the decoder measures nothing but the given
// carrier, as in real use.
Scene MakeStaticScene(float noise_ire, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nz(0.0F, std::max(noise_ire, 1e-6F));

  Plane YF(kFrameLines, kW);
  ComplexPlane CF(kFrameLines, kW);
  for (int y = 0; y < kFrameLines; ++y) {
    for (int x = 0; x < kW; ++x) {
      // The luma carries a modulated component AT the subcarrier frequency
      // (pi/2 rad/px at 4fsc): luma that looks like chroma. This is the
      // content class the pipeline's validated gains come from — a single
      // 2D data equation per pixel cannot resolve the Y/C ambiguity there
      // (the prior mis-attributes it: structural cross-colour), while the
      // 3D phase diversity of the neighbour equations resolves it exactly.
      // Without this term the scene has no ambiguity, 2D is already
      // near-perfect, and the neighbour equations can only add noise —
      // measured -4 dB, on the REFERENCE too (its own regression scene is
      // an encoded SMPTE chart, ambiguity-dominated like this one).
      YF.at(y, x) = 30.0F + 22.0F * std::sin(0.06F * (y + 2 * x)) +
                    14.0F * std::cos((kPi / 2.0F) * x + 0.4F * y) *
                        std::sin(0.15F * y);
      const bool in = (x > 40 && x < 90 && y > 24 && y < 72);
      CF.at(y, x) = in ? Complex{12.0F, -8.0F} : Complex{3.0F, 2.0F};
    }
  }

  Scene sc;
  const int lines = kFrameLines / 2;
  for (int f = 0; f < kFields; ++f) {
    const int parity = f % 2;
    FieldObs obs;
    obs.parity = parity;
    obs.s = Plane(lines, kW);
    obs.carrier = ComplexPlane(lines, kW);
    ComplexPlane chi_t(lines, kW);
    for (int r = 0; r < lines; ++r) {
      const int fy = 2 * r + parity;
      for (int x = 0; x < kW; ++x) {
        const float phi = (kPi / 2.0F) * x + kPi * r +
                          static_cast<float>(f) * (3.0F * kPi / 2.0F);
        const Complex c{std::cos(phi), std::sin(phi)};
        obs.carrier.at(r, x) = c;
        chi_t.at(r, x) = CF.at(fy, x);
        obs.s.at(r, x) = YF.at(fy, x) + (CF.at(fy, x) * c).real() +
                         (noise_ire > 0.0F ? nz(rng) : 0.0F);
      }
    }
    // Crude corrupted init (cross-colour-like), the stand-in for the
    // holographic init — keeps the test FFT-free.
    DecodedField init;
    init.chroma = ComplexPlane(lines, kW);
    for (int r = 0; r < lines; ++r) {
      for (int x = 0; x < kW; ++x) {
        const float e = 4.0F * std::sin(0.8F * x) * std::cos(0.6F * r);
        init.chroma.at(r, x) = chi_t.at(r, x) + Complex{e, -e};
      }
    }
    sc.chi_truth.push_back(std::move(chi_t));
    sc.fields.push_back(std::move(obs));
    sc.inits.push_back(std::move(init));
  }
  return sc;
}

double ChromaPsnr(const std::vector<DecodedField>& dec,
                  const std::vector<ComplexPlane>& truth) {
  double m = 0.0;
  long n = 0;
  for (size_t j = 0; j < dec.size(); ++j) {
    for (size_t i = 0; i < dec[j].chroma.size(); ++i) {
      m += std::norm(dec[j].chroma[i] - truth[j][i]);
      ++n;
    }
  }
  return 10.0 * std::log10(1e4 / (m / static_cast<double>(n)));
}

bool AllFinite(const std::vector<DecodedField>& dec) {
  for (const DecodedField& d : dec) {
    for (size_t i = 0; i < d.luma.size(); ++i)
      if (!std::isfinite(d.luma[i])) return false;
    for (size_t i = 0; i < d.chroma.size(); ++i)
      if (!std::isfinite(d.chroma[i].real()) ||
          !std::isfinite(d.chroma[i].imag()))
        return false;
  }
  return true;
}

}  // namespace

void RunTests() {
  FieldGeometry g;  // unused by the WithInits driver, but part of the API
  const Scene sc = MakeStaticScene(/*noise_ire=*/0.8F, /*seed=*/7);

  HvdConfig base;
  base.cg_iterations = 40;  // keep the test quick

  // --- per-field 2D baseline (temporal off) --------------------------------
  HvdConfig c2d = base;
  c2d.temporal_strength = 0.0F;
  const std::vector<DecodedField> dec2d =
      hvd::DecodeFieldWindowWithInits(sc.fields, sc.inits, g, c2d);
  CHECK(AllFinite(dec2d));
  const double p2d = ChromaPsnr(dec2d, sc.chi_truth);

  // --- 3D, the reference's own regression shape ----------------------------
  HvdConfig c3d = base;
  c3d.temporal_strength = 2.0F;  // run_tests.py's value for the static test
  const std::vector<DecodedField> dec3d =
      hvd::DecodeFieldWindowWithInits(sc.fields, sc.inits, g, c3d);
  CHECK(AllFinite(dec3d));
  const double p3d = ChromaPsnr(dec3d, sc.chi_truth);
  std::printf("  sequence: 2D=%.2f dB  3D=%.2f dB\n", p2d, p3d);
  CHECK(p3d > p2d);  // "regression: 3D worse than 2D on static content"

  // --- output_fidelity: the purist luma identity is exact ------------------
  for (size_t j = 0; j < dec3d.size(); ++j) {
    float max_err = 0.0F;
    for (size_t i = 0; i < dec3d[j].luma.size(); ++i) {
      const float recon =
          dec3d[j].luma[i] +
          (dec3d[j].chroma[i] * sc.fields[j].carrier[i]).real();
      max_err = std::max(max_err, std::fabs(sc.fields[j].s[i] - recon));
    }
    CHECK(max_err < 1e-3F);
  }

  // --- anchored multi-pass path (synth_reference + joint solve) ------------
  {
    HvdConfig cfg = c3d;
    cfg.temporal_strength = 0.5F;  // run_tests.py's anchored-mode values
    cfg.passes = 2;
    const std::vector<DecodedField> dec =
        hvd::DecodeFieldWindowWithInits(sc.fields, sc.inits, g, cfg);
    CHECK(AllFinite(dec));
    const double p = ChromaPsnr(dec, sc.chi_truth);
    std::printf("  anchored 2-pass: %.2f dB\n", p);
    CHECK(p > p2d);  // the anchor must not undo the 3D gain on static
  }

  // --- fast mode: same algorithm, close result -----------------------------
  {
    HvdConfig cfg = c3d;
    cfg.fast = true;
    cfg.passes = 2;  // exercise the shared cache + deferred coherence
    HvdConfig slow = cfg;
    slow.fast = false;
    const std::vector<DecodedField> df =
        hvd::DecodeFieldWindowWithInits(sc.fields, sc.inits, g, cfg);
    const std::vector<DecodedField> ds =
        hvd::DecodeFieldWindowWithInits(sc.fields, sc.inits, g, slow);
    CHECK(AllFinite(df));
    const double pf = ChromaPsnr(df, sc.chi_truth);
    const double ps = ChromaPsnr(ds, sc.chi_truth);
    std::printf("  fast=%.2f dB  slow=%.2f dB\n", pf, ps);
    CHECK(pf > ps - 0.5);  // reference bound: never worse than ~0.2 dB
  }

  // --- drizzle output mode --------------------------------------------------
  {
    HvdConfig cfg = c3d;
    const std::vector<DecodedField> dec =
        hvd::DecodeFieldWindowWithInits(sc.fields, sc.inits, g, cfg);
    std::vector<Plane> Ys;
    std::vector<ComplexPlane> chis;
    std::vector<int> parities;
    for (size_t j = 0; j < dec.size(); ++j) {
      Ys.push_back(dec[j].luma);
      chis.push_back(dec[j].chroma);
      parities.push_back(sc.fields[j].parity);
    }
    const int scale = 2;
    const hvd::DrizzleResult dz =
        hvd::DrizzleFrame(/*j0=*/2, Ys, chis, parities, cfg, scale);
    const int L = Ys[2].height();
    CHECK(dz.luma.height() == 2 * L * scale);
    CHECK(dz.luma.width() == Ys[2].width());
    // Against the ANALYTIC truth evaluated at the fine rows (the scene is
    // continuous in y, so super-resolved rows have an exact reference).
    double m = 0.0;
    long n = 0;
    bool finite = true;
    for (int r = 0; r < dz.luma.height(); ++r) {
      const float fy = static_cast<float>(r) / static_cast<float>(scale);
      for (int x = 8; x < dz.luma.width() - 8; ++x) {
        if (!std::isfinite(dz.luma.at(r, x))) finite = false;
        const float yt =
            30.0F + 22.0F * std::sin(0.06F * (fy + 2 * x)) +
            14.0F * std::cos((kPi / 2.0F) * x + 0.4F * fy) *
                std::sin(0.15F * fy);
        const double d = dz.luma.at(r, x) - yt;
        m += d * d;
        ++n;
      }
    }
    CHECK(finite);
    const double psnr = 10.0 * std::log10(1e4 / (m / n));
    std::printf("  drizzle luma vs analytic fine truth: %.2f dB\n", psnr);
    CHECK(psnr > 24.0);  // super-resolved rows track the continuous scene
  }

  // --- coherence-gate geometry (the reference row_offset bug) ---------------
  // Static cross-parity fields with vertically-varying chroma; the
  // trajectory-snapped motion carries h_k, so the gate warp must NOT add
  // row_offset on top (double-compensation = a full-line shift). Locks the
  // fixed convention: gamma on static content stays ~1.
  {
    const int L = 64, W = 96;
    ComplexPlane chij(L, W), chik(L, W);
    for (int r = 0; r < L; ++r) {
      for (int x = 0; x < W; ++x) {
        auto C = [&](float fy) {
          return Complex{10.0F * std::sin(0.5F * fy) +
                             3.0F * std::cos(0.13F * x),
                         -6.0F * std::cos(0.5F * fy)};
        };
        chij.at(r, x) = C(2.0F * r);         // parity 0
        chik.at(r, x) = C(2.0F * r + 1.0F);  // parity 1
      }
    }
    const int tile = 32;
    const int th = (L + tile - 1) / tile, tw = (W + tile - 1) / tile;
    Plane dy(th, tw, 0.5F), dx(th, tw, 0.0F);  // snapped: h_k = +0.5
    const hvd::PerPixelMotion v = hvd::VectorsPerPixel(dy, dx, tile, L, W);
    Plane re(L, W), im(L, W);
    for (size_t i = 0; i < chik.size(); ++i) {
      re[i] = chik[i].real();
      im[i] = chik[i].imag();
    }
    const Plane wr = hvd::WarpBilinearTiles(re, v, 0.0F, L, W);
    const Plane wi = hvd::WarpBilinearTiles(im, v, 0.0F, L, W);
    ComplexPlane cw(L, W);
    for (size_t i = 0; i < cw.size(); ++i) cw[i] = Complex{wr[i], wi[i]};
    const Plane gamma = hvd::ComplexCoherence(chij, cw, 6);
    double m = 0.0;
    long n = 0;
    for (int r = 6; r < L - 6; ++r)
      for (int x = 6; x < W - 6; ++x) {
        m += gamma.at(r, x);
        ++n;
      }
    CHECK(m / n > 0.99);  // buggy double-compensation measured ~0.97 here
  }

  // --- psi_init on the static scene ----------------------------------------
  {
    HvdConfig cfg = c3d;
    cfg.psi_init = true;
    const std::vector<DecodedField> dec =
        hvd::DecodeFieldWindowWithInits(sc.fields, sc.inits, g, cfg);
    CHECK(AllFinite(dec));
    const double p = ChromaPsnr(dec, sc.chi_truth);
    std::printf("  psi_init: %.2f dB\n", p);
    CHECK(p > p3d - 0.5);  // must not degrade the static solve
  }
}

TEST_MAIN()
