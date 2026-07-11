// SPDX-License-Identifier: GPL-3.0-or-later

#include "engine/holographic_init.h"

#include <cmath>
#include <vector>

#include "engine/gradients.h"

namespace hvd {

namespace {

// numpy.fft.fftfreq(n, d): [0, 1, ..., ceil(n/2)-1, -floor(n/2), ..., -1] / (n*d)
std::vector<float> FftFreq(int n, double d) {
  std::vector<float> f(n);
  const double inv = 1.0 / (static_cast<double>(n) * d);
  const int half = (n % 2 == 0) ? n / 2 : (n + 1) / 2;  // count of non-negative
  for (int i = 0; i < n; ++i) {
    const int k = (i < half) ? i : i - n;
    f[i] = static_cast<float>(k * inv);
  }
  return f;
}

// Separable 2-D Gaussian low-pass response (the "hologram crop"), built from
// per-axis cutoffs. cutoff_x in Hz; cutoff_y in cycles/line.
Plane GaussianLpf(int h, int w, const FieldGeometry& g, float lpf_h_mhz,
                  float lpf_v_cph) {
  const std::vector<float> fx = FftFreq(w, 1.0 / g.sample_rate);  // Hz
  const std::vector<float> fy = FftFreq(h, 1.0);                  // cyc/line
  const float cutoff_x = lpf_h_mhz * 1.0e6F;
  const float cutoff_y = lpf_v_cph / (2.0F * static_cast<float>(h));

  std::vector<float> gx(w), gy(h);
  for (int x = 0; x < w; ++x) {
    const float r = fx[x] / cutoff_x;
    gx[x] = std::exp(-0.5F * r * r);
  }
  for (int y = 0; y < h; ++y) {
    const float r = fy[y] / cutoff_y;
    gy[y] = std::exp(-0.5F * r * r);
  }
  Plane g_out(h, w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) g_out.at(y, x) = gy[y] * gx[x];
  return g_out;
}

// 1-D box filter of length (2r+1), zero-padded, 'same' output, normalised by the
// full length — matching numpy.convolve(..., 'same') on a 1/(2r+1) kernel
// (edges are attenuated by the zero padding).
void BoxBlur1DRows(Plane* a, int r) {
  const int h = a->height();
  const int w = a->width();
  const float norm = 1.0F / static_cast<float>(2 * r + 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int x = 0; x < w; ++x) {
    // col is declared INSIDE the loop so each parallel iteration (each
    // thread, for whichever columns it's assigned) gets its own buffer —
    // sharing one `col` across iterations the way the original sequential
    // version did would be a data race once this loop is split across
    // threads.
    std::vector<float> col(h);
    for (int y = 0; y < h; ++y) col[y] = a->at(y, x);
    for (int y = 0; y < h; ++y) {
      float acc = 0.0F;
      for (int k = -r; k <= r; ++k) {
        const int yy = y + k;
        if (yy >= 0 && yy < h) acc += col[yy];
      }
      a->at(y, x) = acc * norm;
    }
  }
}

void BoxBlur1DCols(Plane* a, int r) {
  const int h = a->height();
  const int w = a->width();
  const float norm = 1.0F / static_cast<float>(2 * r + 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < h; ++y) {
    // Same per-iteration-buffer reasoning as BoxBlur1DRows above.
    std::vector<float> row(w);
    for (int x = 0; x < w; ++x) row[x] = a->at(y, x);
    for (int x = 0; x < w; ++x) {
      float acc = 0.0F;
      for (int k = -r; k <= r; ++k) {
        const int xx = x + k;
        if (xx >= 0 && xx < w) acc += row[xx];
      }
      a->at(y, x) = acc * norm;
    }
  }
}

Plane BoxBlur(Plane a, int r) {
  BoxBlur1DRows(&a, r);
  BoxBlur1DCols(&a, r);
  return a;
}

// Re[chi * carrier] as a real plane.
Plane RealOfProduct(const ComplexPlane& chi, const ComplexPlane& carrier) {
  Plane out(chi.height(), chi.width());
  const long n = static_cast<long>(chi.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long i = 0; i < n; ++i) out[i] = (chi[i] * carrier[i]).real();
  return out;
}

// Residual luma Y = S - Re[chi * carrier].
Plane ResidualLuma(const Plane& s, const ComplexPlane& chi,
                   const ComplexPlane& carrier) {
  Plane y(s.height(), s.width());
  const long n = static_cast<long>(s.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long i = 0; i < n; ++i)
    y[i] = s[i] - (chi[i] * carrier[i]).real();
  return y;
}

// Edge-energy map E = box_blur(|Dx Y| + |Dy Y|, r=3), used as the inverse
// blend weight (smoother residual luma => higher weight).
Plane EdgeEnergy(const Plane& y) {
  const Plane gx = Dx(y);
  const Plane gy = Dy(y);
  Plane e(y.height(), y.width());
  const long n = static_cast<long>(y.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long i = 0; i < n; ++i)
    e[i] = std::fabs(gx[i]) + std::fabs(gy[i]);
  return BoxBlur(std::move(e), 3);
}

}  // namespace

HoloInit HolographicInit(const Plane& s, const ComplexPlane& carrier,
                         const FieldGeometry& g, const HvdConfig& cfg,
                         Fft2d* fft) {
  const int h = s.height();
  const int w = s.width();
  const long n_total = static_cast<long>(s.size());

  // demod = S * conj(carrier); shift the chroma sideband to DC.
  ComplexPlane demod(h, w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long i = 0; i < n_total; ++i) demod[i] = s[i] * std::conj(carrier[i]);
  const ComplexPlane spectrum = fft->Forward(demod);

  // Accumulate the per-pixel weighted blend of the crop variants.
  ComplexPlane chi_num(h, w);
  Plane w_sum(h, w);

  // Two complementary anisotropic crops: (narrow-x, wide-y) and (wide-x,
  // narrow-y). Values copied verbatim from the reference.
  // NOTE: this outer loop (2 iterations) stays sequential on purpose — the
  // FFT plan cache (Fft2d::PlanFor) isn't safe to look up/insert into
  // concurrently from multiple threads sharing one Fft2d, and chi_num/
  // w_sum are shared accumulators across variants. The per-pixel loops
  // WITHIN each variant below are what's parallelised instead — that's
  // most of the actual elementwise work anyway.
  const float variants[2][2] = {{0.8F, 120.0F}, {1.8F, 30.0F}};
  for (const auto& v : variants) {
    const Plane kernel = GaussianLpf(h, w, g, v[0], v[1]);
    ComplexPlane cropped(h, w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n_total; ++i) cropped[i] = spectrum[i] * kernel[i];
    ComplexPlane chi_v = fft->Inverse(cropped);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n_total; ++i) chi_v[i] *= 2.0F;

    const Plane y_v = ResidualLuma(s, chi_v, carrier);
    const Plane e = EdgeEnergy(y_v);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n_total; ++i) {
      const float weight = 1.0F / (e[i] + 0.5F);
      chi_num[i] += chi_v[i] * weight;
      w_sum[i] += weight;
    }
  }

  // Optional third variant: spectral-symmetry certified chroma ("Transform
  // NTSC, repaired"). Off by default; see the reference for the rationale.
  if (cfg.symmetry_variant) {
    // sym[k] = min(|D[k]|, |D[-k reflected]|) / (|D[k]| + eps), a lower bound
    // of chroma that luma almost never fakes. The point-reflection is
    // roll(roll(D[::-1,::-1], 1, 0), 1, 1) so bin k pairs with bin -k.
    Plane mag(h, w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n_total; ++i) mag[i] = std::abs(spectrum[i]);
    ComplexPlane certified(h, w);
    const Plane kernel = GaussianLpf(h, w, g, 1.3F, 60.0F);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int ry = (h - y) % h;  // reflect+shift by 1 => index (h-y) mod h
        const int rx = (w - x) % w;
        const float m = mag.at(y, x);
        const float mr = mag.at(ry, rx);
        const float sym = std::min(m, mr) / (m + 1e-6F);
        certified.at(y, x) = spectrum.at(y, x) * sym * kernel.at(y, x);
      }
    }
    ComplexPlane chi_s = fft->Inverse(certified);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n_total; ++i) chi_s[i] *= 2.0F;
    const Plane y_s = ResidualLuma(s, chi_s, carrier);
    const Plane e = EdgeEnergy(y_s);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n_total; ++i) {
      const float weight = 1.0F / (e[i] + 0.5F);
      chi_num[i] += chi_s[i] * weight;
      w_sum[i] += weight;
    }
  }

  HoloInit out;
  out.chroma = ComplexPlane(h, w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long i = 0; i < n_total; ++i)
    out.chroma[i] = chi_num[i] / w_sum[i];
  out.luma = ResidualLuma(s, out.chroma, carrier);
  return out;
}

}  // namespace hvd
