// SPDX-License-Identifier: GPL-3.0-or-later

#include "engine/variational.h"

#include <algorithm>
#include <cmath>

#include "engine/gradients.h"

namespace hvd {

namespace {

// ---- small elementwise helpers (kept local; they read like the NumPy) -----

// Y = S - Re[chi * carrier].
Plane Luma(const Plane& s, const ComplexPlane& chi,
           const ComplexPlane& carrier) {
  Plane y(s.height(), s.width());
  for (size_t i = 0; i < s.size(); ++i)
    y[i] = s[i] - (chi[i] * carrier[i]).real();
  return y;
}

// Elementwise real product a .* b (both real planes).
Plane Mul(const Plane& a, const Plane& b) {
  Plane out(a.height(), a.width());
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] * b[i];
  return out;
}

// Elementwise scale of a complex plane by a real plane: a .* w.
ComplexPlane MulReal(const ComplexPlane& a, const Plane& w) {
  ComplexPlane out(a.height(), a.width());
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] * w[i];
  return out;
}

// Charbonnier diffusivity  eps / sqrt(g^2 + eps^2), applied elementwise to a
// gradient plane. Returns a weight plane in (0, 1].
Plane CharbonnierWeight(const Plane& g, float eps) {
  Plane w(g.height(), g.width());
  const float e2 = eps * eps;
  for (size_t i = 0; i < g.size(); ++i)
    w[i] = eps / std::sqrt(g[i] * g[i] + e2);
  return w;
}

// Coupled chroma diffusivity  eps_c / sqrt(c^2 + k*gY^2 + eps_c^2), with c the
// magnitude of the chroma gradient and gY the co-located luma gradient.
Plane CoupledWeight(const Plane& c_abs, const Plane& g_luma, float eps_c,
                    float k) {
  Plane w(c_abs.height(), c_abs.width());
  const float e2 = eps_c * eps_c;
  for (size_t i = 0; i < c_abs.size(); ++i) {
    const float denom = c_abs[i] * c_abs[i] + k * g_luma[i] * g_luma[i] + e2;
    w[i] = eps_c / std::sqrt(denom);
  }
  return w;
}

// |Dx chi| etc.: magnitude of a complex plane, elementwise.
Plane Magnitude(const ComplexPlane& a) {
  Plane out(a.height(), a.width());
  for (size_t i = 0; i < a.size(); ++i) out[i] = std::abs(a[i]);
  return out;
}

// sum(|g|^2) over a complex plane (the real CG residual norm).
double SumAbs2(const ComplexPlane& g) {
  double acc = 0.0;
  const long n = static_cast<long>(g.size());
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : acc) schedule(static)
#endif
  for (long i = 0; i < n; ++i)
    acc += static_cast<double>(g[i].real()) * g[i].real() +
           static_cast<double>(g[i].imag()) * g[i].imag();
  return acc;
}

// Re(sum(conj(a) * b)) over complex planes.
double DotReal(const ComplexPlane& a, const ComplexPlane& b) {
  double acc = 0.0;
  const long n = static_cast<long>(a.size());
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : acc) schedule(static)
#endif
  for (long i = 0; i < n; ++i)
    acc += static_cast<double>(a[i].real()) * b[i].real() +
           static_cast<double>(a[i].imag()) * b[i].imag();
  return acc;
}

}  // namespace

RefineResult VariationalRefine(const Plane& s, const ComplexPlane& carrier,
                               const ComplexPlane& chi0, const HvdConfig& cfg) {
  const float eps = cfg.charbonnier_eps;
  const float eps_c = cfg.chroma_eps;
  const float mu_h = cfg.lambda_c * cfg.chroma_aniso;  // chroma broader in x
  const float mu_v = cfg.lambda_c;
  const float k = cfg.structure_coupling;

  const int irls_outer = std::max(1, cfg.irls_outer);
  const int n_inner = std::max(1, cfg.cg_iterations / irls_outer);

  const int H = s.height();
  const int W = s.width();
  const size_t N = s.size();

  ComplexPlane chi = chi0;  // working copy

  // ---- Workspace: every frame-sized buffer the hot loops need, allocated
  // ONCE here instead of once per grad()/curv() call. The previous version
  // allocated ~15 frame-sized planes per CG iteration (~900 allocations per
  // frame at cg_iterations=60), which dominated the runtime. The maths below
  // is unchanged; only the memory traffic is.
  Plane wx(H, W), wy(H, W), wcx(H, W), wcy(H, W);
  Plane yc(H, W), tmpr1(H, W), tmpr2(H, W), img(H, W);
  Plane gxY(H, W), gyY(H, W), magx(H, W), magy(H, W);
  ComplexPlane tmpc1(H, W), tmpc2(H, W), tmpc3(H, W), cprior(H, W);
  ComplexPlane g(H, W), g_new(H, W), d(H, W);
  Plane dy(H, W), dxdy(H, W), dydy(H, W);
  ComplexPlane dxc(H, W), dyc(H, W);

  // g = -[2 (DxT(wx Dx Yc) + DyT(wy Dy Yc))] * conj(carrier)
  //     + 2 (mu_h DxT(wcx Dx chi) + mu_v DyT(wcy Dy chi))
  auto grad = [&](const ComplexPlane& c, ComplexPlane& out) {
    // Yc = S - Re[c * carrier]
    const long n = static_cast<long>(N);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) yc[i] = s[i] - (c[i] * carrier[i]).real();

    // img = 2 * (DxT(wx .* Dx(yc)) + DyT(wy .* Dy(yc)))
    DxInto(yc, tmpr1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) tmpr1[i] *= wx[i];
    DxTInto(tmpr1, img);

    DyInto(yc, tmpr2);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) tmpr2[i] *= wy[i];
    DyTInto(tmpr2, tmpr1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) img[i] = 2.0F * (img[i] + tmpr1[i]);

    // cprior = 2 * (mu_h * DxT(wcx .* Dx(c)) + mu_v * DyT(wcy .* Dy(c)))
    DxInto(c, tmpc1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) tmpc1[i] *= wcx[i];
    DxTInto(tmpc1, cprior);

    DyInto(c, tmpc2);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) tmpc2[i] *= wcy[i];
    DyTInto(tmpc2, tmpc3);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i)
      cprior[i] = 2.0F * (mu_h * cprior[i] + mu_v * tmpc3[i]);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i)
      out[i] = -img[i] * std::conj(carrier[i]) + cprior[i];
  };

  // H = sum wx (Dx dY)^2 + wy (Dy dY)^2
  //   + mu_h sum wcx |Dx dC|^2 + mu_v sum wcy |Dy dC|^2,   dY = -Re[dC c].
  auto curv = [&](const ComplexPlane& dc) -> double {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < static_cast<long>(N); ++i)
      dy[i] = -(dc[i] * carrier[i]).real();
    DxInto(dy, dxdy);
    DyInto(dy, dydy);
    DxInto(dc, dxc);
    DyInto(dc, dyc);
    double h = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : h) schedule(static)
#endif
    for (long i = 0; i < static_cast<long>(N); ++i) {
      h += static_cast<double>(wx[i]) * dxdy[i] * dxdy[i];
      h += static_cast<double>(wy[i]) * dydy[i] * dydy[i];
      h += static_cast<double>(mu_h) * wcx[i] * std::norm(dxc[i]);
      h += static_cast<double>(mu_v) * wcy[i] * std::norm(dyc[i]);
    }
    return h;
  };

  const double n_pixels = static_cast<double>(N);
  const float e2 = eps * eps;
  const float ec2 = eps_c * eps_c;

  for (int outer = 0; outer < irls_outer; ++outer) {
    // Recompute IRLS weights at the current chi (lagged diffusivity).
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < static_cast<long>(N); ++i)
      yc[i] = s[i] - (chi[i] * carrier[i]).real();
    DxInto(yc, gxY);
    DyInto(yc, gyY);
    DxInto(chi, tmpc1);
    DyInto(chi, tmpc2);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < static_cast<long>(N); ++i) {
      wx[i] = eps / std::sqrt(gxY[i] * gxY[i] + e2);
      wy[i] = eps / std::sqrt(gyY[i] * gyY[i] + e2);
      magx[i] = std::abs(tmpc1[i]);
      magy[i] = std::abs(tmpc2[i]);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < static_cast<long>(N); ++i) {
      const float dx_den = magx[i] * magx[i] + k * gxY[i] * gxY[i] + ec2;
      const float dy_den = magy[i] * magy[i] + k * gyY[i] * gyY[i] + ec2;
      wcx[i] = eps_c / std::sqrt(dx_den);
      wcy[i] = eps_c / std::sqrt(dy_den);
    }

    // Conjugate gradient on the fixed-weight quadratic.
    grad(chi, g);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < static_cast<long>(N); ++i) d[i] = -g[i];
    double gg = SumAbs2(g);

    for (int it = 0; it < n_inner; ++it) {
      const double hh = curv(d);
      if (hh <= 1e-12) break;
      const double gd = DotReal(g, d);
      const float alpha = static_cast<float>(-0.5 * gd / hh);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (long i = 0; i < static_cast<long>(N); ++i) chi[i] += alpha * d[i];

      grad(chi, g_new);
      const double gg_new = SumAbs2(g_new);
      const float beta = static_cast<float>(gg_new / std::max(gg, 1e-30));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (long i = 0; i < static_cast<long>(N); ++i)
        d[i] = -g_new[i] + beta * d[i];
      g = g_new;
      gg = gg_new;
      if (gg < 1e-10 * n_pixels) break;
    }
  }

  RefineResult out;
  out.luma = Luma(s, chi, carrier);
  out.chroma = std::move(chi);
  return out;
}

}  // namespace hvd
