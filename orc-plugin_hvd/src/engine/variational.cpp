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
  for (size_t i = 0; i < g.size(); ++i)
    acc += static_cast<double>(g[i].real()) * g[i].real() +
           static_cast<double>(g[i].imag()) * g[i].imag();
  return acc;
}

// Re(sum(conj(a) * b)) over complex planes.
double DotReal(const ComplexPlane& a, const ComplexPlane& b) {
  double acc = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
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

  ComplexPlane chi = chi0;  // working copy

  // Gradient of the (fixed-weight) quadratic majorant with respect to chi.
  // g_chi = -[ 2 (DxT(wx Dx Yc) + DyT(wy Dy Yc)) ] * conj(carrier)
  //          + 2 ( mu_h DxT(wcx Dx chi) + mu_v DyT(wcy Dy chi) )
  // The weights wx, wy, wcx, wcy are captured by reference and held fixed
  // within one IRLS outer pass (lagged diffusivity).
  auto grad = [&](const ComplexPlane& c, const Plane& wx, const Plane& wy,
                  const Plane& wcx, const Plane& wcy) -> ComplexPlane {
    const Plane yc = Luma(s, c, carrier);
    // Image (luma) term, folded back onto chi via -conj(carrier).
    const Plane img = [&] {
      Plane t1 = DxT(Mul(wx, Dx(yc)));
      Plane t2 = DyT(Mul(wy, Dy(yc)));
      Plane out(yc.height(), yc.width());
      for (size_t i = 0; i < out.size(); ++i) out[i] = 2.0F * (t1[i] + t2[i]);
      return out;
    }();
    // Chroma prior term.
    const ComplexPlane cprior = [&] {
      ComplexPlane t1 = DxT(MulReal(Dx(c), wcx));
      ComplexPlane t2 = DyT(MulReal(Dy(c), wcy));
      ComplexPlane out(c.height(), c.width());
      for (size_t i = 0; i < out.size(); ++i)
        out[i] = 2.0F * (mu_h * t1[i] + mu_v * t2[i]);
      return out;
    }();
    ComplexPlane g(c.height(), c.width());
    for (size_t i = 0; i < g.size(); ++i)
      g[i] = -img[i] * std::conj(carrier[i]) + cprior[i];
    return g;
  };

  // Curvature (quadratic form) of the majorant along a complex direction dC:
  //   H = sum wx (Dx dY)^2 + wy (Dy dY)^2
  //     + mu_h sum wcx |Dx dC|^2 + mu_v sum wcy |Dy dC|^2,   dY = -Re[dC c].
  auto curv = [&](const ComplexPlane& dc, const Plane& wx, const Plane& wy,
                  const Plane& wcx, const Plane& wcy) -> double {
    Plane dy(dc.height(), dc.width());
    for (size_t i = 0; i < dy.size(); ++i) dy[i] = -(dc[i] * carrier[i]).real();
    const Plane dxdy = Dx(dy);
    const Plane dydy = Dy(dy);
    const ComplexPlane dxc = Dx(dc);
    const ComplexPlane dyc = Dy(dc);
    double h = 0.0;
    for (size_t i = 0; i < dy.size(); ++i) {
      h += static_cast<double>(wx[i]) * dxdy[i] * dxdy[i];
      h += static_cast<double>(wy[i]) * dydy[i] * dydy[i];
      h += static_cast<double>(mu_h) * wcx[i] * std::norm(dxc[i]);
      h += static_cast<double>(mu_v) * wcy[i] * std::norm(dyc[i]);
    }
    return h;
  };

  const double n_pixels = static_cast<double>(s.size());

  for (int outer = 0; outer < irls_outer; ++outer) {
    // Recompute IRLS weights at the current chi (lagged diffusivity).
    const Plane y = Luma(s, chi, carrier);
    const Plane gxY = Dx(y);
    const Plane gyY = Dy(y);
    const Plane wx = CharbonnierWeight(gxY, eps);
    const Plane wy = CharbonnierWeight(gyY, eps);
    const Plane wcx = CoupledWeight(Magnitude(Dx(chi)), gxY, eps_c, k);
    const Plane wcy = CoupledWeight(Magnitude(Dy(chi)), gyY, eps_c, k);

    // Conjugate gradient on the fixed-weight quadratic.
    ComplexPlane g = grad(chi, wx, wy, wcx, wcy);
    ComplexPlane d(g.height(), g.width());
    for (size_t i = 0; i < d.size(); ++i) d[i] = -g[i];
    double gg = SumAbs2(g);

    for (int it = 0; it < n_inner; ++it) {
      const double hh = curv(d, wx, wy, wcx, wcy);
      if (hh <= 1e-12) break;
      const double gd = DotReal(g, d);
      const float alpha = static_cast<float>(-0.5 * gd / hh);
      for (size_t i = 0; i < chi.size(); ++i) chi[i] += alpha * d[i];

      const ComplexPlane g_new = grad(chi, wx, wy, wcx, wcy);
      const double gg_new = SumAbs2(g_new);
      const float beta = static_cast<float>(gg_new / std::max(gg, 1e-30));
      for (size_t i = 0; i < d.size(); ++i) d[i] = -g_new[i] + beta * d[i];
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
