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

// sum(a^2) over a real plane.
double SumSq(const Plane& a) {
  double acc = 0.0;
  const long n = static_cast<long>(a.size());
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : acc) schedule(static)
#endif
  for (long i = 0; i < n; ++i) acc += static_cast<double>(a[i]) * a[i];
  return acc;
}

// sum(a .* b) over real planes.
double DotRealPlane(const Plane& a, const Plane& b) {
  double acc = 0.0;
  const long n = static_cast<long>(a.size());
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : acc) schedule(static)
#endif
  for (long i = 0; i < n; ++i) acc += static_cast<double>(a[i]) * b[i];
  return acc;
}

}  // namespace

RefineResult VariationalRefine(const Plane& s, const ComplexPlane& carrier,
                               const ComplexPlane& chi0, const HvdConfig& cfg,
                               const std::vector<NeighborTerm>& neighbors) {
  const float eps = cfg.charbonnier_eps;
  const float eps_c = cfg.chroma_eps;
  const float eps_t = cfg.temporal_eps;
  const float mu_h = cfg.lambda_c * cfg.chroma_aniso;  // chroma broader in x
  const float mu_v = cfg.lambda_c;
  const float k = cfg.structure_coupling;
  const float nu = neighbors.empty() ? 0.0F : cfg.temporal_strength;
  const size_t n_nbr = neighbors.size();

  const int irls_outer = std::max(1, cfg.irls_outer);
  const int n_inner = std::max(1, cfg.cg_iterations / irls_outer);

  const int H = s.height();
  const int W = s.width();
  const size_t N = s.size();
  const long n = static_cast<long>(N);

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
  Plane rt(H, W);  // per-neighbour temporal residual scratch

  // Per-neighbour delta dc = carrier - c_w and IRLS weight wt, precomputed
  // once (dc is fixed for the whole solve) / recomputed per outer pass
  // (wt is lagged-diffusivity, like wx/wy/wcx/wcy).
  std::vector<ComplexPlane> dcs(n_nbr);
  std::vector<Plane> wts(n_nbr);
  for (size_t j = 0; j < n_nbr; ++j) {
    dcs[j] = ComplexPlane(H, W);
    wts[j] = Plane(H, W);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) dcs[j][i] = carrier[i] - neighbors[j].carrier[i];
  }

  // temporal_residual(chi, S_w, dc) = (S_w - S) + Re[chi * dc]  — the
  // reference's algebraic reformulation of S_w - Y - Re[chi*c_w] after
  // substituting Y = S - Re[chi*carrier] (see variational.h's doc comment
  // on this function for the derivation). Written into `rt` (reused
  // scratch), not returned, to avoid an allocation per neighbour per call.
  auto temporal_residual_into = [&](const ComplexPlane& c, const NeighborTerm& nbr,
                                    const ComplexPlane& dc) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i)
      rt[i] = (nbr.composite[i] - s[i]) + (c[i] * dc[i]).real();
  };

  // g = -[2 (DxT(wx Dx Yc) + DyT(wy Dy Yc))] * conj(carrier)
  //     + 2 (mu_h DxT(wcx Dx chi) + mu_v DyT(wcy Dy chi))
  //     + 2 * nu * sum_k wt_k * temporal_residual_k * conj(dc_k)
  auto grad = [&](const ComplexPlane& c, ComplexPlane& out) {
    // Yc = S - Re[c * carrier]
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

    for (size_t j = 0; j < n_nbr; ++j) {
      temporal_residual_into(c, neighbors[j], dcs[j]);
      const Plane& wt = wts[j];
      const ComplexPlane& dc = dcs[j];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (long i = 0; i < n; ++i)
        out[i] += 2.0F * nu * wt[i] * rt[i] * std::conj(dc[i]);
    }
  };

  // H = sum wx (Dx dY)^2 + wy (Dy dY)^2
  //   + mu_h sum wcx |Dx dC|^2 + mu_v sum wcy |Dy dC|^2,   dY = -Re[dC c].
  //   + nu * sum_k sum wt_k * Re[dC * dc_k]^2
  auto curv = [&](const ComplexPlane& dc_dir) -> double {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) dy[i] = -(dc_dir[i] * carrier[i]).real();
    DxInto(dy, dxdy);
    DyInto(dy, dydy);
    DxInto(dc_dir, dxc);
    DyInto(dc_dir, dyc);
    double h = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : h) schedule(static)
#endif
    for (long i = 0; i < n; ++i) {
      h += static_cast<double>(wx[i]) * dxdy[i] * dxdy[i];
      h += static_cast<double>(wy[i]) * dydy[i] * dydy[i];
      h += static_cast<double>(mu_h) * wcx[i] * std::norm(dxc[i]);
      h += static_cast<double>(mu_v) * wcy[i] * std::norm(dyc[i]);
    }
    for (size_t j = 0; j < n_nbr; ++j) {
      const Plane& wt = wts[j];
      const ComplexPlane& dc = dcs[j];
      double acc = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : acc) schedule(static)
#endif
      for (long i = 0; i < n; ++i) {
        const float re = (dc_dir[i] * dc[i]).real();
        acc += static_cast<double>(wt[i]) * re * re;
      }
      h += nu * acc;
    }
    return h;
  };

  const double n_pixels = static_cast<double>(N);
  const float e2 = eps * eps;
  const float ec2 = eps_c * eps_c;
  const float et2 = eps_t * eps_t;

  for (int outer = 0; outer < irls_outer; ++outer) {
    // Recompute IRLS weights at the current chi (lagged diffusivity).
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) yc[i] = s[i] - (chi[i] * carrier[i]).real();
    DxInto(yc, gxY);
    DyInto(yc, gyY);
    DxInto(chi, tmpc1);
    DyInto(chi, tmpc2);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) {
      wx[i] = eps / std::sqrt(gxY[i] * gxY[i] + e2);
      wy[i] = eps / std::sqrt(gyY[i] * gyY[i] + e2);
      magx[i] = std::abs(tmpc1[i]);
      magy[i] = std::abs(tmpc2[i]);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) {
      const float dx_den = magx[i] * magx[i] + k * gxY[i] * gxY[i] + ec2;
      const float dy_den = magy[i] * magy[i] + k * gyY[i] * gyY[i] + ec2;
      wcx[i] = eps_c / std::sqrt(dx_den);
      wcy[i] = eps_c / std::sqrt(dy_den);
    }
    if (nu > 0.0F) {
      for (size_t j = 0; j < n_nbr; ++j) {
        temporal_residual_into(chi, neighbors[j], dcs[j]);
        Plane& wt = wts[j];
        const Plane& conf = neighbors[j].confidence;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long i = 0; i < n; ++i)
          wt[i] = conf[i] * et2 / (rt[i] * rt[i] + et2);
      }
    }

    // Conjugate gradient on the fixed-weight quadratic.
    grad(chi, g);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) d[i] = -g[i];
    double gg = SumAbs2(g);

    for (int it = 0; it < n_inner; ++it) {
      const double hh = curv(d);
      if (hh <= 1e-12) break;
      const double gd = DotReal(g, d);
      const float alpha = static_cast<float>(-0.5 * gd / hh);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (long i = 0; i < n; ++i) chi[i] += alpha * d[i];

      grad(chi, g_new);
      const double gg_new = SumAbs2(g_new);
      const float beta = static_cast<float>(gg_new / std::max(gg, 1e-30));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (long i = 0; i < n; ++i) d[i] = -g_new[i] + beta * d[i];
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

RefineResult VariationalRefineJoint(const Plane& s, const ComplexPlane& carrier,
                                    const Plane& y0, const ComplexPlane& chi0,
                                    const HvdConfig& cfg,
                                    const std::vector<NeighborTerm>& neighbors,
                                    const AnchorTerm* anchor) {
  const float eps = cfg.charbonnier_eps;
  const float eps_c = cfg.chroma_eps;
  const float eps_t = cfg.temporal_eps;
  const float mu_h = cfg.lambda_c * cfg.chroma_aniso;
  const float mu_v = cfg.lambda_c;
  const float k = cfg.structure_coupling;
  const float nu = neighbors.empty() ? 0.0F : cfg.temporal_strength;
  const float nu_a = anchor ? cfg.nr_anchor : 0.0F;

  const int irls_outer = std::max(1, cfg.irls_outer);
  const int n_inner = std::max(1, cfg.cg_iterations / irls_outer);

  const int H = s.height();
  const int W = s.width();
  const size_t N = s.size();
  const size_t n_nbr = neighbors.size();

  Plane Y = y0;
  ComplexPlane chi = chi0;

  // ---- Workspace, allocated once (same rationale as VariationalRefine) ----
  Plane wx(H, W), wy(H, W), wcx(H, W), wcy(H, W);
  Plane gxY(H, W), gyY(H, W), magx(H, W), magy(H, W);
  Plane tmpr1(H, W), tmpr2(H, W);
  ComplexPlane tmpc1(H, W), tmpc2(H, W), tmpc3(H, W), cprior(H, W);
  Plane r0(H, W), rk(H, W);
  Plane gY(H, W), gY_new(H, W), dY(H, W);
  ComplexPlane gC(H, W), gC_new(H, W), dC(H, W);
  Plane dy_r(H, W), dxdy(H, W), dydy(H, W);
  ComplexPlane dxc(H, W), dyc(H, W);
  std::vector<Plane> wts(n_nbr);
  for (auto& w : wts) w = Plane(H, W);

  const long n = static_cast<long>(N);
  const double n_pixels = static_cast<double>(N);
  const float e2 = eps * eps;
  const float ec2 = eps_c * eps_c;
  const float et2 = eps_t * eps_t;

  // grad(Y, chi) -> (gY_out, gC_out): the joint objective's gradient —
  // exact data term + raw-neighbour terms + anchor term + spatial priors.
  auto grad = [&](const Plane& yy, const ComplexPlane& cc, Plane& gy_out,
                  ComplexPlane& gc_out) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) {
      r0[i] = s[i] - yy[i] - (cc[i] * carrier[i]).real();
      gy_out[i] = -2.0F * r0[i];
      gc_out[i] = -2.0F * r0[i] * std::conj(carrier[i]);
    }
    for (size_t j = 0; j < n_nbr; ++j) {
      const NeighborTerm& nbr = neighbors[j];
      const Plane& wt = wts[j];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (long i = 0; i < n; ++i) {
        const float rkv = nbr.composite[i] - yy[i] - (cc[i] * nbr.carrier[i]).real();
        gy_out[i] += -2.0F * nu * wt[i] * rkv;
        gc_out[i] += -2.0F * nu * wt[i] * rkv * std::conj(nbr.carrier[i]);
      }
    }
    if (nu_a > 0.0F) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (long i = 0; i < n; ++i) {
        gy_out[i] += 2.0F * nu_a * anchor->weight[i] * (yy[i] - anchor->luma[i]);
        gc_out[i] += 2.0F * nu_a * anchor->weight[i] * (cc[i] - anchor->chroma[i]);
      }
    }

    // Spatial Charbonnier priors on grad Y, grad chi (same shape as the
    // pass-1 solver, just added directly onto (gy_out, gc_out) here).
    DxInto(yy, tmpr1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) tmpr1[i] *= wx[i];
    DxTInto(tmpr1, tmpr2);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) gy_out[i] += 2.0F * tmpr2[i];

    DyInto(yy, tmpr1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) tmpr1[i] *= wy[i];
    DyTInto(tmpr1, tmpr2);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) gy_out[i] += 2.0F * tmpr2[i];

    DxInto(cc, tmpc1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) tmpc1[i] *= wcx[i];
    DxTInto(tmpc1, cprior);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) gc_out[i] += 2.0F * mu_h * cprior[i];

    DyInto(cc, tmpc2);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) tmpc2[i] *= wcy[i];
    DyTInto(tmpc2, cprior);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) gc_out[i] += 2.0F * mu_v * cprior[i];
  };

  // curv(dY, dC): the quadratic form's curvature along direction (dY, dC).
  auto curv = [&](const Plane& d_y, const ComplexPlane& d_c) -> double {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) dy_r[i] = d_y[i] + (d_c[i] * carrier[i]).real();
    double h = SumSq(dy_r);
    for (size_t j = 0; j < n_nbr; ++j) {
      const NeighborTerm& nbr = neighbors[j];
      const Plane& wt = wts[j];
      double acc = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : acc) schedule(static)
#endif
      for (long i = 0; i < n; ++i) {
        const float dk = d_y[i] + (d_c[i] * nbr.carrier[i]).real();
        acc += static_cast<double>(wt[i]) * dk * dk;
      }
      h += nu * acc;
    }
    if (nu_a > 0.0F) {
      double acc = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : acc) schedule(static)
#endif
      for (long i = 0; i < n; ++i) {
        acc += static_cast<double>(anchor->weight[i]) *
               (static_cast<double>(d_y[i]) * d_y[i] + std::norm(d_c[i]));
      }
      h += nu_a * acc;
    }
    DxInto(d_y, dxdy);
    DyInto(d_y, dydy);
    DxInto(d_c, dxc);
    DyInto(d_c, dyc);
    double spatial = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : spatial) schedule(static)
#endif
    for (long i = 0; i < n; ++i) {
      spatial += static_cast<double>(wx[i]) * dxdy[i] * dxdy[i];
      spatial += static_cast<double>(wy[i]) * dydy[i] * dydy[i];
      spatial += static_cast<double>(mu_h) * wcx[i] * std::norm(dxc[i]);
      spatial += static_cast<double>(mu_v) * wcy[i] * std::norm(dyc[i]);
    }
    return h + spatial;
  };

  for (int outer = 0; outer < irls_outer; ++outer) {
    // IRLS weights, lagged on the current (Y, chi) — same Charbonnier shape
    // as the pass-1 solver, just reading Y directly instead of deriving it
    // from S - Re[chi*carrier] (Y is now a free unknown).
    DxInto(Y, gxY);
    DyInto(Y, gyY);
    DxInto(chi, tmpc1);
    DyInto(chi, tmpc2);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) {
      wx[i] = eps / std::sqrt(gxY[i] * gxY[i] + e2);
      wy[i] = eps / std::sqrt(gyY[i] * gyY[i] + e2);
      magx[i] = std::abs(tmpc1[i]);
      magy[i] = std::abs(tmpc2[i]);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) {
      const float dx_den = magx[i] * magx[i] + k * gxY[i] * gxY[i] + ec2;
      const float dy_den = magy[i] * magy[i] + k * gyY[i] * gyY[i] + ec2;
      wcx[i] = eps_c / std::sqrt(dx_den);
      wcy[i] = eps_c / std::sqrt(dy_den);
    }
    if (nu > 0.0F) {
      for (size_t j = 0; j < n_nbr; ++j) {
        const NeighborTerm& nbr = neighbors[j];
        Plane& wt = wts[j];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long i = 0; i < n; ++i) {
          const float rkv = nbr.composite[i] - Y[i] - (chi[i] * nbr.carrier[i]).real();
          wt[i] = nbr.confidence[i] * et2 / (rkv * rkv + et2);
        }
      }
    }

    grad(Y, chi, gY, gC);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n; ++i) {
      dY[i] = -gY[i];
      dC[i] = -gC[i];
    }
    double gg = SumSq(gY) + SumAbs2(gC);

    for (int it = 0; it < n_inner; ++it) {
      const double hh = curv(dY, dC);
      if (hh <= 1e-12) break;
      const double gd = DotRealPlane(gY, dY) + DotReal(gC, dC);
      const float alpha = static_cast<float>(-0.5 * gd / hh);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (long i = 0; i < n; ++i) {
        Y[i] += alpha * dY[i];
        chi[i] += alpha * dC[i];
      }

      grad(Y, chi, gY_new, gC_new);
      const double gg_new = SumSq(gY_new) + SumAbs2(gC_new);
      const float beta = static_cast<float>(gg_new / std::max(gg, 1e-30));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (long i = 0; i < n; ++i) {
        dY[i] = -gY_new[i] + beta * dY[i];
        dC[i] = -gC_new[i] + beta * dC[i];
      }
      gY = gY_new;
      gC = gC_new;
      gg = gg_new;
      if (gg < 1e-10 * n_pixels) break;
    }
  }

  RefineResult out;
  out.luma = std::move(Y);
  out.chroma = std::move(chi);
  return out;
}

}  // namespace hvd
