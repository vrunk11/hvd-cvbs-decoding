// SPDX-License-Identifier: GPL-3.0-or-later
//
// variational.h — the Y/C arbitration solver (the heart of the decoder).
//
// Ports the 2-D (no-temporal-neighbour) path of `variational_refine` from
// `reference/hvd/decoder.py`.
//
// Structural fact that makes this different from a comb filter: the data term
// ||S - Y - Re[chi e^{i phi}]||^2 is *invariant* to transfers between luma and
// modulated chroma — the holographic init already fits the composite exactly.
// So Y/C separation is not a filtering problem, it is an *arbitration* problem.
// We enforce exact data fidelity by eliminating Y (Y := S - Re[chi e^{i phi}])
// and solve, over chi alone,
//
//     argmin_chi  sum rho( grad Y(chi) )
//               + mu_h sum rho_c( Dx chi ) + mu_v sum rho_c( Dy chi )
//
// with rho, rho_c the edge-preserving Charbonnier penalty, solved by IRLS
// (lagged diffusivity) + conjugate gradient. Dot crawl is carrier-frequency
// ripple in Y (huge grad-Y penalty) so it migrates into chi; twin-image /
// cross-colour is a 2fsc oscillation in chi (huge grad-chi penalty) so it
// migrates back into Y. The two classic NTSC artefacts police each other.
//
// The chroma diffusivity is coupled to the luma gradients (parallel level sets)
// so a luma edge opens a matching chroma edge — this removes hanging dots at
// vertical chroma transitions.
//
// NOTE (3-D seam): the reference also accepts motion-compensated temporal
// neighbours here (an extra robust data term per neighbour). That path is not
// wired up in this first port; the function signature and the objective are
// arranged so it can be added without changing callers. See docs/PORTING.md.

#ifndef ORC_PLUGIN_HVD_ENGINE_VARIATIONAL_H_
#define ORC_PLUGIN_HVD_ENGINE_VARIATIONAL_H_

#include "engine/hvd_config.h"
#include "engine/plane.h"

namespace hvd {

struct RefineResult {
  Plane luma;           // Y = S - Re[chi * carrier]  (IRE)
  ComplexPlane chroma;  // chi = V - iU  (IRE)
};

// Refine `chi0` by arbitration. `s` is the active composite (IRE), `carrier` is
// exp(i*phi), `chi0` the holographic init. On return the pair satisfies
// luma + Re[chroma * carrier] == s exactly (a lossless split).
RefineResult VariationalRefine(const Plane& s, const ComplexPlane& carrier,
                               const ComplexPlane& chi0, const HvdConfig& cfg);

}  // namespace hvd

#endif  // ORC_PLUGIN_HVD_ENGINE_VARIATIONAL_H_
