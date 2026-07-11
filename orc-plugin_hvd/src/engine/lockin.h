// SPDX-License-Identifier: GPL-3.0-or-later
//
// lockin.h — colour-burst phase recovery by lock-in detection.
//
// Ports `burst_lockin_phase`, `_tridiag_smooth` and `burst_amplitude_ire`
// from `reference/hvd/decoder.py`.
//
// The carrier phase is recovered per line with a lock-in amplifier on the
// colour burst (multiply by a local oscillator, integrate, read the complex
// output), then the per-line phase *trajectory* is smoothed by a weighted
// tridiagonal solve + one IRLS re-weighting. Because every downstream equation
// depends on phi, hardening it here hardens the whole decoder against real,
// damaged media (dropouts, head noise). The reference measures unchanged decode
// quality with 25 % of bursts destroyed.

#ifndef ORC_PLUGIN_HVD_ENGINE_LOCKIN_H_
#define ORC_PLUGIN_HVD_ENGINE_LOCKIN_H_

#include <vector>

#include "engine/ntsc_geometry.h"
#include "engine/plane.h"

namespace hvd {

// Per-line subcarrier phase offset theta[line] such that
//   phi(line, x) = theta[line] + (pi/2) * x.
// `field_ire` is the full field in IRE (height x field_width). Lines with no
// detectable burst inherit the pi-per-line model phase.
std::vector<float> BurstLockinPhase(const Plane& field_ire,
                                    const FieldGeometry& g);

// Measured colour-burst amplitude (IRE), median over burst-bearing lines, for
// Automatic Color Control. Returns 20.0 (nominal) when no burst is found.
float BurstAmplitudeIre(const Plane& field_ire, const FieldGeometry& g);

// Solve (diag(a) + lam * L) x = a * d, with L the 1-D graph Laplacian, via the
// Thomas algorithm (O(n)). Exposed for unit testing. `a`, `d` have length n.
std::vector<float> TridiagSmooth(const std::vector<float>& d,
                                 const std::vector<float>& a, float lam);

}  // namespace hvd

#endif  // ORC_PLUGIN_HVD_ENGINE_LOCKIN_H_
