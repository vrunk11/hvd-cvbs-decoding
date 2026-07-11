// SPDX-License-Identifier: GPL-3.0-or-later
//
// motion.h — tiled block-matching motion estimation, for the (upcoming) 3-D
// temporal path. Faithful port of estimate_motion()/_bm_pass() in
// reference/hvd/decoder.py: coarse-to-fine integer-pel block matching (4x
// decimated exhaustive search, then a +/-3 px full-res refinement around the
// coarse vector), a parabolic half-pel sub-pixel fit, a "prefer zero motion"
// margin rule (regularises against spurious vectors in flat/noisy tiles),
// and a per-tile confidence estimate.
//
// STATUS: this is stage 1 of the 3-D port (see docs/PORTING.md). It is
// self-contained and independently testable (tests/engine/motion_test.cpp)
// but not yet wired into HvdEngine — the temporal blending that consumes
// this motion field (motion-compensated reference synthesis, joint
// variational refinement across frames, drizzle) is later work, ported and
// tested incrementally rather than in one pass.

#ifndef ORC_PLUGIN_HVD_ENGINE_MOTION_H_
#define ORC_PLUGIN_HVD_ENGINE_MOTION_H_

#include "engine/plane.h"

namespace hvd {

// Per-tile motion field. dy/dx/confidence are all (tiles_h x tiles_w),
// where tiles_h = ceil(height / tile), tiles_w = ceil(width / tile) of the
// planes passed to EstimateMotion. dy/dx are in full-resolution pixels
// (integer part + up to +/-0.5 px sub-pixel refinement); confidence is in
// [0, 1], with 0 meaning "don't trust this tile's vector".
struct MotionField {
  Plane dy;
  Plane dx;
  Plane confidence;
  int tile = 32;
};

// Estimate per-tile motion from y_ref (reference/previous field) to y_cur
// (current field), both same-shaped luma-like planes (any consistent linear
// unit — IRE or raw code deltas both work, only relative contrast matters).
// `tile`: block size in pixels (reference default 32).
// `search`: full-resolution search radius in pixels (reference default 8) —
// the actual full-res refinement pass is always +/-3 px around a coarse
// vector found by a 4x-decimated search over +/-ceil(search/4), which is
// what makes this ~50x cheaper than an exhaustive full-res search of the
// same radius while matching the reference's behaviour (see estimate_motion
// in reference/hvd/decoder.py for the rationale).
MotionField EstimateMotion(const Plane& y_ref, const Plane& y_cur,
                            int tile = 32, int search = 8);

}  // namespace hvd

#endif  // ORC_PLUGIN_HVD_ENGINE_MOTION_H_
