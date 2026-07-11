// SPDX-License-Identifier: GPL-3.0-or-later
//
// engine.h — the HVD decoder facade.
//
// This is the single, narrow interface the decode-orc SDK layer talks to. It
// contains NO SDK types, so it can be unit-tested and reasoned about in
// isolation. It orchestrates the ported 2-D woven-frame pipeline:
//
//   per field:  lock-in burst phase  (lockin.h)
//   weave:      interleave the two fields' active picture into frame geometry
//   init:       holographic reconstruction of chi           (holographic_init.h)
//   refine:     Y/C arbitration by IRLS + conjugate gradient (variational.h)
//   output:     lossless Y/C split  luma + chroma == composite
//
// A single Fft2d (FFTW plan cache) is owned by the engine and reused across
// frames, so repeated decoding of a sequence does not re-plan.
//
// 3-D / temporal decoding is a documented extension point: DecodeFrame takes
// only the current frame's two fields today; the temporal terms in the
// reference operate on neighbouring fields and would be added as an overload
// that also accepts the neighbour window (see docs/PORTING.md).

#ifndef ORC_PLUGIN_HVD_ENGINE_ENGINE_H_
#define ORC_PLUGIN_HVD_ENGINE_ENGINE_H_

#include <memory>

#include "engine/hvd_config.h"
#include "engine/ntsc_geometry.h"
#include "engine/plane.h"

namespace hvd {

class Fft2d;  // forward declaration (owned via a unique_ptr)

// One field, in IRE, plus the metadata the weave needs. `samples` is
// (field_height x field_width).
struct FieldInput {
  Plane samples;        // composite, IRE
  bool is_first_field;  // true => this field's lines are the top (even) rows
};

// Woven-frame decode result over the active picture area. The three planes
// share the same geometry (2 * active_lines rows x active_width cols) and, by
// construction, satisfy  luma + chroma == composite  elementwise (the lossless
// Y/C split). `chroma_phasor` is the baseband chroma chi = V - iU, provided for
// an optional colour/preview path (not needed for the split output).
struct FrameYc {
  Plane luma;                  // Y (IRE)
  Plane chroma;                // C = S - Y = Re[chi * carrier]  (IRE, AC)
  Plane composite;             // S (IRE), for verification / passthrough
  ComplexPlane chroma_phasor;  // chi = V - iU (IRE)
  float acc_gain = 1.0F;       // measured Automatic Color Control gain
};

class HvdEngine {
 public:
  HvdEngine();
  ~HvdEngine();

  HvdEngine(const HvdEngine&) = delete;
  HvdEngine& operator=(const HvdEngine&) = delete;

  // Decode one frame from its two fields. `first` and `second` may be given in
  // capture order; the engine orders first-field lines onto the even rows using
  // is_first_field. `g` is the field geometry, `cfg` the knobs.
  FrameYc DecodeFrame(const FieldInput& first, const FieldInput& second,
                      const FieldGeometry& g, const HvdConfig& cfg);

  // For sources that are ALREADY Y/C separated at capture (has_separate_
  // channels() on the host side — S-Video-style captures, some hi-fi VHS
  // formats): `first`/`second` here are the CHROMA-ONLY fields (already
  // re-centred to a signed zero-mean oscillation by the caller — see
  // frame_bridge.cpp's chroma_dc handling), with no luma mixed in. There is
  // no separation problem to solve here — the source already solved it —
  // so this skips the variational arbitration entirely (nothing to
  // arbitrate: residual luma is ~0 by construction) and just does burst
  // lock-in + a holographic-bandwidth crop to get chi. Cheaper than
  // DecodeFrame and correct for this input shape; using DecodeFrame here
  // instead (on a real composite reconstructed as luma-only, no chroma)
  // silently produces zero chroma — that used to be a real bug.
  // Returned FrameYc.luma is meaningless (approximately zero) — the caller
  // should get luma from the source's own clean Y channel directly instead.
  FrameYc DecodeChromaOnly(const FieldInput& first, const FieldInput& second,
                           const FieldGeometry& g, const HvdConfig& cfg);

 private:
  std::unique_ptr<Fft2d> fft_;
};

}  // namespace hvd

#endif  // ORC_PLUGIN_HVD_ENGINE_ENGINE_H_
