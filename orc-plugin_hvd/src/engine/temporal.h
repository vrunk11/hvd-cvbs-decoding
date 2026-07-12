// SPDX-License-Identifier: GPL-3.0-or-later
//
// temporal.h — motion-compensated warp and envelope primitives for the
// (upcoming) 3-D temporal path. Faithful port of the RAW-COMPOSITE side of
// reference/hvd/decoder.py: _vectors_per_pixel, warp_by_tiles,
// _warp_bilinear_tiles, envelope_of, complex_coherence, mc_warp,
// motion_compensate_envelope, motion_compensate_prev.
//
// Deliberately NOT included here: synth_reference. That function builds a
// denoised reference from already-DECODED neighbor fields (Ys[k]/chis[k]),
// which is the Gauss-Seidel-within-a-chunk dependency discussed for the
// chunked pipeline — a later step once the chunk/window driver exists.
// Everything in this file only ever touches neighbors' RAW measurements
// (composite + measured carrier phase), which is why motion.h/this file
// have no inter-frame decode-order dependency at all.

#ifndef ORC_PLUGIN_HVD_ENGINE_TEMPORAL_H_
#define ORC_PLUGIN_HVD_ENGINE_TEMPORAL_H_

#include "engine/motion.h"
#include "engine/plane.h"

#include <vector>

namespace hvd {

// Bilinearly interpolates a per-tile motion field (mdy/mdx, shape
// tiles_h x tiles_w) between TILE CENTERS onto a full-resolution (h x w)
// per-pixel field, after a 3x3 component-wise median outlier-snap (a
// vector more than 3px away from its own 3x3 median is replaced by that
// median — squashes isolated garbage from flat tiles before it can bleed
// into neighbors through the interpolation; coherent motion clusters pass
// through untouched). This is what removes the visible tile-seam artefacts
// of a piecewise-constant warp.
struct PerPixelMotion {
  Plane dy;  // h x w
  Plane dx;  // h x w
};
PerPixelMotion VectorsPerPixel(const Plane& mdy, const Plane& mdx, int tile,
                               int out_h, int out_w);

// Nearest-tile, integer-pel warp: shifts each pixel of `a` by its
// (interpolated, then rounded) per-pixel vector. Matches warp_by_tiles —
// legal on a raw carrier-modulated signal, since it's a pure sample shift
// (no resampling of the carrier itself).
Plane WarpByTiles(const Plane& a, const Plane& mdy, const Plane& mdx,
                  int tile);

// Bilinear, sub-pixel warp of a BASEBAND array (post envelope_of — never
// call this on a still-modulated composite, sub-pixel interpolation of a
// carrier is not legal). `row_offset` is a constant extra vertical shift
// (grid alignment between two fields of different parity); the output grid
// size can differ from the input's (envelope arrays have one fewer row).
Plane WarpBilinearTiles(const Plane& a, const Plane& dyf, const Plane& dxf,
                        int tile, float row_offset, int out_h, int out_w);

// Line-pair comb separation of a raw field into baseband envelope samples
// on the HALF-LINE grid (between lines m and m+1) — see envelope_of in the
// reference for the derivation. `carrier` is exp(i*phi) at the SAME
// resolution as `s` (this engine already builds that via MakeCarrier,
// unlike the Python reference which stores raw phi and re-exponentiates
// here — passing the carrier directly avoids redoing that plumbing).
// Output height is s.height() - 1.
struct Envelope {
  Plane luma;          // Yb, baseband luma on the half-line grid
  ComplexPlane chroma;  // chi_b, baseband chroma phasor on the half-line grid
};
Envelope EnvelopeOf(const Plane& s, const ComplexPlane& carrier);

// InSAR-style local complex coherence between two chroma phasor fields:
//   gamma = |<z1 * conj(z2)>| / sqrt(<|z1|^2> * <|z2|^2>)
// (box-blurred local averages, radius r). Phase-sensitive where a plain
// magnitude/SSD comparison isn't: two chroma fields can have similar
// energy yet decorrelated phase (motion residual, scene change) — exactly
// where a temporal chroma equation would otherwise turn toxic. ~1 where
// the phasors line up, -> 0 where they decorrelate. Output in [0, 1].
Plane ComplexCoherence(const ComplexPlane& z1, const ComplexPlane& z2, int r);

// Same integer-pel, nearest-tile warp as WarpByTiles, but for complex data.
// Only valid for an INTEGER shift with no interpolation (as used in
// MotionCompensatePrev's carrier warp) — warping a complex exponential by a
// pure sample shift commutes with warping its angle then re-exponentiating,
// which is exactly what makes this legal; it would NOT be legal to use this
// for a sub-pixel/bilinear warp of a still-modulated carrier.
ComplexPlane WarpByTilesComplex(const ComplexPlane& a, const Plane& mdy,
                                const Plane& mdx, int tile);

// Upsamples a per-tile confidence field to per-pixel: NEAREST-tile (not
// VectorsPerPixel's smooth interpolation — matches the reference's
// conf[ty,tx] indexing exactly), squared (crushes weak matches harder than
// a linear scale would), then box-blurred (radius 8) to avoid hard tile
// seams in the confidence itself.
Plane UpsampleConfidence(const Plane& conf, int tile, int out_h, int out_w);

// Estimates motion from y_from toward y_to and warps every plane in
// `arrays` accordingly (integer-pel — these are still-modulated composite-
// domain arrays, not baseband). Matches mc_warp in the reference.
struct McWarpResult {
  std::vector<Plane> warped;
  Plane confidence;
};
McWarpResult McWarp(const Plane& y_from, const Plane& y_to,
                    const std::vector<Plane>& arrays, int tile, int search);

// One neighbour frame's raw measurements. `luma` is a STABLE reference used
// only for motion matching (the reference uses the holographic-init luma
// here, never a fully-refined decode — deliberately, so multiple IRLS/CG
// passes can't feed back into the motion field they were themselves
// computed from); `composite`/`carrier` are the actual raw measurement
// (composite samples + measured carrier exp(i*phi)) that ends up in the
// synthesized data term.
struct NeighborRawState {
  Plane luma;
  Plane composite;
  ComplexPlane carrier;
};

// Envelope-domain neighbour resampling (motion_compensate_envelope in the
// reference): comb-separates the neighbour's raw field into baseband
// (Yb, chi_b), motion-warps it with SUB-PIXEL bilinear interpolation
// (legal in baseband — forbidden on the still-modulated carrier), then
// re-encodes it at the CURRENT field's own phase + 180 deg. This is what
// makes the half-line parity offset between opposite-parity fields vanish
// by construction (the comb's half-grid IS the other parity's grid) and
// gives |dc| = 2 (maximal chroma leverage) for every neighbour, vs sqrt(2)
// for the plain raw-adjacent-field pairing in MotionCompensatePrev below.
// `parity_cur`/`parity_nb` are 0 (top field) or 1 (bottom field) — pass the
// TRUE spatial parity (e.g. from capture metadata), not just index parity,
// on sources that can start on a second field or skip fields.
// If `motion` is null, it's estimated internally from neighbor.luma vs
// y_cur; pass a precomputed one to reuse a motion field across calls.
struct MotionCompensatedResult {
  Plane composite;       // S_w: re-encoded composite estimate
  ComplexPlane carrier;  // c_w: the (phase-flipped) carrier S_w is expressed against
  Plane confidence;
};
MotionCompensatedResult MotionCompensateEnvelope(
    const NeighborRawState& neighbor, const Plane& y_cur,
    const ComplexPlane& carrier_cur, int parity_cur, int parity_nb, int tile,
    int search, const MotionField* motion = nullptr);

// Simpler neighbour resampling (motion_compensate_prev in the reference):
// integer-pel warp of the neighbour's raw composite AND raw carrier phase
// toward the current frame — no envelope separation, no sub-pixel warp, no
// re-encoding. Cheaper, and the natural choice for same-parity neighbours
// (e.g. +/-2 fields, 1 frame apart) where MotionCompensateEnvelope's
// cross-parity machinery isn't needed.
MotionCompensatedResult MotionCompensatePrev(
    const NeighborRawState& prev, const Plane& y_cur_init, int tile,
    int search, const MotionField* motion = nullptr);

}  // namespace hvd

#endif  // ORC_PLUGIN_HVD_ENGINE_TEMPORAL_H_
