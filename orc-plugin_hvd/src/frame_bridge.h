// SPDX-License-Identifier: GPL-3.0-or-later
//
// frame_bridge.h — SDK-independent bridge between a decode-orc flat frame
// buffer and the HVD engine.
//
// This file has NO decode-orc dependency: it operates on a plain int16_t flat
// buffer plus a small `FrameParams` mirror of the fields the stage reads from
// orc::SourceParameters. That keeps the tricky, testable logic (level
// conversion, field-sequential de-weave/re-weave, lossless requantisation) out
// of the SDK glue.
//
// LAYOUT (verified against orc/stage/video_frame_representation.h and
// cvbs_signal_constants.h): a CVBS_U10_4FSC frame is a flat buffer laid out
// FIELD-SEQUENTIALLY — all `field1_lines` lines of field 1, then the remaining
// lines of field 2:  [field1_line0 .. field1_lineN | field2_line0 .. ].
// Field 1 is the top spatial field, so spatial row 2k = field1 line k and
// spatial row 2k+1 = field2 line k. The engine runs per-field lock-in, so we
// split the buffer by field, decode, and write the Y/C split back into the same
// field-sequential layout.
//
// The active-picture line indices in orc::SourceParameters
// (first/last_active_frame_line) follow the ld-decode woven convention where
// field_line = frame_line / 2; that halving is applied here.

#ifndef ORC_PLUGIN_HVD_FRAME_BRIDGE_H_
#define ORC_PLUGIN_HVD_FRAME_BRIDGE_H_

#include <cstdint>
#include <vector>

#include "engine/engine.h"
#include "engine/hvd_config.h"
#include "engine/ntsc_geometry.h"

namespace hvd {

// The subset of orc::SourceParameters the decode needs, in the 10-bit domain.
// Populated by the stage from the host's SourceParameters; kept SDK-free so the
// bridge and engine stay unit-testable without the host.
struct FrameParams {
  int frame_width = 0;   // samples per line (910 NTSC)
  int frame_height = 0;  // total flat lines (525 NTSC)
  int field1_lines = 0;  // lines of field 1 in the flat buffer (263 NTSC)
  int active_video_start = 0;
  int active_video_end = 0;
  int colour_burst_start = 0;
  int colour_burst_end = 0;
  int first_active_frame_line = 0;  // woven-frame convention (see header)
  int last_active_frame_line = 0;   // 0 => derive from field height
  float black_level = 0.0F;
  float white_level = 0.0F;
  float blanking_level = 0.0F;
  float chroma_dc = 0.0F;  // chroma DC centre for the C channel
  double sample_rate = kFs4Fsc;
};

// A decoded Y/C frame in the 10-bit sample domain (int16_t == VFR sample_type),
// same field-sequential geometry as the input. By construction
// luma[i] + (chroma[i] - chroma_dc) reconstructs the composite to within one
// code. Both buffers are frame_width * frame_height samples.
//
// u_plane/v_plane carry the BASEBAND chrominance (chroma_phasor = V - iU),
// not the modulated `chroma` channel above — they are what a colour preview
// needs (no carrier to demodulate), on the same code-delta numeric scale as
// luma. Zero outside the active picture. Same frame_width * frame_height size.
struct YcFrameS16 {
  int width = 0;
  int height = 0;
  float chroma_dc = 0.0F;
  std::vector<int16_t> luma;
  std::vector<int16_t> chroma;
  std::vector<double> u_plane;
  std::vector<double> v_plane;
};

// Build the engine's field geometry from the host frame parameters.
FieldGeometry FieldGeometryFromParams(const FrameParams& fp);

// Decode one field-sequential CVBS frame into a lossless Y/C split.
// `engine` is reused across calls by the caller (its FFTW plan cache only
// pays the planning cost once per distinct frame size, not every frame —
// see engine.h). NOT safe to call concurrently on the same `engine` from
// multiple threads; each thread needs its own.
//   `frame` : frame_width * frame_height samples (row-major, 10-bit codes).
YcFrameS16 DecodeFrameBuffer(const int16_t* frame, const FrameParams& fp,
                             const HvdConfig& cfg, HvdEngine& engine);

// Convenience overload for tests/one-off calls: constructs a throwaway
// engine internally. Prefer the engine-taking overload above for any
// repeated/per-sequence decoding (which is every real use in the plugin) —
// this one re-plans FFTW from scratch on every call.
YcFrameS16 DecodeFrameBuffer(const int16_t* frame, const FrameParams& fp,
                             const HvdConfig& cfg);

// For sources that are ALREADY Y/C separated at capture (host's
// has_separate_channels() == true, e.g. get_frame_luma()/get_frame_chroma()
// on an S-Video-style or hi-fi-VHS TBC): `luma` and `chroma` are two
// separate field-sequential buffers, same geometry as `frame` above.
// `chroma` is expected UNSIGNED with fp.chroma_dc as its zero point (raw
// TBC convention — see orc_source_parameters.h's chroma_dc_offset) rather
// than already centred; this function does the re-centring.
// There is no Y/C separation problem to solve here (the source solved it),
// so this is much cheaper than DecodeFrameBuffer: luma passes through
// directly, and only the chroma undergoes burst lock-in + demodulation.
// Do NOT feed such a source's get_frame() (composite) into
// DecodeFrameBuffer instead — for a Y/C source that call returns the luma
// plane with no chroma in it at all, which silently decodes to zero
// chroma.
// Same engine-reuse and thread-safety notes as DecodeFrameBuffer above.
YcFrameS16 DecodeYcFrameBuffer(const int16_t* luma, const int16_t* chroma,
                               const FrameParams& fp, const HvdConfig& cfg,
                               HvdEngine& engine);

YcFrameS16 DecodeYcFrameBuffer(const int16_t* luma, const int16_t* chroma,
                               const FrameParams& fp, const HvdConfig& cfg);

}  // namespace hvd

#endif  // ORC_PLUGIN_HVD_FRAME_BRIDGE_H_
