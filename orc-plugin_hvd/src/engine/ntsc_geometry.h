// SPDX-License-Identifier: GPL-3.0-or-later
//
// ntsc_geometry.h — NTSC 4fsc signal geometry and level conversions.
//
// Groups the small, dependency-free helpers that describe how a time-base
// corrected NTSC field maps onto the holographic model used by the decoder:
//
//   S(x, y) = Y(x, y) + Re[ chi(x, y) * exp(i * phi(x, y)) ],   chi = V - iU
//
// with the carrier phase phi(line, x) = theta[line] + (pi/2) * x  (90 deg per
// sample at 4fsc). These mirror `VideoParameters` and the phase/noise helpers in
// `reference/hvd/decoder.py` and `reference/hvd/tbc.py`.

#ifndef ORC_PLUGIN_HVD_ENGINE_NTSC_GEOMETRY_H_
#define ORC_PLUGIN_HVD_ENGINE_NTSC_GEOMETRY_H_

#include <vector>

#include "engine/plane.h"

namespace hvd {

// NTSC 4fsc colour subcarrier and sample rate (Hz). fsc = 315e6 / 88.
constexpr double kFscNtsc = 315.0e6 / 88.0;      // 3 579 545.45... Hz
constexpr double kFs4Fsc = 4.0 * kFscNtsc;        // 14 318 181.8 Hz

// Composite geometry needed by the engine, expressed in the sample domain of a
// single field. Populated by the SDK layer from decode-orc SourceParameters (or
// by tests directly). All indices are 0-based sample / line positions.
struct FieldGeometry {
  int field_width = 910;          // samples per line (full, incl. sync/blanking)
  int field_height = 263;         // lines per field (incl. VBI)
  int active_video_start = 134;   // first active-picture sample (inclusive)
  int active_video_end = 894;     // one past last active-picture sample
  int colour_burst_start = 78;    // burst window start sample (inclusive)
  int colour_burst_end = 110;     // burst window end sample (exclusive)
  int first_active_field_line = 21;  // first active line within a field
  int last_active_field_line = 0;    // one past last active line; 0 => field_height
  double sample_rate = kFs4Fsc;   // Hz

  int active_width() const { return active_video_end - active_video_start; }
  int last_active_line() const {
    return last_active_field_line != 0 ? last_active_field_line : field_height;
  }
  int active_lines() const {
    return last_active_line() - first_active_field_line;
  }
};

// Convert one plane of composite samples from the caller's linear IRE-like unit.
// The SDK layer performs the actual 10-bit -> IRE mapping using the source's
// black/white levels; the engine itself always works in IRE floats. Provided
// here as a free function so both the stage and tests share one definition.
//
//   ire = (sample - black) / ((white - black) / 100)
inline float SampleToIre(float sample, float black_level, float white_level) {
  const float scale = (white_level - black_level) / 100.0F;
  return (sample - black_level) / scale;
}

//   sample = black + ire * ((white - black) / 100)
inline float IreToSample(float ire, float black_level, float white_level) {
  const float scale = (white_level - black_level) / 100.0F;
  return black_level + ire * scale;
}

// Build the per-sample carrier phase map phi over the ACTIVE picture, given the
// per-line phase offsets theta (already restricted to the active lines) and the
// field geometry. Result has one row per entry in `theta` and `active_width()`
// columns. Matches `phase_map(...)[:, a0:a1]` in the reference.
ComplexPlane MakeCarrier(const std::vector<float>& theta, const FieldGeometry& g);

// Robust per-field noise estimate (IRE), from the stride-4 horizontal second
// difference. At 4fsc the carrier completes 360 deg over 4 samples, so
// S[x] - 2 S[x+4] + S[x+8] cancels chroma AND smooth luma exactly, leaving
// noise. The 25th percentile of |centered d| is outlier-proof against sparse
// luma detail. Used to auto-calibrate the (future) temporal gates.
float EstimateNoiseIre(const Plane& s);

}  // namespace hvd

#endif  // ORC_PLUGIN_HVD_ENGINE_NTSC_GEOMETRY_H_
