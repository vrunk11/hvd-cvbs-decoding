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

#include <cmath>
#include <cstdint>
#include <vector>

#include "engine/plane.h"

namespace hvd {

// NTSC 4fsc colour subcarrier and sample rate (Hz). fsc = 315e6 / 88.
constexpr double kFscNtsc = 315.0e6 / 88.0;      // 3 579 545.45... Hz
constexpr double kFs4Fsc = 4.0 * kFscNtsc;        // 14 318 181.8 Hz

// PAL 4fsc: fsc = 4 433 618.75 Hz. On the stored 1135-sample grid the
// subcarrier advances exactly 283.75 cycles/line => 270 deg/line; the
// 25 Hz offset survives as a slow drift the lock-in MEASURES (the PAL
// reference in reference-pal/, THEORY-PAL.md sections 1-2).
constexpr double kFscPal = 4433618.75;
constexpr double kFs4FscPal = 4.0 * kFscPal;      // 17 734 475 Hz

// Which composite standard a field's carrier obeys. This is the ONLY
// switch the numerical core dispatches on: everything downstream of the
// carrier builder (init, arbitration, temporal equations, anchor loop)
// is generic in the effective carrier c — NTSC is the special case
// c = exp(i*phi); PAL folds the V-switch into c (see MakeCarrierPal).
enum class VideoStandard { kNtsc, kPal };

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
  VideoStandard standard = VideoStandard::kNtsc;

  // NON-STANDARD SUBCARRIER (0 => use the standard's nominal fsc).
  //
  // Exists for captures whose composite is NTSC in every respect the host
  // measures (line rate, sync, blanking, burst window, 4fsc-nominal sample
  // grid) but whose colour subcarrier is NOT fs/4 — e.g. a disc/tape format
  // with a deliberately lowered subcarrier. The sample RATE is unchanged
  // (the host still stores CVBS_U10_4FSC on the standard's grid); only the
  // carrier that rides on that grid moves.
  //
  // Nothing else in the engine reads this directly: it feeds phase_per_sample()
  // and line_advance() below, and every equation downstream is generic in the
  // effective carrier c. See docs for what does NOT follow automatically.
  double subcarrier_hz = 0.0;

  // Effective colour subcarrier frequency (Hz).
  double fsc() const {
    if (subcarrier_hz > 0.0) return subcarrier_hz;
    return standard == VideoStandard::kPal ? kFscPal : kFscNtsc;
  }

  // Carrier phase advance per SAMPLE (rad). Exactly pi/2 when the grid is
  // 4fsc (the standard case); anything else for a moved subcarrier.
  double phase_per_sample() const {
    return 2.0 * 3.14159265358979323846 * fsc() / sample_rate;
  }

  // Per-line carrier phase advance on the stored grid, derived rather than
  // tabulated. cycles/line = fsc * field_width / sample_rate, so the standard
  // cases fall out unchanged: NTSC 227.5 cycles/line -> 180 deg, PAL 283.75
  // -> 270 deg. A non-standard fsc that is NOT an odd multiple of fH/2 gives
  // something else, and that is diagnostic: several assumptions in this engine
  // (the 2-line leak cancellation, the same-parity 180 deg flip) are only true
  // at 180 deg.
  double line_advance() const {
    const double cycles =
        fsc() * static_cast<double>(field_width) / sample_rate;
    const double frac = cycles - std::floor(cycles);
    return 2.0 * 3.14159265358979323846 * frac;
  }
  // Nominal burst amplitude for ACC: 20 IRE (NTSC) / 21.43 IRE (PAL,
  // +/-150 mV on the 700 mV white).
  float nominal_burst_ire() const {
    return standard == VideoStandard::kPal ? 21.43F : 20.0F;
  }

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

// PAL EFFECTIVE CARRIER (the load-bearing result of the PAL port,
// THEORY-PAL.md section 1). With one global chroma phasor chi = V - iU
// and the V-switch parity s[line] (+1 unswitched / -1 switched),
//
//   S = Y + Re[chi * c],   c = +exp(+i*phi) where s = +1
//                              -exp(-i*phi) where s = -1
//
// so the V-switch is a conjugated reference wave folded into a
// unit-modulus per-pixel carrier, and every equation of this engine
// applies unchanged. `vswitch` has one entry per row of `theta`.
ComplexPlane MakeCarrierPal(const std::vector<float>& theta,
                            const std::vector<int8_t>& vswitch,
                            const FieldGeometry& g);

// Robust per-field noise estimate (IRE), from the stride-4 horizontal second
// difference. At 4fsc the carrier completes 360 deg over 4 samples, so
// S[x] - 2 S[x+4] + S[x+8] cancels chroma AND smooth luma exactly, leaving
// noise. The 25th percentile of |centered d| is outlier-proof against sparse
// luma detail. Used to auto-calibrate the (future) temporal gates.
float EstimateNoiseIre(const Plane& s);

}  // namespace hvd

#endif  // ORC_PLUGIN_HVD_ENGINE_NTSC_GEOMETRY_H_
