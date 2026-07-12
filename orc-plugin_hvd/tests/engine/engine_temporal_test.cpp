// SPDX-License-Identifier: GPL-3.0-or-later
//
// Engine-level test for HvdEngine::DecodeFrame's prev_frames wiring (the
// frame-level 3D neighbour extension — see engine.h's doc comment). Checks
// that supplying a previous-frame neighbour with cfg.temporal_strength > 0
// still produces a lossless split for the CURRENT frame (VariationalRefine
// keeps Y exactly eliminated regardless of what the neighbour data looks
// like — this is the same invariant lossless_identity_test.cpp checks
// directly against VariationalRefine, exercised here through the full
// engine path instead: burst lock-in, weave, holographic init,
// MotionCompensatePrev, auto-calibrated temporal_eps).

#include <cmath>

#include "check.h"
#include "engine/engine.h"

namespace {

using hvd::Complex;
using hvd::ComplexPlane;
using hvd::FieldGeometry;
using hvd::FieldInput;
using hvd::HvdConfig;
using hvd::Plane;

constexpr float kHalfPi = 1.57079632679489661923F;
constexpr float kPi = 3.14159265358979323846F;

// One synthetic field: burst in the burst window, luma ramp + modulated
// chroma in the active picture, matching frame_bridge_test.cpp's pattern
// (not meant to be realistic NTSC content — just something BurstLockinPhase
// can lock a consistent phase onto and HolographicInit/VariationalRefine
// can losslessly split, regardless of what it looks like).
Plane MakeField(const FieldGeometry& g) {
  Plane field(g.field_height, g.field_width);
  for (int y = 0; y < g.field_height; ++y) {
    for (int x = 0; x < g.field_width; ++x) {
      float ire = 0.0F;
      if (x >= g.colour_burst_start && x < g.colour_burst_end) {
        ire = 20.0F * std::sin(kHalfPi * static_cast<float>(x) + kPi * static_cast<float>(y));
      }
      if (x >= g.active_video_start && x < g.active_video_end &&
          y >= g.first_active_field_line && y < g.last_active_line()) {
        const float luma = 30.0F + 30.0F * static_cast<float>(x - g.active_video_start) /
                                       static_cast<float>(g.active_video_end - g.active_video_start);
        const float chroma = (x > g.field_width / 2)
                                 ? 12.0F * std::cos(kHalfPi * static_cast<float>(x) +
                                                    kPi * static_cast<float>(y))
                                 : 0.0F;
        ire = luma + chroma;
      }
      field.at(y, x) = ire;
    }
  }
  return field;
}

float MaxLosslessResidual(const hvd::FrameYc& out) {
  float m = 0.0F;
  for (size_t i = 0; i < out.composite.size(); ++i) {
    m = std::max(m, std::fabs(out.composite[i] - (out.luma[i] + out.chroma[i])));
  }
  return m;
}

}  // namespace

void RunTests() {
  FieldGeometry g;
  g.field_width = 260;
  g.field_height = 24;
  g.active_video_start = 20;
  g.active_video_end = 240;
  g.colour_burst_start = 4;
  g.colour_burst_end = 16;
  g.first_active_field_line = 4;
  g.last_active_field_line = 22;

  FieldInput top, bottom;
  top.samples = MakeField(g);
  top.is_first_field = true;
  bottom.samples = MakeField(g);
  bottom.is_first_field = false;

  HvdConfig cfg;
  cfg.cg_iterations = 20;  // keep the test fast

  hvd::HvdEngine engine;

  // --- Baseline: no neighbours, temporal_strength == 0 (the default) —
  // should behave exactly like before this wiring existed. ---------------
  const hvd::FrameYc baseline = engine.DecodeFrame(top, bottom, g, cfg);
  CHECK_NEAR(MaxLosslessResidual(baseline), 0.0, 1e-3);

  // --- With a synthetic "previous frame" neighbour active. Its exact
  // content doesn't matter for this check (see file header): the lossless
  // split is a structural property of VariationalRefine's chi-only
  // elimination of Y, not something that depends on neighbour quality. ---
  hvd::NeighborRawState prev;
  prev.luma = baseline.luma;
  prev.composite = baseline.composite;
  prev.carrier = ComplexPlane(baseline.composite.height(), baseline.composite.width());
  for (size_t i = 0; i < prev.carrier.size(); ++i) {
    const float theta = 0.37F * static_cast<float>(i % 97);
    prev.carrier[i] = Complex(std::cos(theta), std::sin(theta));
  }

  HvdConfig cfg_3d = cfg;
  cfg_3d.enable_temporal = true;
  cfg_3d.temporal_strength = 0.5F;
  cfg_3d.temporal_eps = 0.0F;  // exercise the auto-calibration path too

  const hvd::FrameYc with_neighbor =
      engine.DecodeFrame(top, bottom, g, cfg_3d, {prev});
  CHECK(with_neighbor.luma.height() == baseline.luma.height());
  CHECK(with_neighbor.luma.width() == baseline.luma.width());
  CHECK_NEAR(MaxLosslessResidual(with_neighbor), 0.0, 1e-3);
  for (size_t i = 0; i < with_neighbor.chroma_phasor.size(); ++i) {
    CHECK(std::isfinite(with_neighbor.chroma_phasor[i].real()));
    CHECK(std::isfinite(with_neighbor.chroma_phasor[i].imag()));
  }
}

TEST_MAIN()
