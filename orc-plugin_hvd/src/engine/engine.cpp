// SPDX-License-Identifier: GPL-3.0-or-later

#include "engine/engine.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "engine/colour.h"
#include "engine/fft2d.h"
#include "engine/holographic_init.h"
#include "engine/lockin.h"
#include "engine/variational.h"

namespace hvd {

namespace {

// Extract the active picture of one field: lines
// [first_active_field_line, last_active_line) x columns
// [active_video_start, active_video_end).
Plane ActivePicture(const Plane& field, const FieldGeometry& g) {
  const int fal = g.first_active_field_line;
  const int lal = g.last_active_line();
  const int a0 = g.active_video_start;
  const int a1 = g.active_video_end;
  const int lines = std::max(0, lal - fal);
  const int width = std::max(0, a1 - a0);
  Plane out(lines, width);
  for (int y = 0; y < lines; ++y)
    for (int x = 0; x < width; ++x) out.at(y, x) = field.at(fal + y, a0 + x);
  return out;
}

}  // namespace

HvdEngine::HvdEngine() : fft_(std::make_unique<Fft2d>()) {}
HvdEngine::~HvdEngine() = default;

FrameYc HvdEngine::DecodeFrame(const FieldInput& first, const FieldInput& second,
                               const FieldGeometry& g, const HvdConfig& cfg) {
  // Order the two fields so the first (top) field lands on the even rows. The
  // caller may hand them over in capture order; is_first_field decides.
  const FieldInput* top = &first;
  const FieldInput* bot = &second;
  if (!first.is_first_field && second.is_first_field) std::swap(top, bot);

  // Per-field burst phase (full field), then restrict to the active lines.
  const std::vector<float> theta_top = BurstLockinPhase(top->samples, g);
  const std::vector<float> theta_bot = BurstLockinPhase(bot->samples, g);

  const Plane s_top = ActivePicture(top->samples, g);
  const Plane s_bot = ActivePicture(bot->samples, g);

  const int lines = s_top.height();  // active lines per field
  const int width = s_top.width();
  const int fal = g.first_active_field_line;

  // Weave into frame geometry: even rows = top field, odd rows = bottom field.
  Plane s(2 * lines, width);
  std::vector<float> theta(2 * lines, 0.0F);
  for (int y = 0; y < lines; ++y) {
    for (int x = 0; x < width; ++x) {
      s.at(2 * y, x) = s_top.at(y, x);
      s.at(2 * y + 1, x) = s_bot.at(y, x);
    }
    theta[2 * y] = theta_top[fal + y];
    theta[2 * y + 1] = theta_bot[fal + y];
  }

  // Carrier over the woven active picture (absolute-sample phase per row).
  const ComplexPlane carrier = MakeCarrier(theta, g);

  // Holographic init, then (optionally) variational refinement.
  HoloInit init = HolographicInit(s, carrier, g, cfg, fft_.get());
  Plane luma;
  ComplexPlane chi;
  if (cfg.cg_iterations > 0 && !cfg.monochrome) {
    RefineResult r = VariationalRefine(s, carrier, init.chroma, cfg);
    luma = std::move(r.luma);
    chi = std::move(r.chroma);
  } else {
    luma = std::move(init.luma);
    chi = std::move(init.chroma);
  }

  if (cfg.monochrome) {
    for (size_t i = 0; i < chi.size(); ++i) chi[i] = Complex{0.0F, 0.0F};
    // chroma == 0 => luma must equal the composite for the split to hold.
    for (size_t i = 0; i < luma.size(); ++i) luma[i] = s[i];
  }

  // Lossless Y/C split: chroma = S - Y (== Re[chi * carrier] when not mono).
  FrameYc out;
  out.composite = s;
  out.luma = luma;
  out.chroma = Plane(2 * lines, width);
  for (size_t i = 0; i < out.chroma.size(); ++i) out.chroma[i] = s[i] - luma[i];
  out.chroma_phasor = std::move(chi);

  // ACC gain (for the colour path only; does not touch the split above).
  if (cfg.acc) {
    const float a_top = BurstAmplitudeIre(top->samples, g);
    const float a_bot = BurstAmplitudeIre(bot->samples, g);
    out.acc_gain = AccGain(0.5F * (a_top + a_bot));
  }
  return out;
}

}  // namespace hvd
