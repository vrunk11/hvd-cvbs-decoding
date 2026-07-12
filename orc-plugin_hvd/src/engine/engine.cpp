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

constexpr float kPi = 3.14159265358979323846F;

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

// Shared by DecodeFrame and DecodeChromaOnly: order the two fields (top on
// even rows), weave their active pictures into frame geometry, lock onto
// burst phase per field, apply the chroma-phase correction, and build the
// carrier. `s` is the woven active picture (composite for DecodeFrame,
// chroma-only for DecodeChromaOnly — the maths that follows doesn't care
// which, only the caller's interpretation of the result differs).
struct WovenFrame {
  Plane s;
  ComplexPlane carrier;
  const FieldInput* top = nullptr;
  const FieldInput* bot = nullptr;
};

WovenFrame WeaveAndBuildCarrier(const FieldInput& first, const FieldInput& second,
                                const FieldGeometry& g, const HvdConfig& cfg) {
  const FieldInput* top = &first;
  const FieldInput* bot = &second;
  if (!first.is_first_field && second.is_first_field) std::swap(top, bot);

  const std::vector<float> theta_top = BurstLockinPhase(top->samples, g);
  const std::vector<float> theta_bot = BurstLockinPhase(bot->samples, g);

  const Plane s_top = ActivePicture(top->samples, g);
  const Plane s_bot = ActivePicture(bot->samples, g);

  const int lines = s_top.height();
  const int width = s_top.width();
  const int fal = g.first_active_field_line;

  WovenFrame w;
  w.top = top;
  w.bot = bot;
  w.s = Plane(2 * lines, width);
  std::vector<float> theta(2 * lines, 0.0F);
  for (int y = 0; y < lines; ++y) {
    for (int x = 0; x < width; ++x) {
      w.s.at(2 * y, x) = s_top.at(y, x);
      w.s.at(2 * y + 1, x) = s_bot.at(y, x);
    }
    theta[2 * y] = theta_top[fal + y];
    theta[2 * y + 1] = theta_bot[fal + y];
  }

  // Chroma phase correction: rotate the burst-locked reference itself
  // (same spirit as Comb::FrameBuffer::transformIQ's `theta = (33 +
  // chromaPhase) * pi/180`), so the solver below decomposes the composite
  // against an already-corrected carrier instead of us rotating chi
  // afterwards. A pure sign flip (the old 180 deg-only fix) is just the
  // special case cos(pi)=-1, sin(pi)=0 of this, and is sign-invariant (+180
  // and -180 are the same rotation) — that's the only value this has
  // actually been validated at. For anything other than 180, the recovered
  // phasor's phase moves opposite to the reference's (lock-in convention:
  // recovered phase = signal phase - LO phase), hence the minus sign below;
  // dial a small value and confirm hue rotates the expected direction on a
  // colour-bar test before trusting it away from 180.
  if (cfg.chroma_phase_deg != 0.0F) {
    const float offset = -cfg.chroma_phase_deg * kPi / 180.0F;
    for (float& t : theta) t += offset;
  }

  w.carrier = MakeCarrier(theta, g);
  return w;
}

}  // namespace

HvdEngine::HvdEngine() : fft_(std::make_unique<Fft2d>()) {}
HvdEngine::~HvdEngine() = default;

void HvdEngine::SetFftThreads(int n) { fft_->SetThreadCount(n); }

FrameYc HvdEngine::DecodeFrame(const FieldInput& first, const FieldInput& second,
                               const FieldGeometry& g, const HvdConfig& cfg,
                               const std::vector<NeighborRawState>& prev_frames) {
  const WovenFrame w = WeaveAndBuildCarrier(first, second, g, cfg);
  const Plane& s = w.s;
  const ComplexPlane& carrier = w.carrier;

  // Holographic init, then (optionally) variational refinement.
  HoloInit init = HolographicInit(s, carrier, g, cfg, fft_.get());
  Plane luma;
  ComplexPlane chi;
  if (cfg.cg_iterations > 0 && !cfg.monochrome) {
    std::vector<NeighborTerm> neighbors;
    HvdConfig effective_cfg = cfg;
    if (!prev_frames.empty() && cfg.enable_temporal) {
      // Auto-calibrate temporal_eps from measured composite noise if the
      // caller left it at 0 — see hvd_config.h's doc comment for why a
      // bare 0 would otherwise silently zero out every neighbour weight.
      if (effective_cfg.temporal_eps <= 0.0F) {
        const float sigma = EstimateNoiseIre(s);
        effective_cfg.temporal_eps = std::clamp(7.0F * sigma, 4.0F, 20.0F);
      }
      neighbors.reserve(prev_frames.size());
      for (const NeighborRawState& prev : prev_frames) {
        const MotionCompensatedResult mc =
            MotionCompensatePrev(prev, init.luma, cfg.mc_tile, cfg.mc_search);
        neighbors.push_back(NeighborTerm{mc.composite, mc.carrier, mc.confidence});
      }
    }
    RefineResult r = VariationalRefine(s, carrier, init.chroma, effective_cfg, neighbors);
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
  out.carrier = carrier;
  out.luma = luma;
  out.chroma = Plane(s.height(), s.width());
  for (size_t i = 0; i < out.chroma.size(); ++i) out.chroma[i] = s[i] - luma[i];
  out.chroma_phasor = std::move(chi);

  // ACC gain (for the colour path only; does not touch the split above).
  if (cfg.acc) {
    const float a_top = BurstAmplitudeIre(w.top->samples, g);
    const float a_bot = BurstAmplitudeIre(w.bot->samples, g);
    out.acc_gain = AccGain(0.5F * (a_top + a_bot));
  }
  return out;
}

FrameYc HvdEngine::DecodeChromaOnly(const FieldInput& first,
                                    const FieldInput& second,
                                    const FieldGeometry& g,
                                    const HvdConfig& cfg) {
  const WovenFrame w = WeaveAndBuildCarrier(first, second, g, cfg);
  const Plane& s = w.s;  // chroma-only, signed, zero-mean — no luma mixed in
  const ComplexPlane& carrier = w.carrier;

  // No separation problem to solve (the source already separated Y from C),
  // so just the holographic bandwidth crop — no IRLS/CG arbitration needed.
  // `cfg.monochrome`/`cfg.cg_iterations` intentionally don't apply here:
  // there is no luma-vs-chroma trade-off to make on an input that's
  // already pure chroma.
  const HoloInit init = HolographicInit(s, carrier, g, cfg, fft_.get());

  FrameYc out;
  out.composite = s;
  out.carrier = carrier;
  out.luma = Plane(s.height(), s.width());  // meaningless; caller ignores it
  out.chroma_phasor = init.chroma;
  out.chroma = Plane(s.height(), s.width());
  for (size_t i = 0; i < out.chroma.size(); ++i) {
    out.chroma[i] = (out.chroma_phasor[i] * carrier[i]).real();
  }

  if (cfg.acc) {
    const float a_top = BurstAmplitudeIre(w.top->samples, g);
    const float a_bot = BurstAmplitudeIre(w.bot->samples, g);
    out.acc_gain = AccGain(0.5F * (a_top + a_bot));
  }
  return out;
}

}  // namespace hvd
