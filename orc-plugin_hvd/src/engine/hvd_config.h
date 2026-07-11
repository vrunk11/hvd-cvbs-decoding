// SPDX-License-Identifier: GPL-3.0-or-later
//
// hvd_config.h — tunable knobs for the holographic-variational decoder.
//
// This is the C++ mirror of `DecoderConfig` in `reference/hvd/decoder.py`.
// Only the fields exercised by the ported 2-D woven-frame path are wired up in
// this first port; the 3-D / noise-reduction / drizzle fields are declared and
// documented so the temporal extensions can be added behind the existing engine
// interface without touching the SDK layer (see docs/PORTING.md).
//
// Defaults are copied verbatim from the reference so that, for the same input,
// the C++ 2-D decode matches the Python 2-D decode within numerical tolerance.

#ifndef ORC_PLUGIN_HVD_ENGINE_HVD_CONFIG_H_
#define ORC_PLUGIN_HVD_ENGINE_HVD_CONFIG_H_

namespace hvd {

struct HvdConfig {
  // --- Arbitration priors (the heart of the method) -----------------------
  // lambda_c: chroma-smoothness vs luma-plausibility trade-off. Higher =
  // smoother chroma / less rainbowing; lower = sharper chroma / less dot crawl
  // pushed into luma.
  float lambda_c = 1.0F;
  // Chroma is broader horizontally than vertically: mu_h = chroma_aniso *
  // lambda_c, mu_v = lambda_c.
  float chroma_aniso = 0.5F;
  // Charbonnier edge-preservation scales (in IRE) for the luma and chroma
  // priors respectively.
  float charbonnier_eps = 0.5F;
  float chroma_eps = 1.0F;
  // Parallel-level-sets coupling: open the chroma diffusivity wherever the
  // residual luma sees an edge (kills hanging dots at vertical transitions).
  float structure_coupling = 0.25F;

  // --- Solver budget ------------------------------------------------------
  // Total conjugate-gradient iterations across all IRLS outer passes.
  // 0 => pure holographic reconstruction (fast preview, no refinement).
  int cg_iterations = 60;
  // Number of IRLS (lagged-diffusivity) outer re-weightings.
  int irls_outer = 4;

  // --- Holographic init bandwidths (sideband crop) ------------------------
  float init_lpf_h_mhz = 1.3F;    // horizontal chroma bandwidth (MHz)
  float init_lpf_v_cph = 60.0F;   // vertical bandwidth (cycles / picture height)
  // Enable the spectral-symmetry ("Transform NTSC, repaired") init variant.
  bool symmetry_variant = false;

  // --- Colour / levels ----------------------------------------------------
  // NOTE: no ntsc_j flag here. The Python reference has one because it
  // works from raw TBC files with no host calibration; in this plugin,
  // orc::SourceParameters::black_level already reflects the real measured
  // pedestal for this specific capture (0 IRE for NTSC-J, 7.5 IRE for
  // standard NTSC — see orc_source_parameters.h's black_level_override /
  // has_nonstandard_values), and frame_bridge.cpp already uses it
  // throughout. Re-applying a black-level shift here would double-correct
  // NTSC-J sources and silently corrupt standard NTSC-M ones.
  // Automatic Color Control: calibrate saturation from the measured burst
  // amplitude (nominal 20 IRE), as every analogue TV does.
  bool acc = true;
  float chroma_gain = 1.0F;
  bool monochrome = false;
  // Chroma phase correction, in degrees, applied directly to the burst-
  // locked phase reference (theta) BEFORE it's used to build the carrier —
  // same idea as Comb::Configuration::chromaPhase in the classic decoder
  // (comb.cpp's transformIQ, theta = (33 + chromaPhase) * pi/180), except
  // here it's injected at the actual phase-reference stage instead of
  // rotating U/V after the fact. 0 = no correction. The recovered chroma
  // has been persistently 180 deg off since the Python reference, so 180 is
  // the known-good starting point; treat it as tunable per-capture rather
  // than assuming every source needs exactly 180.
  float chroma_phase_deg = 180.0F;

  // --- Geometry -----------------------------------------------------------
  // Weave both fields into frame geometry before decoding (default, best
  // quality on static material). false => legacy per-field decode.
  bool frame_decode = true;

  // --- 3-D / temporal (declared for the next iteration; not yet used) -----
  // See docs/PORTING.md §"Deferred". Kept here so the engine interface is
  // stable when the temporal terms are added.
  float temporal_strength = 0.0F;  // 0 => pure 2-D
  bool bidirectional = true;
  int passes = 1;
  float temporal_eps = 0.0F;  // 0 => auto-calibrate to measured noise
  float nr_anchor = 1.0F;
  float nr_eps = 0.0F;
  int nr_radius = 2;
  bool drizzle = false;
};

}  // namespace hvd

#endif  // ORC_PLUGIN_HVD_ENGINE_HVD_CONFIG_H_
