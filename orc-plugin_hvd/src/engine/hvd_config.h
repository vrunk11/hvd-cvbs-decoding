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

  // --- Engine performance (not in the Python reference — this only exists
  // because of how differently a single C++ process schedules threads
  // compared to a one-shot numpy script) ------------------------------------
  // How many threads FFTW uses internally for each 2-D transform in the
  // preview/single-frame path (parallel-export workers always force this
  // to 1 themselves, regardless of this value — see
  // hvd_chroma_decoder_stage.cpp's worker_fn). FFTW's own thread
  // synchronisation overhead is fixed per call, so on an image this small
  // (~700x480 active picture) it can easily cost more than it saves; 1
  // effectively disables FFTW's internal threading. Tune this live instead
  // of guessing-and-recompiling: try 1 (off), 2, 4, and your full core
  // count, and keep whichever measures fastest — there's no way to predict
  // the right value analytically, it depends on the exact CPU and image
  // size.
  int fft_threads = 4;

  // --- 3-D / temporal --------------------------------------------------
  // Frame-level neighbour extension (decode_frame's own 3D mode in the
  // reference — NOT decode_sequence's separate, field-granularity chunked
  // pipeline, which is a richer but much larger alternative architecture
  // deferred for later; see docs/PORTING.md). Wired into HvdEngine::
  // DecodeFrame via an optional list of previous WOVEN FRAMES' raw state
  // (luma/composite/carrier), motion-compensated with MotionCompensatePrev
  // and fed to VariationalRefine's neighbours parameter.
  //
  // enable_temporal is the actual on/off switch — NOT in the Python
  // reference, which just uses temporal_strength == 0 as "off" (no
  // separate toggle needed there since it's a one-shot script, not a UI
  // with a value you want to keep dialled in while flipping 3D on/off).
  // Decoupling them means temporal_strength can default to an actually-
  // useful working value instead of 0, without that turning 3D on by
  // itself — this default (unlike every other value in this file) is NOT
  // from the reference, since the reference has no equivalent "on but at
  // a sensible strength" state; treat it as a starting point to tune, not
  // a verified-correct constant.
  bool enable_temporal = false;
  float temporal_strength = 1.0F;
  bool bidirectional = true;       // declared for decode_sequence's richer
                                    // path; DecodeFrame only ever sees PAST
                                    // frames (a future frame isn't available
                                    // yet when decoding sequentially)
  int passes = 1;                  // ditto — Gauss-Seidel passes, only
                                    // meaningful for the chunked pipeline
  // 0 => auto-calibrate from the measured composite noise (clip(7*sigma,
  // 4, 20), same formula decode_sequence itself uses) via
  // EstimateNoiseIre — NOT decode_frame's own literal Python (which just
  // uses cfg.temporal_eps as given, no auto-cal at that level): a fixed 0
  // default would otherwise make every neighbour weight collapse to ~0
  // silently (wt = conf*eps_t^2/(rt^2+eps_t^2) degenerates when eps_t==0),
  // exactly the "declared but does nothing" trap this project has hit
  // more than once already — auto-calibrating avoids reintroducing it here.
  float temporal_eps = 0.0F;
  float nr_anchor = 1.0F;
  float nr_eps = 0.0F;
  int nr_radius = 2;
  bool drizzle = false;
  int mc_tile = 32;    // block-matching tile size (px)
  int mc_search = 16;  // block-matching search radius (px)

  // decode_sequence-only fields (the field-granularity chunked pipeline,
  // not yet ported — see docs/PORTING.md). Declared with the reference's
  // own defaults so the config surface is complete and ready, but nothing
  // reads these yet; DecodeFrame's simpler frame-level 3D mode has no
  // concept of "extended" neighbour offsets (its neighbour list is just
  // whatever the caller passes) or of chunking/coherence gating at all.
  bool extended_temporal = true;  // decode_sequence: also use fields f±3
  float coherence_gate = 0.6F;
  int chunk_frames = 6;
  int chunk_overlap = 2;
  bool output_fidelity = true;
  bool psi_init = false;
};

}  // namespace hvd

#endif  // ORC_PLUGIN_HVD_ENGINE_HVD_CONFIG_H_
