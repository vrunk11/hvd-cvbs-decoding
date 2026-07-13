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

  // Oriented (+/-45 deg) chroma prior weight, relative to mu_v and
  // distance-normalised by 1/2 (a diagonal step is sqrt(2) px); 0 disables.
  // Measured in the reference as a trade-off, not a win: -1.0 dB on
  // axis-aligned sharp chroma (SMPTE) vs +2.0 dB on diagonal cross-colour
  // torture (zoneplate) — hence off by default; a documented dial for
  // diagonal-artifact-heavy material (fine weaves, venetian blinds). The
  // horizontal/vertical/diagonal weights are renormalised together so the
  // TOTAL chroma prior mass is unchanged: the oriented terms redistribute
  // smoothing across directions, they don't add more of it. (THEORY 9e.)
  float diag_prior = 0.0F;

  // --- Solver budget ------------------------------------------------------
  // Total conjugate-gradient iterations across all IRLS outer passes.
  // 0 => pure holographic reconstruction (fast preview, no refinement).
  int cg_iterations = 60;
  // Number of IRLS (lagged-diffusivity) outer re-weightings.
  int irls_outer = 4;
  // CG relative gradient-norm early-exit: stop an inner CG loop once
  // ||g||^2 < tol^2 * ||g0||^2. 0 = auto (0.02 slow / 0.10 fast), matching
  // the reference's cg_tol. Iteration counts stay the CEILING; this lets
  // already-converged solves return early instead of burning the budget.
  float cg_tol = 0.0F;
  // FAST MODE (THEORY 9f — written as the optimisation spec for this port):
  // same algorithm, cheaper logistics. In this engine it currently wires up
  // the parts the frame-level path exercises — the 2/3 CG-budget cap, the
  // looser auto cg_tol (0.10), and tile-resolution confidence maps in
  // MotionCompensatePrev (bilinear interpolation of the 24x24-ish tile map
  // instead of a full-res squared upsample + radius-8 blur; ~256x cheaper,
  // visually identical). The decode_sequence-only components (shared
  // motion cache, predicted+verified ME for long offsets, deferred
  // coherence) have their building blocks ported (VerifyMotion,
  // FitTrajectory/TrajectorySnap, the MotionField* precompute hooks) and
  // activate when that pipeline lands. Reference measurement: >=2x wall
  // clock, never worse than 0.2 dB.
  bool fast = false;

  // --- Holographic init bandwidths (sideband crop) ------------------------
  // NOTE: vestigial in the REFERENCE too — its holographic_init overrides
  // these at every call site (dataclasses.replace with 0.8/120, 1.8/30 for
  // the two Dubois variants and 1.3/60 for the symmetry certifier), so the
  // user-facing values are never consumed there either. Kept declared for
  // config-surface parity; the C++ init hardcodes the same three pairs.
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
  // FIELD ORDER. The TBC stores fields in TEMPORAL order; for a
  // well-formed ld-decode .tbc the FORMAT already fixes the spatial
  // mapping (frames are assembled first-field-first and the first field
  // carries the even frame lines) — so there is nothing to guess. The
  // host API surface, however, exposes no per-field isFirstField, so
  // rather than trusting an assumption across every host/capture, AUTO
  // (default) MEASURES the order from the signal itself: under the true
  // order, each field's lines interpolate the other field's at +0.5 line,
  // under the inverted order at -0.5 — a deterministic half-line vertical
  // correlation test (DetectFieldParity in frame_bridge), decided per
  // frame with a relative margin and falling back to the format
  // convention (field 1 = top) when content has no vertical detail to
  // vote with. 1/2 force the order manually (diagnosis: with the wrong
  // order, static horizontal edges serrate one line and motion combs even
  // through a player's deinterlacer).
  int field_order = 0;  // 0 = auto (measured), 1 = field 1 top, 2 = field 1 bottom
  // DECODE UNIT for the 2D path. false (reference default frame_decode=True):
  // weave the two fields and decode at frame geometry — maximum vertical
  // chroma resolution on STATIC content, but the woven frame mixes two
  // instants 1/60 s apart, and the vertical chroma prior then smooths chi
  // across lines from DIFFERENT times: on anything moving, chi is wrong
  // along the serrated edges and the residual shows as uncancelled chroma
  // carrier (dot-crawl/rainbow) in the fidelity luma Y = S - Re[chi c].
  // THEORY section 5 calls weaving-before-decoding "a failure" for exactly
  // this reason; the reference ships decode_field / --per-field as the
  // remedy. true: decode each field independently (the sequence pipeline
  // with no temporal terms — same math as the reference's decode_field),
  // motion-clean at the cost of vertical chroma resolution on static
  // detail. The 3D path is per-field always.
  bool per_field = false;
  bool enable_temporal = false;
  // Measured (synthetic pan + static scenes, harness in docs/PORTING.md
  // section 8): at strength 1.0 the frame-level neighbour term LOWERS
  // chroma PSNR by 3-5 dB even on static content — the |dc|^2 ~= 4 chroma
  // leverage triples the effective data weight against lambda_c, lifting
  // chroma noise; the Python reference's frame-level path behaves
  // identically (its validated 3D gains come from decode_sequence, not
  // from this term). 0.25 keeps the term subtle instead of harmful; the
  // previous 1.0 default was a guess this project explicitly flagged as
  // "a starting point to tune, not a verified-correct constant".
  float temporal_strength = 0.25F;
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
  //
  // trajectory_fit (THEORY 9e): fit ONE per-tile velocity across all the
  // sequence pipeline's temporal offsets (six noisy measurements of one
  // physical motion) and snap agreeing pairwise vectors onto k*v under
  // consensus; disagreement is preserved (occlusion/acceleration is
  // signal). The engine primitives (FitTrajectory/TrajectorySnap in
  // motion.h) are ported and unit-tested; this flag gates their use once
  // the multi-offset pipeline exists. A single-neighbour DecodeFrame has
  // only one offset, so there is no trajectory to fit yet.
  bool trajectory_fit = true;
  bool extended_temporal = true;  // decode_sequence: also use fields f±3
  float coherence_gate = 0.6F;
  int chunk_frames = 6;
  int chunk_overlap = 2;
  bool output_fidelity = true;
  bool psi_init = false;
};

}  // namespace hvd

#endif  // ORC_PLUGIN_HVD_ENGINE_HVD_CONFIG_H_
