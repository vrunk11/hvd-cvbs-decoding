# Porting hvd-decode to a decode-orc stage plugin

Map between the Python/NumPy reference (`reference/`, a copy of hvd-decode) and
this C++ plugin: what was ported, what was deferred, and how it was verified.

## 1. Shape of the port

The reference is NumPy; decode-orc plugins are compiled C++ (C++17, Google
style) loaded as `orc-stage-plugin-*` shared objects. So the port has two
layers, kept strictly separate:

```
src/engine/*        numerical core        — no decode-orc dependency
src/frame_bridge.*  CVBS<->IRE<->engine    — no decode-orc dependency
src/hvd_chroma_decoder_stage.*  stage + wrapper representation (SDK)
src/plugin.cpp      descriptor + entrypoints (SDK macros)
```

`hvd_core` (engine + bridge) links only FFTW and is unit-tested without the
host. Only two files depend on the SDK.

## 2. Verified against the real SDK

The stage layer was written and compile-checked against the actual decode-orc
plugin SDK headers (`orc/sdk/include/orc/...`) and modelled on the in-tree
`mask_line` transform stage. Confirmed points that were previously assumptions:

- Registration: `ORC_STAGE_PLUGIN_DESCRIPTOR(id, version, license, is_core)`
  (4 args) + `ORC_DEFINE_STAGE_PLUGIN(descriptor, Stage)`; stage identity comes
  from `get_node_type_info().stage_name`, and the stage is default-constructed.
- Stage base: `DAGStage` + `ParameterizedStage` + `IStagePreviewCapability`;
  `execute()` wraps the input `VideoFrameRepresentation` in a wrapper
  representation (the mask_line pattern). `set_parameters` is a name→value map
  returning bool; `ParameterValue` is `variant<int32,uint32,double,bool,string>`.
- Samples: `VideoFrameRepresentation::sample_type == int16_t`, 10-bit domain.
- Levels/geometry come from `orc::SourceParameters` (`black_level`,
  `white_level`, `blanking_level`, `chroma_dc_offset`, `active_video_start/end`,
  `first/last_active_frame_line`, `frame_width_nominal`, `frame_height`,
  `system`). The colour-burst window comes from `colour_burst_range(system)`
  (NTSC = samples 72..108) and the field split from `field1_lines(system)`
  (NTSC field 1 = 263 lines).
- Preview: `PreviewHelpers::make_signal_preview_capability(cached_output_)`.
- Instructions: the bare `ORC_STAGE_INSTRUCTIONS_MD` member macro; the build
  copies `instructions.md` next to the `.so`.

Both `hvd_chroma_decoder_stage.cpp` and `plugin.cpp` pass `g++ -fsyntax-only`
against the SDK headers. They were not link-tested, because that needs the full
SDK build (spdlog/fmt/etc.); the CI plugin job covers that on a host with the
SDK provisioned.

### Frame layout (corrected)

A CVBS_U10_4FSC frame is a flat buffer laid out **field-sequentially**: all
`field1_lines` lines of field 1, then field 2 — not even/odd interleaved. Field
1 is the top spatial field. `frame_bridge` splits the buffer by field, runs the
engine's per-field lock-in, and re-weaves the Y/C split back into the same
field-sequential layout. The active-line indices in `SourceParameters` follow
the ld-decode woven convention (`field_line = frame_line / 2`), applied in
`FieldGeometryFromParams`.

## 3. Ported (the 2-D woven-frame path)

Function map (reference line numbers are cross-referenced in the C++ headers):

| Reference (`hvd/decoder.py`, `hvd/tbc.py`) | C++ |
| --- | --- |
| `VideoParameters`, IRE conversion | `engine/ntsc_geometry.*` |
| `estimate_noise_ire` | `EstimateNoiseIre` (defined; not yet wired — see §5) |
| `burst_lockin_phase`, `_tridiag_smooth`, `burst_amplitude_ire` | `engine/lockin.*` |
| carrier synthesis | `MakeCarrier` |
| `_gaussian_lpf_kernel_fft`, `_box_blur`, `holographic_init` | `engine/holographic_init.*` |
| `_dx/_dy/_dxT/_dyT` | `engine/gradients.h` |
| `variational_refine` (IRLS + CG, Charbonnier, structure coupling) | `engine/variational.*` |
| `decode_frame` weave + `prepare_frame` | `engine/engine.*` |
| `yuv_to_rgb16`, ACC gain | `engine/colour.*` |

### The lossless split

`HvdEngine` guarantees `luma + Re[chroma_phasor * carrier] == composite`
(the reference's purity contract). In the 10-bit domain this becomes
`luma + (chroma - chroma_dc) == composite` to within one code, verified by
`frame_bridge_test`. Operations that break `Y + C = S` (chroma gain, ACC,
YUV→RGB) are therefore not applied in the default output — they belong to the
downstream colour render path; `engine/colour.*` ports them for that use.

## 4. Deferred (marked seams, not silently dropped)

- **3-D / temporal / noise-reduction / drizzle** (`decode_sequence` and
  friends). `HvdConfig` keeps the fields declared; the solver's temporal
  neighbour terms are the marked seam in `engine/variational.*`.
- **Dropout / RF-defect handling** beyond what the 2-D prior tolerates.
- **PAL.** NTSC-only, like the reference; the stage advertises
  `VideoFormatCompatibility::NTSC_ONLY`.

## 5. Known gaps to finish

- `EstimateNoiseIre` is ported but not yet consumed by the orchestration, so the
  reference's noise-adaptive parameter scaling is inactive. Wiring it into
  `HvdEngine::DecodeFrame` is a small, local change.
- Numerical **parity vs the Python reference** beyond the lossless contract is
  not yet automated. Plan: use `reference/hvd/encode.py` to synthesize a known
  composite, decode with both `hvd_decode.py` and `hvd_core`, and compare Y and
  chroma magnitude by PSNR at ~1e-3 IRE (float-vs-double and CG stopping
  criteria prevent bit-exactness). Operator-level checks can dump reference
  arrays to `.npy` and compare in a small C++ harness (the approach used during
  development).
- Field parity and the exact vertical line mapping should be eyeballed on a real
  capture; both are isolated in `frame_bridge` for a one-line fix.

## 6. A bug this port caught

The adjoint operators initially returned `-a` for a degenerate single-row/
single-column plane, where the forward difference is the zero operator and the
adjoint must be zero. `tests/engine/gradients_test.cpp` (the
`<Dx a,b> == <a,DxT b>` identity) flagged it; fixed in `engine/gradients.h`.
The CG solver's correctness depends on that identity — keep the test green.

## 7. The THEORY 9e/9f additions (reference "V2")

The reference gained a set of concept additions (see `reference/THEORY.md`
§9e/§9f, written explicitly as the optimisation spec for this port). Status:

| Reference (`hvd/decoder.py`) | C++ | Notes |
| --- | --- | --- |
| `_d1/_d1T/_d2/_d2T` (oriented ±45° prior) | `D1Into/D1TInto/D2Into/D2TInto` in `engine/gradients.h` | adjoint identity in `gradients_test` |
| `diag_prior` weight + total-mass renormalisation | both solvers in `engine/variational.cpp` | `variational_v2_test`; exposed as the stage's "Diagonal chroma prior" |
| `cg_tol` relative gradient-norm early-exit (auto 0.02/0.10) | both solvers | config-only, like the reference |
| `fast` — 2/3 CG cap + loose tol | both solvers | stage's "Fast mode" |
| `fast` — tile-resolution confidence maps | `UpsampleConfidenceFast` + `MotionCompensatePrev(..., fast)` in `engine/temporal.*` | active in `DecodeFrame`'s 3D path |
| integral-image `_box_blur` | `BoxBlur` (now public in `engine/motion.h`, shared by `temporal.cpp`) | equivalence vs the naive form in `motion_v2_test` |
| per-pixel warp vectors once per warp trio | `PerPixelMotion` overloads of `WarpByTiles`/`WarpBilinearTiles`/`WarpByTilesComplex`; used by `MotionCompensatePrev`/`MotionCompensateEnvelope`/`McWarp` | exact rewrite, both modes |
| `verify_motion` (predicted+verified ME, 2 SSD evals) | `VerifyMotion` in `engine/motion.*` | `motion_v2_test` |
| trajectory fit + consensus snap | `FitTrajectory`/`TrajectorySnap` in `engine/motion.*` | `motion_v2_test`; consumed once `decode_sequence` lands (a single-neighbour `DecodeFrame` has one offset — nothing to fit) |
| motion cache, deferred coherence gate | not yet | `decode_sequence`-only logistics; `MotionCompensatePrev`'s `MotionField*` parameter is the caller-side cache hook |

`trajectory_fit` is declared in `HvdConfig` (default on, like the reference)
and gates the snap once the multi-offset chunked pipeline is ported.

## 8. Field audit: combing, "3D does nothing", "fast isn't faster"

A meticulous review triggered by real-use reports. Measured on a 480x700
synthetic scene (moving colour rectangle, 8 px pan, frame carrier flip,
3 IRE composite noise), C++ vs the Python reference on identical data:

* **The port is numerically faithful.** Frame-level 3D at strength 1.0:
  reference 33.90 dB, C++ 33.89 dB (2D baseline 37.34 for both; RMS of the
  two 3D outputs matches to 0.001 IRE). None of the reported symptoms is a
  porting bug.
* **The frame-level 3D term is a liability, and that's inherited.** The
  reference's own test suite validates 3D exclusively through
  `decode_sequence` (`run_tests.py` — "PSNR 3D" is a `decode_sequence`
  call); `decode_frame(prev=...)` is validated nowhere, and measured it
  LOWERS chroma PSNR by 3-5 dB at strength 1.0 **even on static content**
  (48.78 → 43.48 dB noise-free): the neighbour's |dc|² ≈ 4 chroma leverage
  triples the effective data weight against λ_c, lifting chroma noise.
  Mitigations shipped: the InSAR coherence gate is now wired into
  `DecodeFrame` (previously `ComplexCoherence` existed but nothing called
  it — `NeighborRawState` gained the baseband `chroma` member to feed it),
  `temporal_strength` defaults to 0.25, and the UI text states the honest
  status. The REAL fix is porting `decode_sequence`.
* **Combing on moving objects is the frame-weave.** `decode_frame` weaves
  two fields 1/60 s apart before decoding; motion combs by construction,
  in the reference exactly as here. `decode_sequence` (field granularity,
  half-line-grid envelope MC) is the concept's designed answer — same
  missing pipeline.
* **Preview 3D silently never ran.** `decoded()` only had a neighbour if
  frame id-1 happened to be the immediately preceding decode of the same
  session; scrubbing anywhere made "Enable 3D" a no-op. The preview now
  self-primes (decodes id-1 on the spot, caches it).
* **Fast mode is real but was diluted.** The solver alone measures 913 →
  511 ms (~1.8x); but motion estimation ran single-threaded (~240 ms per
  neighbour) and the slow path's new cg_tol=0.02 early-exit already speeds
  the slow mode up. `BmPass` is now parallelised over tile rows (THEORY
  9f: "tiles parallelise trivially"). The larger fast wins (motion cache,
  predicted+verified ME) are `decode_sequence` logistics.

**Update: `decode_sequence` is now ported** — see section 9.

## 9. The decode_sequence port (engine/sequence.{h,cpp})

The field-granularity chunked 3D pipeline is ported end to end:
`PrepareFieldObs` (prepare_field), `SynthReference`, `PsiClosedForm`,
`DecodeFieldWindow` (the per-window body of decode_sequence: noise
self-calibration of temporal_eps/nr_eps, per-pass pair motion with the
trajectory consensus snap, fast mode's shared motion cache +
predicted/verified ME + deferred coherence, the InSAR gate, PSI init at
pass 0, the synth-reference anchor + joint solve from pass 1, and the
output-fidelity luma policy). Bridged through
`DecodeFrameSequenceWindow` (frame_bridge: field split, window ACC,
parity weave back to YcFrameS16 — the physical field-sequential block a
row lands in is chosen by ROW parity, the content by FIELD parity, so an
inverted-weave capture wouldn't scramble the output buffer) and wired
into the stage's temporal export as chunked windows
(`decode_sequence_chunk_and_write_rgb24`), with the frame-level chain
kept as the fallback for Y/C-native sources. The preview stays on the
frame-level path (interactive latency); judge 3D from an export.

Cross-validation (sequence_test.cpp, FFT-free via the WithInits seam,
against the reference's own functions run on identical synthetic
fields): 2D 35.57 vs reference 35.55 dB; 3D 37.47 vs 37.36 dB; anchored
2-pass 42.00 dB (+6.4 dB over 2D); fast within 0.1 dB of slow. A test-
design lesson worth keeping: on a scene with NO Y/C ambiguity
(low-frequency luma), 3D measures ~4 dB WORSE than 2D — in the reference
too — because the neighbour equations can only lift chroma noise when
there is no cross-colour to resolve; the pipeline's validated gains
require carrier-frequency luma content, which the test scene now
contains (and which real footage does).

## 10. Completeness audit + fast-mode field parallelism

A systematic sweep of every DecoderConfig field against C++ consumption
(the "absolument tout" audit) closed the last gaps:

* **bidirectional** — confirmed complete: the offset set includes
  +1/+2/+3, the export windows carry chunk_overlap frames of FUTURE
  context, and the flag gates the offset list exactly as in the
  reference.
* **drizzle** — `DrizzleFrame` is now ported (engine/sequence.cpp):
  vertical super-resolution by robust scatter accumulation over the
  half-line parity phases and sub-pixel motion, with the woven-interp
  fallback where coverage is thin. Validated against the ANALYTIC scene
  evaluated at the super-resolved fine rows (39.3 dB — the recovered
  rows track the continuous truth). Only the export wiring is deferred:
  a drizzled frame is (2*lines*scale) rows tall and the raw headerless
  RGB stream has a fixed per-frame-size contract.
* **monochrome** — the sequence bridge now zeroes chroma at OUTPUT
  (reference's `Cf = zeros_like`), keeping the arbitration-shaped luma.
* **init_lpf_h_mhz / init_lpf_v_cph** — vestigial IN THE REFERENCE too
  (its holographic_init overrides them at every call site); documented,
  not "fixed".
* **ntsc_j / frame_decode** — host-level / structural respectively,
  documented in hvd_config.h.
* per-field `isFirstField` metadata plumbing remains deferred
  (FrameParams carries none; parity falls back to index order, the
  reference's own fallback).

**Fast-mode field parallelism (colored Gauss-Seidel).** The sequential
per-field sweep is the reference's Gauss-Seidel iteration; its data
dependencies only reach `max(largest temporal offset, nr_radius)` fields
(the neighbour equations, the coherence gate, the anchor blend). In fast
mode the sweep is therefore reordered into a COLORED Gauss-Seidel:
fields spaced reach+1 apart share no read/write dependency and solve
concurrently (OpenMP across fields; the solvers' inner loops deactivate
inside the region — one field per core beats one memory-bound loop
across all cores). Same fixed point, a different but convergent
iteration trajectory; measured within the fast-mode envelope on the
regression scene (38.93 vs 39.04 dB slow). The shared fast motion cache
is mutex-guarded (lookups/inserts locked, estimation outside the lock —
a duplicated estimate on a race is benign, a corrupted map is not). The
slow mode keeps the strictly sequential sweep, bit-faithful to the
reference. A second shared-work optimisation: the per-pixel vector
interpolation is computed once per neighbour and shared between the raw
composite/carrier warps and the coherence gate's chi warp.



## 11. Reference audit against THEORY (bugs found in the reference itself)

Porting re-derives every convention, and the derivation disagreed with the
reference in one place that mattered plus several that didn't:

* **Coherence-gate half-line double-compensation** — REAL BUG, fixed in
  both codebases (THEORY 9g has the full writeup). The bilinear-warp
  convention "motion carries no parity component; row_offset supplies it"
  held while the zero-motion margin rule zeroed static cross-parity
  estimates, and broke when the 9e trajectory snap deliberately
  re-introduced h_k into the pairwise vectors: snapped motion +
  row_offset shifted the neighbour chi by a FULL line on exactly the
  static consensus tiles the gate protects. Settled numerically (MAE
  table for zeroed/snapped/raw motion x with/without row_offset; gamma
  0.970 -> 0.999 on static vertically-varying chroma) and locked by a
  sequence_test case. synth_reference correctly KEEPS its row_offset
  (fresh margin-zeroed estimates — exact on static, robust-weighted on
  motion).
* **Comment contradicting the code** on the h_k sign (comment said
  (p_j - p_k)/2, code correctly uses (p_k - p_j)/2) — comment fixed.
* **Dead code in the reference**, removed there: `_laplacian`, the
  unused rows/r0/fr block in drizzle_frame, the unused Y cofactor in
  psi_closed_form (a per-pixel 3x3 cofactor expansion computed and
  thrown away — the C++ port had already silently omitted it).
* **Deliberately kept**: mc_warp / envelope_of /
  motion_compensate_envelope are call-site dead in both codebases but
  are the recorded negative result of the envelope-resampled neighbour
  experiment; annotated, and their C++ equivalents stay tested.
* Routing reminder: the field pipeline serves the 3D export; preview,
  2D export, and the Y/C fallback remain frame-level — the same split
  the reference itself makes (decode_sequence falls back to decode_frame
  when temporal_strength <= 0).

## 12. Field order: measured, not guessed

Build fix first: decode_sequence_chunk_and_write_rgb24 was declared in the
representation's private section; moved to public (trigger() calls it).

On the combing report: the three internal weave conventions (frame-path
writer, ReorderToWoven, sequence bridge) were re-audited and are mutually
consistent for ANY parity of first_active_frame_line (all three index by
absolute frame line). What the frame buffer cannot state by itself is
which SPATIAL half the temporally-first stored field carries. For a
well-formed ld-decode .tbc the FORMAT fixes it (frames are assembled
first-field-first; the first field carries the even frame lines) — there
is genuinely nothing to guess. But the host API surface exposes no
per-field isFirstField to verify a given source against that convention,
so instead of trusting an assumption, the order is now MEASURED from the
signal: under the true order each field's lines interpolate the other's
at +0.5 line, under the inverted order at -0.5 — a half-line vertical
correlation vote on low-passed active luma (the horizontal 4-tap average
keeps the 4fsc chroma carrier from masquerading as vertical structure).
DetectFieldParity votes per frame with a relative margin; the sequence
export majority-votes per window; the frame path resolves per frame,
which is flicker-safe by construction (the detector only abstains on
content where the two weaves are visually identical). "Field order" in
the UI: 0 = Auto (default), 1/2 force. field_order replaces the earlier
blind field1_is_bottom toggle. Covered by field_order_test (both storage
orders, chroma-carrier immunity, flat-content abstention).

## 13. Per-field 2D decode (the residual-chroma-on-motion report)

The reported artifact — uncancelled chroma (dot-crawl/rainbow) on
everything that moves, while static content separates cleanly — is the
frame-weave contaminating the SEPARATION, not the weave order: the woven
frame mixes two instants 1/60 s apart, the vertical chroma prior then
smooths chi across lines from different times, and every chi error on a
moving edge re-emerges as carrier-frequency residue in the fidelity luma
Y = S - Re[chi c]. THEORY section 5 names weave-before-decode "a failure"
for exactly this, and the reference ships decode_field / --per-field as
the remedy — which the C++ 2D path had never adopted.

Wired now, reusing the sequence pipeline (with the temporal terms off it
IS the reference's decode_field, field by field — verified against
decoder.py line 1161):

* "Per-field decode (2D)" stage parameter (HvdConfig::per_field, default
  off = the reference's own frame_decode=True default). The description
  names the symptom so users know when to flip it.
* Export: the field pipeline now serves per_field OR temporal; chunk
  overlap drops to 0 when the temporal terms are off (decoupled fields
  need no context), and the colored sweep degenerates to stride 1 —
  every field of a window solves concurrently.
* Preview: decoded() now routes through the SAME field pipeline
  (per-field 2D, or a mini 3D window of frame id±1 when temporal is on),
  so the preview finally shows export-grade output. The frame-level 3D
  preview chain (prev_frame_id_/prev_frame_state_ and its self-priming)
  is deleted; the frame-level temporal term survives only as the
  Y/C-native export fallback.

## 14. Field mandatory, adaptive strength, cleanup

**No more frame-decode option.** The source is interlaced by definition;
weaving before decoding contaminates the separation across time (section
13), so the composite pipeline is FIELD-granularity everywhere — preview
and export, 2D and 3D. The per_field flag is gone; DecodeFrame survives
only as the internal last-resort fallback (2D, no temporal state), and
the whole frame-level 3D machinery was deleted (engine's prev_frames
path + gate, DecodeFrameBuffer's prev/out_state, the stage's chain
state and the Y/C temporal-chain export fallback, engine_temporal_test).

**Adaptive temporal strength (temporal_strength = 0 = auto, the new
default).** The right strength is a property of the content: measured
repeatedly, 3D resolves Y/C ambiguity where it exists and only lifts
chroma noise where it doesn't. Auto measures the ambiguity per window
from the phase physics: demodulate S by each field's own carrier with a
triangle-7 kernel — a DOUBLE null at f_sc, so ordinary smooth luma
(which the demod shifts to ~pi/2) is rejected across the whole
neighbourhood, not just on the exact bin (the single-null 4-tap leaked
~9% a tenth of a radian off-carrier, enough for a luma ramp to dwarf
real ambiguity); between same-parity fields the carrier has flipped, so
true chroma is coherent while luma leakage flips sign, and
(d_j - d_{j+2})/2 isolates exactly the ambiguous energy. Noise is
subtracted in quadrature; the mapping has NO floor (amb ~ 0 => genuinely
OFF — a forced floor cost ~3 dB on the clean scene). Regression:
ambiguous scene 39.78 dB (beats BOTH fixed extremes: 2D 35.57, fixed-2.0
39.02); clean scene 44.13 dB, bit-identical to pure 2D. Driver
convention: strength < 0 = OFF (the stage passes -1 when the 3D switch
is off), 0 = auto, > 0 = fixed.

**Two real bugs found on the way.** (1) The "2D" export had been
silently running 3D-lite: a positive default strength leaked through the
sequence driver regardless of the Enable-3D switch — which is also why
toggling 3D appeared to change nothing. The switch now forces the
negative sentinel. (2) With passes >= 2, the temporal-off path still
engaged the synth-reference anchor — diverging from the reference (whose
decode_field has neither passes nor anchor) AND coupling the
supposedly-decoupled fields through SynthReference: a data race under
the stride-1 parallel sweep. Temporal-off now means decode_field
semantics: one pass, no anchor.

**Defaults aligned with the reference CLI**: passes = 2 (the anchor
engages when 3D is on, the configuration measuring +6.4 dB), strength
auto.

**Dead code removed in BOTH codebases**: mc_warp / envelope_of /
motion_compensate_envelope (the rejected envelope-resampled experiment;
the negative result stays recorded in prose here and in THEORY 9g) and
their C++ equivalents McWarp / EnvelopeOf / MotionCompensateEnvelope
with their test sections; engine_temporal_test (its subject no longer
exists). A cleanup hazard for the record: a careless cut around the
Python trio briefly took _warp_bilinear_tiles and complex_coherence with
it — the reference suite caught it immediately; both restored verbatim.

## 15. Edge-rainbow investigation + diagnostic maps

Report: soft vertical transitions (a few pixels) and weak diagonals still
rainbow; strength=1 changes little. Two things to know before chasing:
(1) the report predates v9, whose gating fix means every earlier "3D vs
2D" comparison actually compared 3D-lite vs 3D — re-testing on the
current build is a prerequisite; (2) a systematic synthetic hunt did NOT
reproduce a localized edge artifact:

* Static sharp/soft VERTICAL edges (1..4 px ramps), HORIZONTAL edges
  (transitions along Y over 2..8 frame lines — the per-field grid's weak
  axis, since the carrier is vertically IN PHASE within a field), and a
  1:16 weak diagonal, with a realistic triangle-demod init: worst local
  chi error ~1.1 IRE in 2D, and 3D IMPROVES every band (~0.6) — including
  the hypothesized failure mode where the odd-offset half-line bias might
  have let the ±1 equations inject at vertical transitions (it does not:
  the joint solve absorbs it).
* Adding measured-carrier imperfections (2 deg rms per-line lock-in noise
  + drift) multiplies the 2D error x2.4 uniformly and 3D halves it — a
  real quality lever (the burst lock-in path), but still not a localized
  edge artifact.

Since the artifact doesn't reproduce, the build now measures it in situ:
`debug_dir` (Diagnostics directory) makes the export write per-frame PGM
maps of the residual carrier-band energy in the decoded luma — the
visible rainbow itself, bright where separation failed — plus diag.txt
with the decoder's per-chunk decisions (adaptive strength, measured
ambiguity, noise, gates, field-order vote), via the new
SequenceDiagnostics plumbing. An artifact report is now a map + two log
lines instead of a description.

## 16. The rainbow map follow-up: measurement ghost + a real hole

The user's first diagnostic map (v10) led to two findings, one about the
map and one about the decoder — full derivation and the failed-variant
record in THEORY 9h:

* The map itself had a texture false-positive (any luma near f_sc lit
  up, separation failed or not); it now uses the same-parity pair-sum
  demod, which cancels legitimate luma and keeps only residual
  chroma-in-Y. If you kept maps from v10: re-export, the bright curtain
  was the ghost, the bright thin ledges were real.
* Thin horizontal chroma detail (1..4 frame lines — ledges, blinds,
  "bandes horizontales de faible hauteur") was genuinely being destroyed
  BY the 3D: sub-Nyquist for the opposite parity, whose equations voted
  it away through the robust weight's cosine zeros. Fixed with the
  floored baseband-envelope gate on odd offsets (both codebases),
  locked by a sequence_test case (2-frame-line bands now improve ~3x
  under 3D; the reference chart's numbers are bit-restored).

## 17. The 2D symptoms: auto chroma anisotropy

The curtain/ledge report persisted in 2D, where the odd-offset gate (§16)
does not apply. A knob sweep on the thin-line + textured-curtain scene
found the one lever that moves both without a trade-off: chroma_aniso.
With the total prior mass renormalised, aniso redistributes smoothing
from vertical to horizontal — at 1.0, thin horizontal chroma bands
improve ~15%, chroma noise under fsc-adjacent texture ~14-17%, flat ~7%,
while a fine horizontal chroma-stripe control zone moves <3% (IRLS edge
protection carries it). But a fixed 1.0 costs the reference colour-bar
chart 1.1 dB in 2D (bars are the opposite orientation), so the split is
now AUTO (chroma_aniso = 0, the new default in both codebases), resolved
per solve in ResolveChromaAniso / _resolve_chroma_aniso from the init's
dominant chroma-detail orientation. Three measurement traps are baked
into its design, each found the hard way: p98 quantiles (bar transitions
occupy ~4% of columns — p90 reads noise/noise there); line-pair
averaging (the init's vertical energy is largely alternating cross-
colour leak); and a ratio knee at 1.3 calibrated between the measured
families (chart r ~ 1.1 -> 0.5; line/curtain scenes r ~ 1.6-1.9 -> 1.0).
Reference chart numbers are restored exactly (39.55 / 43.23 / 40.78,
suite green); the lambda_c trade-off (lines want it weak, curtain wants
it strong) was measured and deliberately NOT touched. HVD_DEBUG_ANISO=1
prints the measured ratio in both codebases.

For the record, the honest ceiling: in pure 2D these zones sit on the
Y/C ambiguity itself — a single field cannot fully separate texture at
f_sc from chroma. The measured remedy for the curtain and ledge classes
remains 3D on this build (curtain chi 1.40 -> 0.56 anchored, thin lines
~2-3x), which §16 made safe for exactly that content.

## 18. Attempted: auto lambda_c / charbonnier_eps / chroma_eps /
      structure_coupling -- REJECTED, documented for the next attempt

Natural next step after §17: apply the same recipe (sigma = the noise
measure already validated for temporal_eps/nr_eps in decode_sequence,
mapped through a floor-preserving clip so a clean/synthetic field
reproduces the pre-auto constant exactly) to the four priors §17 left
untouched. Implemented and wired into both solvers in both codebases
(ResolveLambdaC / ResolveEdgeEps / ResolveStructureCoupling next to
ResolveChromaAniso; same 0-is-auto convention, -1 for structure_coupling
since 0 is its own legitimate "disabled" value).

First anchor attempt (sigma=0 -> floor) was already wrong before the
knob mattered: the reference SMPTE test disc's OWN noise_ire=0.8 measures
sigma~0.87 through estimate_noise_ire, so anchoring at literal silence
pushed every knob past its floor on the project's own regression chart --
PSNR dropped 39.55 -> 38.15 dB (3D 43.23 -> 41.56, joint 40.78 -> 38.78)
and tripped the `psnrj > psnr + 1.0` regression assert. Fixed by anchoring
at the project's actual baseline noise level instead of zero
(`max(sigma - 1.0, 0)`) -- reference numbers restored exactly.

With that anchor fixed, a direct sweep at 5x the reference noise
(noise_ire=4.0, one knob moved at a time, the other three held at their
fixed reference value) gave a clean, unambiguous answer for all three:

    eps    0.5 -> 3.0   29.06 -> 19.08 dB  (monotonic loss)
    eps_c  1.0 -> 6.0   29.06 -> 25.98 dB  (monotonic loss)
    lambda_c 1.0 -> 3.0 29.06 -> 27.52 dB  (monotonic loss)
    coupling 0 -> 0.8   28.73 -> 29.01 dB  (flat, ~0.25-0.45 is the
                                            broad plateau; no real lever)

Unlike aniso -- a genuine trade-off with a documented win on real
failure-class footage that a fixed setting could not have -- these three
show NO analogous win on the only material available here: raising them
with noise only ever cost fidelity against the known-good chart, on the
one test bed this port can check against. Shipping a noise-scaled auto
for them would be presenting an unvalidated guess as a calibrated
feature, exactly the thing §17's own measured-sweep discipline exists to
prevent. Reverted; `lambda_c` / `charbonnier_eps` / `chroma_eps` /
`structure_coupling` stay plain fixed knobs, defaults unchanged (1.0 /
0.5 / 1.0 / 0.25).

What would actually settle it: real noisy/textured captures, not just a
sharper synthetic chart. UPDATE -- partially settled: re-running the
same sweep on two real photographs re-encoded through hvd.encode
(a dense aerial street scene full of foliage/crowd texture, and a
portrait with fabric texture, both at noise_ire=4.0) reproduces the
SMPTE verdict on real content: eps 0.5->2.5 costs 27.47->24.22 dB,
chroma_eps 1.0->4.0 costs 27.47->26.62, lambda_c 1.0->2.2 costs
27.47->26.74, all monotonic, both images. structure_coupling again
shows only a flat plateau (27.47 @ 0.25 vs 27.50 @ 0.45 -- noise-level).
So the rejection holds for reconstruction fidelity even on real texture;
the remaining untested claim is only the perceptual one (grain
visibility on real ANALOG noise, which is not white/gaussian), which
needs an actual capture .tbc, not a re-encode. Hand-tuning per source
remains the recommendation for these four.

Methodological trap worth recording (it burned an evening): when
building the ground-truth comparison for arbitrary-resolution photos,
the x-resampling index was accidentally clipped to active_width-1
instead of source_width-1, silently corrupting the right ~60% of the
reference into horizontal streaks. Symptoms looked exactly like a
decoder failure: PSNR pinned at ~12 dB regardless of noise level, and
-- far more alarming -- "3D no better than 2D on real content". Both
were artifacts of the broken reference. After the one-character fix,
real-photo numbers are perfectly sane (2D 33.6 dB at noise 0.8, 3D
static +3.7 dB -- same 3D gain as the SMPTE chart). Lesson: a PSNR
that ignores the noise knob is accusing the meter, not the decoder.

## 19. 3D speed audit: where the time actually goes, what's free,
      and why per-tile "selective 3D" has a low ceiling

Profiled on a real 1939x1440 photo re-encoded at noise 2.5 IRE, 3
frames, decode_sequence, this machine's numpy build. Numbers are per
3-frame run; ratios are what matter.

Free wins, config only, ZERO quality cost (measured equal or slightly
better on both PSNR and the flat-zone rainbow metric of Sec. 18):

    default slow path                17.7 s   30.43 dB   1.59 %
    + fast=True (already shipped)    10.3 s   30.43 dB   1.57 %
    + cg_tol=0.3                      7.8 s   30.55 dB   1.54 %
    anchored (ts=0.5 passes=2 nr=1)  32.0 s   30.88 dB   0.78 %
    anchored + fast + cg_tol=0.3     13.3 s   30.89 dB   0.78 %

i.e. 2.3-2.4x today by flipping switches that already exist. The
default cg_tol (0.02 slow / 0.10 fast) is far tighter than the visual
result needs on real content; 0.3 measured slightly BETTER (earlier
stop = mild implicit regularisation). bidirectional=False is NOT free
(-1.1 dB, rainbow 1.9%): the complementary-failure argument in the
docstring is real.

After those switches the profile is: block matching ~40%, solver ~45%
(of which the temporal residual terms proper are only ~5%), warps and
glue the rest. Two structural ideas evaluated against that profile:

(a) Per-tile selective 3D (2D everywhere + 3D only on flagged tiles,
or equivalently 3D pipeline with 90% of tiles degraded to 2D). Ceiling
measured, then root-caused: a prototype on the simple decode_frame
neighbor path produced ZERO quality gain even full-frame -- the
single-neighbor frame-geometry mechanism is not where the 3D win
lives; it comes from the full field-based machinery (4 parity offsets,
trajectory fit, coherence gates, passes). Inside THAT machinery the
skippable per-tile work (temporal residuals in the solver) is ~5% of
runtime, and numpy vectorisation means masked tiles don't return
their cost anyway. Best case is ~10-15% for real seam/complexity
risk. Rejected. Per-tile 3D-gain concentration was also measured
directly (true per-tile 2D-vs-3D error reduction): top 30% of tiles
by the best predictor found (sqrt(luma_HF * chroma_HF), r=0.60) carry
only ~48% of the gain -- the benefit is spread, not spiky, so ANY
spatial gating tops out near half the 3D advantage.

(b) Static-scene shortcut in estimate_motion (skip the coarse-to-fine
search when a cheap zero-motion check passes). Naive version (return
zero vectors) measured -1.7 dB and rainbow 2.05%: for ODD (opposite
parity) offsets, "static" is NOT the zero vector -- it carries the
half-line parity component h_k = (p_k - p_j)/2 that the trajectory
fit normally reinstates. A correct shortcut must be parity-aware and
produce confidences on the same scale as _motion_conf. Plausible
~1.5-3 s on static content, unproven; left as the documented next
candidate rather than shipped broken.

## 20. Investigated: "bidirectional 2D" (scan-direction symmetry) --
      no gain available, and here is the measurement that closes it

Question raised: the 3D path benefits from using both time directions;
could the 2D solve similarly benefit from processing the frame
bottom-up / right-to-left instead of (or in addition to) top-down?

First, the framing: the 2D solve is not a scan. Conjugate gradient
updates every pixel of Y/chi simultaneously each iteration from a
global gradient; there is no top-to-bottom sweep anywhere in the
solver, so there is no "direction" to reverse in the Gauss-Seidel
sense. What COULD exist is hidden directional asymmetry: one-sided
finite differences, prepend/append boundary conventions, or an
init-stage filter that treats up and down differently. Any such bias
is detectable by a strict symmetry test: flip S and phi TOGETHER
(mathematically still a valid composite + carrier pair), decode,
flip back, and compare to the normal decode.

Measured (real photo at noise 2.5, and the SMPTE chart at 0.8):
vertical flip, horizontal flip, and 180-degree decode all land within
0.001-0.005 dB of the normal decode; the mean pixel disagreement is
0.013% of range on the photo and 0.0002% on the chart -- and what
little there is sits in the first/last ~3 border rows (boundary
conditions), ~10-25x above the interior level but still invisible.
Averaging 2 or 4 directional decodes: +0.002 dB for 2-4x the cost.

Conclusion: the operators are already symmetric to well below visual
relevance; there is no directional information the 2D solve is
leaving on the table, so a "bidirectional 2D" mode would quadruple
cost for nothing. The 3D case is fundamentally different: past and
future fields carry NEW measurements (different carrier phase over
the same scene), whereas re-scanning the same field in another order
re-reads the same equations. Closed. (If border rows ever matter for
some downstream use, the cheap fix is reflective padding at the top/
bottom boundary, not a second decode direction.)

## 21. Selective 3D (reference implementation) -- shipped, with its
      honest operating envelope

Second attempt, this time inside the REAL pipeline, addressing why the
first one failed (Sec. 19a: the single-neighbor decode_frame mechanism
carries none of the 3D win). Two pieces, Python reference only (not
yet ported to C++):

1. decode_sequence(..., roi=(fy0, fy1, x0, x1)): the entire field-based
   machinery -- block matching, trajectory fit, parity gates, solver,
   weave -- runs on a rectangular crop in field coordinates. One crop
   at the top of the chunk loop; everything downstream is shape-
   agnostic (one hardcoded active_width in the output weave fixed to
   the crop width).

2. decode_sequence_selective(...): full-frame 2D per frame + the
   cropped full-3D on the single bounding box of the most ambiguous
   tiles, feather-blended. Ambiguity = the Sec. 19 validated score
   (sqrt(luma_HF x chroma_HF) of the init, r=0.60 vs true gain);
   threshold 4x median (a fixed quantile flags noise tiles on uniform
   content); a >=2-of-8-neighbors density filter drops isolated noise
   tiles (one stray anywhere inflates the single bbox to the full
   frame); border-normalised box smoothing (plain 'same' convolution
   zero-pads and flags the entire first row/column). Falls back to
   plain 2D when the flagged bbox exceeds max_area (default 0.45) --
   a crop that big saves nothing.

Measured, flat scene + textured insert (the fan-grille / Santana
class), noise 2.5, fast + cg_tol=0.3, 3 frames:

    2D                    3.6 s   32.16 dB   rainbow(insert) 2.36 %
    selective (28% area)  6.5 s   32.38 dB   rainbow(insert) 1.48 %
    full 3D               9.4 s   31.36 dB   rainbow(insert) 1.34 %

i.e. ~87% of the 3D rainbow fix at ~69% of full-3D wall time -- and
the best global PSNR of the three, because the flat majority keeps the
2D solve that full 3D was mildly degrading at this noise level.

Envelope, stated plainly: this pays off ONLY on localized-ambiguity
content. On the diffuse real-photo case the detector correctly returns
box=None and the mode degrades to plain 2D (measured) -- by design,
per the Sec. 19 measurement that diffuse content puts ~half the 3D
gain outside any 30% crop. The floor cost is one full 2D decode per
frame; content that is ALL texture should just run full 3D.
