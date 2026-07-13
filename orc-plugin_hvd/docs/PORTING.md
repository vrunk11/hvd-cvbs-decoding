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
