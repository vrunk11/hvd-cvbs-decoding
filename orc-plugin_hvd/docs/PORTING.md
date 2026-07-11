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
