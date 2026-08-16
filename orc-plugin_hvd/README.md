# orc-plugin_hvd_chroma_decoder

A **holographic-variational NTSC/PAL chroma decoder** for
[decode-orc](https://github.com/simoninns/decode-orc), built on the official
`orc-plugin_skeleton` external-plugin template. It is a C++17 port of the
[hvd-decode](../research/reference/) research decoder.

Instead of a comb filter, HVD treats each field as an **off-axis hologram** and
solves Y/C separation as a regularized inverse problem, per frame. The stage
consumes a composite `CVBS_U10_4FSC` source and emits a **lossless Y/C split**
for the host's colour path to render. NTSC and 625-line PAL (see
`../hvd-core/docs/PAL.md`). Slow (seconds/frame).

> Status: the numerical core is unit-tested (5/5 green, real FFTW) and the
> stage/plugin layer compiles against the decode-orc plugin SDK headers. It has
> not yet been link-tested in a full host build or validated on a real capture —
> two logic points (field parity and the vertical line mapping) are isolated in
> `../hvd-core/src/frame_bridge.cpp` for confirmation there. See `docs/PORTING.md`.

## Layout

This directory is the decode-orc **plugin** half of the repository; the
decoder engine lives one level up, in the [`hvd-core`](../hvd-core) git
submodule:

```
src/plugin.h / plugin.cpp        descriptor + the two required entrypoints
src/hvd_chroma_decoder_stage.*   the stage + its Y/C wrapper representation
src/video_writer.*               real-container (.mkv/.mp4/pipe) export, SDK-free but plugin-only
tests/                           SDK tests (stage/entrypoints)
cmake/DecodeOrcPluginSDKHelpers.cmake   orc_add_stage_plugin()
tools/plugin_build_info.cpp      release tooling: prints the compiled descriptor for scripts/package_local.sh
scripts/package_local.sh         packages the binary + writes this platform's manifest fragment
scripts/merge_manifests.sh       combines per-platform fragments into orc-plugin-manifest.yaml
flake.nix                        Nix dev shell matching the host toolchain
instructions.md                  in-app help

../hvd-core/src/frame_bridge.*   CVBS<->IRE<->engine (SDK-free, submodule)
../hvd-core/src/engine/*         numerical core (SDK-free, FFTW only, submodule)
../hvd-core/tests/engine/        engine tests (submodule)
../research/reference/           the Python hvd-decode reference (oracle, NTSC)
../research/reference-pal/       the Python hvd-decode reference (oracle, PAL)
../research/                     the live/evolving Python research package
```

Only `src/plugin.*` and `src/hvd_chroma_decoder_stage.*` depend on the SDK.
Everything under `hvd-core/src/engine/` and `hvd-core/src/frame_bridge.*` is
compiled into the `hvd_core` static library and unit-tested without the host —
see [`../hvd-core/README.md`](../hvd-core/README.md). This plugin's
`CMakeLists.txt` picks it up automatically via `add_subdirectory(../hvd-core)`;
run `git submodule update --init --recursive` from the repository root first.

## The lossless split

The engine guarantees `luma + Re[chroma * carrier] == composite` (the
reference's purity contract); in the 10-bit domain this is
`luma + (chroma - chroma_dc) == composite` to within one code. Saturation / ACC
/ RGB conversion — which would break `Y + C = S` — are therefore not applied in
the default output; they belong to the downstream colour render path.

## Preview modes

Two **preview-only** toggles in the stage parameters. Neither affects
the decode or the export: every export path calls `ReorderToWoven()`
with its default arguments, so written frames always honour the
configured `VideoParameters` crop, exactly as before.

These are **checkboxes in the stage's parameter panel**, not entries in
the preview view dropdown — the dropdown stays stuck on a single "Frame"
entry for this stage on a stock host (see "Why parameters and not host
view options" below). If you are looking for a "Field" option in the
view selector, it is not there and cannot be; use the checkbox.

* **Preview: full raster** — *default ON*. Shows the whole stored
  raster instead of cropping to the active picture: sync, blanking and
  the colour burst appear as monochrome signal in the margins (decoded
  colour stays confined to the active area). Turn OFF to crop to the
  active picture, matching the export.
* **Preview: field view** — *default OFF*. Navigates per FIELD: each
  item is one field at native field height, one row per field line, no
  interpolation. The honest view for per-field artefacts (dropouts,
  weave/field-order errors, PAL V-switch/Hanover checks) that the woven
  frame smears across two fields. Item 2n is frame n's field 1, item
  2n+1 its field 2. This uses the SDK's own `navigation_extent`
  mechanism ("Field 42 of 400" vs "Frame 21 of 200").

Toggling either does **not** re-decode — `set_parameters()` keeps the
frame cache when only preview keys changed.

### Why parameters and not host view options

Because on this decode-orc version the host's view-selector path is
**unreachable for this stage**, verified in `orc/core/preview_renderer.cpp`:
`get_available_outputs()` and `render_output()` both dispatch
`if (capability_stage) {...} else if (custom_renderer) {...}`, and
`get_capability_preview_outputs()` returns exactly one hardcoded output
(`Frame_Field1_First`). A stage implementing `IStagePreviewCapability`
— required for the working histogram and vectorscope panels — never
reaches its own `IStageCustomPreviewRenderer`. This is not specific to
this plugin: `SourceAlignStage`'s own custom options are dead code in
the decode-orc tree today for the identical reason.

This stage implements `IStageCustomPreviewRenderer` anyway (four named
views: Frame / Field 1 / Field 2 / Full raster), ready for the day the
host dispatch is fixed — see
[`patches/0001-preview-renderer-merge-custom-with-capability.patch`](patches/0001-preview-renderer-merge-custom-with-capability.patch),
a ~30-line optional host patch that merges the two option lists. Until
then the parameters above are what actually works, and they are what
the plugin ships enabled.

### Aspect ratio

The preview reports geometry exactly like the SDK's own reference stage
(`PreviewHelpers::make_signal_preview_capability`), so this stage renders
at the same scale as `tbc_source` on the same frame:

* `dar_correction_factor` = `standard_dar_correction(system)`, the
  canonical fixed per-system pixel aspect ratio.
  `cvbs_signal_constants.h` is explicit that it must **not** be
  recomputed from a source's actual active-area values, since that would
  tie display aspect to the chosen window (changing the active area
  would rescale the preview instead of re-framing it).
* `geometry.active_*` describes the **active picture**, not the
  delivered image — the reference reports the active area while
  delivering the full frame.

Both are independent of the view toggles: those change which pixels are
delivered, not the shape of a pixel. Two earlier revisions got this
wrong (first deriving the factor from the delivered dimensions, then
reporting the delivered full-raster size as the active picture, which
left the GUI's "4:3 (Display)" mode reasoning about a 1.73 picture
instead of 1.33). `../hvd-core/tests/engine/preview_aspect_test.cpp` guards both.

### Dropout highlighting is disabled on this stage

The "Dropouts" button greys out, and no plugin change can enable it:
the host hardcodes `dropouts_available = false` for every colour-carrier
output (`get_capability_preview_outputs()` in `preview_renderer.cpp`
constructs the single output with that field literal `false`). The
custom-renderer path *computes* it instead — `!is_chroma_decoder`, which
is true for this stage's `hvd_chroma_decoder` name — so applying
`patches/0001` turns dropout highlighting on together with the extra
views.

### Scope panels in full-raster mode

The host's two analysis tools disagree about carrier `active_*`
semantics, both verified: `vectorscope_analysis.cpp` treats
`active_x_start` as an **absolute plane index**, while the histogram in
`preview_view_registry.cpp` documents the opposite (plane[0] *is* the
first active pixel; only the difference is usable). `ColourFrameCarrier`
has no flag distinguishing the two conventions, so no single choice
satisfies both. This stage marks the whole delivered image active,
which makes both tools iterate the whole delivered image — in
full-raster mode that includes sync/blanking, so the luma distribution
reads low. That is a graceful degradation; the alternative (marking the
true active window) would keep the vectorscope exact but make the
histogram analyse a same-sized rectangle of mostly sync. **Turn
"Preview: full raster" OFF for measurement work.**

## Building

Full instructions (Linux, Windows, engine-only mode, CI): see [BUILD.md](BUILD.md).

First, get the `hvd-core` submodule (the decoder engine this plugin links):

```bash
git submodule update --init --recursive
```

The critical constraint (decode-orc 2.x): the plugin's **toolchain tag must
match the host's exactly**, so build the plugin in the same environment as the
host. The bundled `flake.nix` tracks the same nixpkgs as decode-orc's flake for
exactly this reason.

### Against an in-tree decode-orc checkout (recommended)

```bash
nix develop            # same C++ toolchain as a decode-orc host from its flake
cmake -S . -B build \
    -DORC_INTREE_SDK_DIR=/absolute/path/to/decode-orc \
    -DBUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### Against an installed SDK package

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/decode-orc-install
cmake --build build --parallel
```

FFTW3 single-precision (`fftw3f`) is the engine's only extra dependency; it is
provided by the Nix shell (`fftw`) or found via pkg-config / `-DFFTW3F_LIBRARY`.

Artifact: `orc-plugin_hvd_chroma_decoder_<platform>.{so,dylib,dll}`
(via `scripts/package_local.sh build dist`).

## Parameters

`lambda_c` (chroma smoothness, main knob), `charbonnier_eps` / `chroma_eps`
(edge scales), `structure_coupling` (Y→chroma edge coupling), `cg_iterations`
(0 = fast holographic-init preview), plus `ntsc_j`, `acc`, `monochrome`,
`symmetry_variant` (colour-path / init options).

## Scope

Ported: the validated 2-D woven-frame path. Deferred (marked seams): 3-D /
temporal / noise-reduction / drizzle, dropout handling, PAL. See
`docs/PORTING.md` for the full Python→C++ map, the SDK verification notes, and
the validation plan.

## Licence

GPL-3.0-or-later.
