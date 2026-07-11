# orc-plugin_hvd_chroma_decoder

A **holographic-variational NTSC chroma decoder** for
[decode-orc](https://github.com/simoninns/decode-orc), built on the official
`orc-plugin_skeleton` external-plugin template. It is a C++17 port of the
[hvd-decode](reference/) research decoder.

Instead of a comb filter, HVD treats each field as an **off-axis hologram** and
solves Y/C separation as a regularized inverse problem, per frame. The stage
consumes a composite `CVBS_U10_4FSC` source and emits a **lossless Y/C split**
for the host's colour path to render. NTSC only. Slow (seconds/frame).

> Status: the numerical core is unit-tested (5/5 green, real FFTW) and the
> stage/plugin layer compiles against the decode-orc plugin SDK headers. It has
> not yet been link-tested in a full host build or validated on a real capture —
> two logic points (field parity and the vertical line mapping) are isolated in
> `src/frame_bridge.cpp` for confirmation there. See `docs/PORTING.md`.

## Layout

This repository follows the skeleton template's structure:

```
src/plugin.h / plugin.cpp        descriptor + the two required entrypoints
src/hvd_chroma_decoder_stage.*   the stage + its Y/C wrapper representation
src/frame_bridge.*               CVBS<->IRE<->engine (SDK-free)
src/engine/*                     numerical core (SDK-free, FFTW only)
tests/                           SDK tests (stage/entrypoints) + engine tests
cmake/DecodeOrcPluginSDKHelpers.cmake   orc_add_stage_plugin()
flake.nix                        Nix dev shell matching the host toolchain
instructions.md                  in-app help
reference/                       the Python hvd-decode reference (oracle)
```

Only `src/plugin.*` and `src/hvd_chroma_decoder_stage.*` depend on the SDK.
Everything under `src/engine/` and `src/frame_bridge.*` is compiled into the
`hvd_core` static library and unit-tested without the host.

## The lossless split

The engine guarantees `luma + Re[chroma * carrier] == composite` (the
reference's purity contract); in the 10-bit domain this is
`luma + (chroma - chroma_dc) == composite` to within one code. Saturation / ACC
/ RGB conversion — which would break `Y + C = S` — are therefore not applied in
the default output; they belong to the downstream colour render path.

## Building

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
