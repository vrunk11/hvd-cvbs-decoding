# Building the HVD chroma decoder plugin

This document covers every supported way to build `orc-plugin_hvd` on
Linux and Windows, plus how the GitHub CI mirrors those paths. macOS
follows the Linux/Nix instructions unchanged.

There are two distinct things you can build:

| Target | Needs | Use it for |
|---|---|---|
| **Engine only** (`hvd_core` + engine tests) | cmake, a C++17 compiler, FFTW3 *single precision* | numerical work, PAL/NTSC engine changes, fast CI |
| **Full plugin** (`orc-stage-plugin-hvd-chroma-decoder.{so,dll,dylib}`) | all of the above **+ the decode-orc plugin SDK** | something the decode-orc host actually loads |

The engine is deliberately SDK-free: everything under `src/engine/` and
`src/frame_bridge.*` compiles and unit-tests without the host. All PAL
support lives at that level too, so PAL work never requires the SDK.

---

## 0. The one rule that actually matters (full plugin only)

Since decode-orc 2.x, the host **refuses to load a plugin whose
toolchain tag differs from its own** — compiler family, major version,
and C++ standard library must match exactly. Practical consequences:

* On Linux/macOS, build inside `nix develop`: the bundled `flake.nix`
  tracks the same nixpkgs generation as decode-orc's flake, so the
  default gcc matches the host's by construction. Building with your
  distro's gcc will *compile* fine and then be rejected at load time.
* On Windows, use the same MSVC major version the host release was
  built with.
* When the host moves (e.g. v2.0.0 → v2.1.0), bump `ORC_SDK_REF` in
  `.github/workflows/ci.yml` and re-check `flake.lock` against the
  host's — a nixpkgs generation change means a new gcc means a new tag.

---

## 1. Linux

### 1.1 Engine only (no SDK, no Nix) — the 2-minute path

```bash
sudo apt-get install cmake ninja-build g++ libfftw3-dev   # Debian/Ubuntu
# Fedora: sudo dnf install cmake ninja-build gcc-c++ fftw-devel

cmake -S . -B build-engine -G Ninja -DHVD_ENGINE_ONLY=ON -DBUILD_TESTS=ON
cmake --build build-engine --parallel
ctest --test-dir build-engine --output-on-failure
```

Notes:
* `libfftw3-dev` ships **both** precisions; the engine links the
  single-precision `fftw3f`. If you install FFTW another way, make sure
  the `f` variant exists — the configure step fails loudly otherwise.
* This runs the full engine suite, including the PAL tests:
  `pal_lockin_test` (swinging-burst joint phase+parity estimation + the
  effective-carrier hologram identity) and `pal_aniso_group_test` (the
  standard-dependent cross-colour leak cancellation in AUTO
  `chroma_aniso`). If you touch anything in `src/engine/`, run this
  first.

### 1.2 Full plugin against an in-tree decode-orc checkout (recommended)

```bash
git clone https://github.com/simoninns/decode-orc /path/to/decode-orc
# match the host version you intend to load the plugin into:
git -C /path/to/decode-orc checkout v2.0.0

nix develop            # same toolchain as a decode-orc host built from its flake
cmake -S . -B build \
    -DORC_INTREE_SDK_DIR=/path/to/decode-orc \
    -DBUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./scripts/package_local.sh build dist   # -> dist/orc-plugin_hvd_chroma_decoder_linux.so
```

Inside `nix develop`, cmake, ninja, pkg-config, fmt, spdlog, FFTW and
FFmpeg (libav*, for the .mkv/.mp4 export path — see `video_writer.cpp`)
are all provided; nothing to apt-install.

### 1.3 Full plugin without Nix (at your own risk)

Possible — install `libfftw3-dev`, `libfmt-dev`,
`libavformat-dev libavcodec-dev libavutil-dev libswscale-dev` (for the
.mkv/.mp4/pipe export path — MPEG-4 Part 2 and FFV1 are both native to
avcodec, no separate x264/GPL codec library needed), point
`-DORC_INTREE_SDK_DIR` at the checkout — but the resulting `.so`
carries *your* gcc's toolchain tag. Only useful when you also built the
decode-orc host yourself with the same compiler. For a host installed
from release packages, use 1.2.

---

## 2. Windows

MSVC + vcpkg, matching the CI. From a *x64 Native Tools* prompt (or any
shell with VS 2022's cl on PATH):

```powershell
git clone https://github.com/simoninns/decode-orc C:\src\decode-orc
git -C C:\src\decode-orc checkout v2.0.0

cmake -S . -B build `
    -DORC_INTREE_SDK_DIR=C:/src/decode-orc `
    -DBUILD_TESTS=ON `
    -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release --parallel
ctest --test-dir build --output-on-failure -C Release
```

Notes:
* **Dependencies are declared in `vcpkg.json`** (manifest mode):
  `fftw3` with the `threads` feature, `fmt` for the SDK headers, and
  `ffmpeg` (avcodec/avformat/swscale) for the .mkv/.mp4/pipe export
  path in `video_writer.cpp` -- deliberately *without* the `x264`
  feature: libx264 is a whole separate autotools project vcpkg builds
  on the side, and it's known to fail its `./configure` step under
  some MinGW/MSYS2 setups (unrelated to this plugin's own code). The
  .mp4 path uses avcodec's own native MPEG-4 Part 2 encoder instead --
  bigger files than H.264 at the same visual quality, but nothing
  extra to build. With the toolchain file passed, vcpkg installs
  everything at configure time automatically — do **not** `vcpkg
  install` anything by hand; manifest mode ignores classic-mode
  packages (this bit the CI once: an explicit `vcpkg install fmt` step
  looked load-bearing and fed nothing).
* vcpkg's `fftw3[threads]` compiles the threading symbols directly into
  `fftw3f` (no separate `fftw3f_threads` DLL); the build system detects
  this by *linking a probe*, not by looking for a file, so both
  packaging styles work.
* Engine-only also works on Windows: replace the two `-DORC_…` options
  with `-DHVD_ENGINE_ONLY=ON` and skip the decode-orc checkout.
* MinGW-w64 builds work (the CMake handles the OpenMP link-flag quirk
  explicitly — see the long comment in `CMakeLists.txt`), but remember
  rule 0: the host must be MinGW-built too.

---

## 3. Build options reference

| Option | Default | Meaning |
|---|---|---|
| `HVD_ENGINE_ONLY` | `OFF` | Build `hvd_core` + engine tests only; no SDK required |
| `ORC_INTREE_SDK_DIR` | — | Path to a decode-orc checkout (in-tree SDK headers) |
| `BUILD_TESTS` | `ON` | Build ctest targets (SDK tests skipped in engine-only mode) |
| `ORC_PLUGIN_VERSION` | project version | Version embedded in the descriptor (CI derives it from `v*` tags) |
| `FFTW3F_LIBRARY` / `FFTW3F_INCLUDE_DIR` | auto | Manual FFTW paths when pkg-config can't find `fftw3f` |

Threading: the solver's hot loops use OpenMP (forced `-fopenmp` on
GCC/Clang at compile *and* link — see the CMake comment for the MinGW
war story), frame-level export parallelism uses `std::thread`, and
FFTW's own threading is enabled when the probe finds the symbols.

---

## 4. GitHub CI (`.github/workflows/ci.yml`)

Two-stage pipeline:

1. **`engine-tests`** — Ubuntu, no Nix, no SDK: apt FFTW, configure
   with `-DHVD_ENGINE_ONLY=ON`, run the engine suite (~2 min). Gates
   the expensive matrix; this is where an engine/PAL regression fails
   first and cheapest.
2. **`build-test-package`** — the full matrix, only after
   `engine-tests` passes:
   * `ubuntu-latest` + `macos-latest` via **Nix** (toolchain-tag
     correctness), `windows-latest` via **MSVC + vcpkg manifest**;
   * checks out `simoninns/decode-orc` at `ORC_SDK_REF` for the
     in-tree SDK, verifies the SDK headers exist, enforces the SDK
     boundary (`scripts/check_sdk_boundary.sh`);
   * builds, runs the full ctest suite, packages the artifact per
     platform (`scripts/package_local.sh` on POSIX, DLL harvest on
     Windows), uploads `plugin-{linux,macos,windows}`.
3. **`publish-release`** — on `v*` tags, attaches all packaged
   artifacts to the GitHub release. The plugin version is derived from
   the tag automatically.

To release: `git tag v0.2.0 && git push origin v0.2.0`. To track a new
host version: bump `ORC_SDK_REF`, and if the host's flake moved to a
new nixpkgs generation, update this repo's `flake.lock` to match (rule
0 again).

---

## 5. Troubleshooting

* **"FFTW3 single-precision (fftw3f) not found"** — you installed only
  double precision. Debian's `libfftw3-dev` includes both; elsewhere
  install the `f`/single variant or pass `FFTW3F_LIBRARY` +
  `FFTW3F_INCLUDE_DIR`.
* **Host refuses to load the plugin** — toolchain tag mismatch: rebuild
  under `nix develop` (Linux/macOS) or with the host's MSVC version
  (Windows). Rule 0.
* **`undefined reference to GOMP_parallel` at link** — you bypassed the
  provided CMake (which passes `-fopenmp` at link time explicitly for
  exactly this reason). Don't.
* **Plugin uses ~1 core** — build without OpenMP does that; check the
  configure log for the OpenMP warning.
* **`'int8_t' was not declared in this scope` (or similar for `uint16_t`,
  `std::max`, ...) on MinGW/MSVC, but the same source builds on Linux** —
  libstdc++ pulls many standard headers in transitively; MinGW's and
  MSVC's libraries do not, so a missing `#include <cstdint>` /
  `<algorithm>` is invisible on Linux and fatal elsewhere. If you add
  code, include what you use rather than relying on a Linux build to
  tell you. (This bit the PAL port once: `int8_t` in the V-switch
  vectors, fixed across the tree.)
* **First PAL build against a real SDK** — two identifiers were written
  offline and must be confirmed once:
  `VideoFormatCompatibility::ANY` (plugin.h + descriptor) and
  `VideoSystem::PAL` (stage). `docs/PAL.md` has the details; each is a
  one-line fix if the SDK spells it differently.
  The third item, the PAL `chroma_phase_deg` starting value, is resolved:
  it is **0**, the same as NTSC. See `docs/PAL.md` §3.
