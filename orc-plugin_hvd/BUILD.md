# Building the HVD chroma decoder plugin

This document covers every supported way to build `orc-plugin_hvd` on
Linux and Windows, plus how the GitHub CI mirrors those paths. macOS
follows the Linux/Nix instructions unchanged.

This plugin is the **decode-orc SDK glue** around a separate, SDK-free
decoder engine — [`hvd-core`](../hvd-core), consumed here as a **git
submodule** at `../hvd-core`. If you just want to build/test the decoder
engine itself (no SDK, no plugin), see
[`../hvd-core/README.md`](../hvd-core/README.md) instead — it's a plain
standalone CMake project.

| Target | Needs | Use it for |
|---|---|---|
| **Engine only** (`hvd_core` + engine tests) | cmake, a C++17 compiler, FFTW3 *single precision* | numerical work, PAL/NTSC engine changes, fast CI — build `../hvd-core` directly |
| **Full plugin** (`orc-stage-plugin-hvd-chroma-decoder.{so,dll,dylib}`) | all of the above **+ the decode-orc plugin SDK** | something the decode-orc host actually loads |

---

## 0. Get the submodule first

```bash
git submodule update --init --recursive
```

If `hvd-core/CMakeLists.txt` doesn't exist, configuring this plugin fails
fast with a message telling you to run the command above (or to pass
`-DHVD_CORE_DIR=<path>` at a local checkout instead).

---

## 1. The one rule that actually matters (full plugin only)

Since decode-orc 2.x, the host **refuses to load a plugin whose
toolchain tag differs from its own** — compiler family, major version,
and C++ standard library must match exactly. Practical consequences:

* On Linux/macOS, build inside `nix develop`: the bundled `flake.nix`
  tracks the same nixpkgs generation as decode-orc's flake, so the
  default gcc matches the host's by construction. Building with your
  distro's gcc will *compile* fine and then be rejected at load time.
* On Windows, use the same MSVC major version the host release was
  built with (§3.1) — or MinGW-w64 (§3.2) *only* if you're loading into
  a MinGW-built host of your own, not an official release.
* When the host moves (e.g. v2.0.0 → v2.1.0), bump `ORC_SDK_REF` in
  `.github/workflows/ci.yml` and re-check `flake.lock` against the
  host's — a nixpkgs generation change means a new gcc means a new tag.

---

## 2. Linux

### 2.1 Engine only (no SDK, no Nix, no plugin) — the 2-minute path

```bash
sudo apt-get install cmake ninja-build g++ libfftw3-dev   # Debian/Ubuntu
# Fedora: sudo dnf install cmake ninja-build gcc-c++ fftw-devel

cd ../hvd-core
cmake -S . -B build-engine -G Ninja
cmake --build build-engine --parallel
ctest --test-dir build-engine --output-on-failure
```

This runs the full engine suite, including the PAL tests:
`pal_lockin_test` (swinging-burst joint phase+parity estimation + the
effective-carrier hologram identity) and `pal_aniso_group_test` (the
standard-dependent cross-colour leak cancellation in AUTO
`chroma_aniso`). If you touch anything in `hvd-core/src/engine/`, run
this first — see `../hvd-core/README.md` for details.

### 2.2 Full plugin against an in-tree decode-orc checkout (recommended)

```bash
git submodule update --init --recursive   # get hvd-core

git clone https://github.com/simoninns/decode-orc /path/to/decode-orc
# match the host version you intend to load the plugin into:
git -C /path/to/decode-orc checkout v2.0.0

nix develop            # same toolchain as a decode-orc host built from its flake
cmake -S . -B build \
    -DORC_INTREE_SDK_DIR=/path/to/decode-orc \
    -DBUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./scripts/package_local.sh build dist   # -> dist/orc-plugin_hvd_chroma_decoder_linux_abi<N>.so
```

Inside `nix develop`, cmake, ninja, pkg-config, fmt, spdlog, FFTW and
FFmpeg (libav*, for the .mkv/.mp4 export path — see `video_writer.cpp`)
are all provided; nothing to apt-install.

### 2.3 Full plugin without Nix (at your own risk)

Possible — install `libfftw3-dev`, `libfmt-dev`,
`libavformat-dev libavcodec-dev libavutil-dev libswscale-dev` (for the
.mkv/.mp4/pipe export path — MPEG-4 Part 2 and FFV1 are both native to
avcodec, no separate x264/GPL codec library needed), point
`-DORC_INTREE_SDK_DIR` at the checkout — but the resulting `.so`
carries *your* gcc's toolchain tag. Only useful when you also built the
decode-orc host yourself with the same compiler. For a host installed
from release packages, use 2.2.

---

## 3. Windows

Two supported paths: MSVC (what the official decode-orc releases and this
plugin's own CI use) and MinGW-w64. **They are not interchangeable outputs**
— see the toolchain-tag warning at the end of 3.2 before reaching for MinGW.

### 3.1 MSVC + vcpkg (matches the official CI/host)

From a *x64 Native Tools* prompt (or any shell with VS 2022's cl on PATH):

```powershell
git submodule update --init --recursive

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
* Engine-only also works on Windows: `cd ..\hvd-core` and configure
  that directory directly (see 2.1) instead of this one — no
  `-DORC_…` options, no decode-orc checkout.

### 3.2 MinGW-w64 + vcpkg (`x64-mingw-dynamic`)

Also supported — the CMake already handles the OpenMP link-flag quirk for
GCC/Clang explicitly (see the long comment in `../hvd-core/CMakeLists.txt`),
and MinGW is GCC, so it takes the same branch as Linux there. From an MSYS2
MinGW64 shell (or any shell with a MinGW-w64 g++ on PATH):

```bash
git submodule update --init --recursive

cmake -S . -B build -G "MinGW Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DORC_INTREE_SDK_DIR="C:/path/to/decode-orc" \
    -DCMAKE_TOOLCHAIN_FILE="C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake" \
    -DVCPKG_TARGET_TRIPLET=x64-mingw-dynamic
cmake --build build --parallel
```

Any local `vcpkg` checkout works for `-DCMAKE_TOOLCHAIN_FILE` — it doesn't
need to live inside your `decode-orc` checkout, that's just one convenient
place to keep it if you're already cloning both. `vcpkg.json` manifest mode
installs `x64-mingw-dynamic` builds of `fftw3`/`fmt`/`ffmpeg` the same way it
does `x64-windows` ones for MSVC; no extra flags needed beyond the triplet.

> **This produces a binary the official decode-orc Windows releases will
> reject at load time.** The decode-orc project distributes an **MSVC**-built
> host on Windows (`arch: win64_msvc2022_64` in its own release CI), so its
> toolchain tag is `msvc19/msvc-stl/release-crt`. A MinGW build's tag is
> `gcc<N>/libstdc++` — same shape as a Linux build, different from any MSVC
> host — and the loader requires an **exact** string match (rule 1 above). A
> MinGW build here is genuinely useful for local development (compiling,
> `ctest`, iterating on `hvd-core`/the stage without wrestling with MSVC), but
> don't package or publish it as a release asset unless you are *also*
> building and distributing your own MinGW-built `decode-orc` host to load it
> — which is not what the official project ships.

---

## 4. Build options reference

| Option | Where | Default | Meaning |
|---|---|---|---|
| `HVD_CORE_DIR` | this project | `../hvd-core` | Path to the hvd-core checkout (submodule by default) |
| `HVD_CORE_BUILD_TESTS` | hvd-core | `ON` only when hvd-core is the top-level project | Build hvd-core's own engine tests |
| `ORC_INTREE_SDK_DIR` | this project | — | Path to a decode-orc checkout (in-tree SDK headers) |
| `BUILD_TESTS` | this project | `ON` | Build the plugin's own SDK-level ctest targets |
| `ORC_PLUGIN_VERSION` | this project | project version | Version embedded in the descriptor (CI derives it from `v*` tags) |
| `FFTW3F_LIBRARY` / `FFTW3F_INCLUDE_DIR` | hvd-core | auto | Manual FFTW paths when pkg-config can't find `fftw3f` |

Threading: the solver's hot loops use OpenMP (forced `-fopenmp` on
GCC/Clang at compile *and* link — see the CMake comment in
`../hvd-core/CMakeLists.txt` for the MinGW war story), frame-level
export parallelism uses `std::thread`, and FFTW's own threading is
deliberately not used (see the same file).

---

## 5. GitHub CI (`.github/workflows/ci.yml`, at the repository root)

Two-stage pipeline:

1. **`engine-tests`** — Ubuntu, no Nix, no SDK, no plugin: apt FFTW,
   configures and builds `hvd-core/` directly with
   `-DHVD_CORE_BUILD_TESTS=ON`, runs the engine suite (~2 min). Gates
   the expensive matrix; this is where an engine/PAL regression fails
   first and cheapest.
2. **`build-test-package`** — the full matrix, only after
   `engine-tests` passes:
   * checks out this repository **with submodules** (`hvd-core`
     included) plus `simoninns/decode-orc` at `ORC_SDK_REF`;
   * `ubuntu-latest` + `macos-latest` via **Nix** (toolchain-tag
     correctness), `windows-latest` via **MSVC + vcpkg manifest**;
   * verifies the SDK headers exist, enforces the SDK boundary
     (`scripts/check_sdk_boundary.sh`);
   * configures this directory (`hvd-core` is picked up automatically
     at `../hvd-core`), builds, runs the full ctest suite;
   * packages the artifact and writes that platform's fragment of the
     release manifest in one step (`scripts/package_local.sh`, under
     `bash` on all three OSes including Windows via Git Bash — see
     below), then self-checks the fragment merges
     (`scripts/merge_manifests.sh`) before uploading
     `plugin-{linux,macos,windows}` (binary + fragment together).
3. **`publish-release`** — on `v*` tags: merges the three platforms'
   manifest fragments into `orc-plugin-manifest.yaml`
   (`scripts/merge_manifests.sh`) and attaches it, alongside all
   packaged binaries, to the GitHub release. The plugin version comes
   from the fragments, which got it from the compiled descriptor, which
   got it from the tag — one source of truth end to end.

To release: `git tag v0.2.0 && git push origin v0.2.0`. To track a new
host version: bump `ORC_SDK_REF`, and if the host's flake moved to a
new nixpkgs generation, update this repo's `flake.lock` to match (rule
1 again). To track a new `hvd-core` version: bump the submodule commit
(`cd hvd-core && git pull && cd .. && git add hvd-core && git commit`).

### The release manifest (`orc-plugin-manifest.yaml`)

Every release the decode-orc curated plugin index can offer needs this file:
it declares, per platform, the exact binary filename, the host ABI it was
built against, its toolchain tag, and a sha256 digest — without it the host
refuses to browse, install, or update to the release. See
[Plugin Publishing Guide §3](https://github.com/simoninns/decode-orc/blob/main/docs/technical/plugin-publishing.md)
in the decode-orc repo for the full rationale and schema.

This tooling mirrors [`orc-plugin_skeleton`](https://github.com/simoninns/orc-plugin_skeleton)
(the official decode-orc external-plugin template) field-for-field and
script-for-script — `tools/plugin_build_info.cpp`,
`scripts/package_local.sh`, `scripts/merge_manifests.sh` — so anything
documented there for the manifest applies here unchanged.

**Never hand-write or hand-edit this file.** Every field on it (`abi`,
`toolchain_tag`, `sha256`) is either build-environment-dependent or
binary-dependent, and a value that's wrong or stale is *worse* than a missing
one — it tells a host the wrong thing about a binary it hasn't inspected yet.
It's generated fully automatically, in two steps, exactly so no one has to
keep a hand-maintained ABI number in sync with the SDK:

1. **Per platform** (`scripts/package_local.sh`, run in each matrix job
   right after building): finds `orc-plugin-build-info` — a tiny
   executable linked only against the SDK headers, built unconditionally
   alongside the plugin (see `tools/plugin_build_info.cpp`) — and runs it.
   It prints `plugin_id` / `plugin_version` / `stage_name` / `abi` /
   `toolchain_tag` straight out of `kPluginDescriptor`, i.e. exactly what
   that platform's just-built binary actually embeds, never re-derived by
   a second, independent calculation that could drift from it. Combined
   with the packaged file's own `sha256`, this becomes a complete,
   valid, single-artifact `plugin-manifest-<platform>.yaml` — good enough
   to publish as-is for a one-platform release — uploaded alongside the
   binary. (Fragment names deliberately don't start with `orc-plugin_`, so
   the release-asset glob in `publish-release` never picks them up.)
2. **Once, in `publish-release`** (`scripts/merge_manifests.sh`): after
   all three matrix jobs finish, combines their three fragments (each job
   only sees its own platform) into the final `orc-plugin-manifest.yaml`
   and uploads it as a release asset next to the binaries. Every matrix
   job also self-checks this step on its own single fragment (the "Check
   manifest fragment merges" CI step) so a bug in the merge script fails a
   PR instead of only surfacing at release time.

To regenerate one locally (e.g. to sanity-check the format after touching
either script): build the plugin normally, then

```bash
./scripts/package_local.sh build dist
# -> dist/orc-plugin_hvd_chroma_decoder_linux_abi<N>.so
# -> dist/plugin-manifest-linux.yaml   (already a complete, valid manifest)

# Merging one platform's fragment with itself, or with others you've built
# separately, produces the same shape publish-release does:
./scripts/merge_manifests.sh orc-plugin-manifest.yaml dist/plugin-manifest-linux.yaml
```

---

## 6. Troubleshooting

* **"hvd-core not found at '...'"** — you forgot
  `git submodule update --init --recursive`, or you're building from a
  tarball/zip download that doesn't carry submodules (GitHub's
  "Download ZIP" does not fetch submodule contents — clone with git
  instead, or pass `-DHVD_CORE_DIR=<path>` at a manual checkout).
* **"FFTW3 single-precision (fftw3f) not found"** — you installed only
  double precision. Debian's `libfftw3-dev` includes both; elsewhere
  install the `f`/single variant or pass `FFTW3F_LIBRARY` +
  `FFTW3F_INCLUDE_DIR`.
* **Host refuses to load the plugin** — toolchain tag mismatch: rebuild
  under `nix develop` (Linux/macOS) or with the host's MSVC version
  (Windows). Rule 1.
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
  `VideoSystem::PAL` (stage). `../hvd-core/docs/PAL.md` has the details;
  each is a one-line fix if the SDK spells it differently.
  The third item, the PAL `chroma_phase_deg` starting value, is resolved:
  it is **0**, the same as NTSC. See `../hvd-core/docs/PAL.md` §3.
