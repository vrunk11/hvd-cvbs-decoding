# hvd-core

The holographic-variational NTSC/PAL Y/C separator's numerical engine — the
decoder itself, with **no dependency on decode-orc or any host SDK**.

This is meant to be consumed as a **git submodule** by whatever is doing the
actual decoding I/O — today that's [`orc-plugin_hvd`](../orc-plugin_hvd)
(a decode-orc stage plugin) living alongside this directory in the
[hvd-cvbs-decoding](https://github.com/vrunk11/hvd-cvbs-decoding) repository,
but nothing here assumes that: `hvd_core` is a plain static library any C++17
project can link against.

## What's here

| Path | Contents |
|---|---|
| `src/engine/` | The engine itself: NTSC/PAL geometry, lock-in, 2D FFT, holographic init, the IRLS/CG variational solver, colour, motion, temporal chain, frame sequencing. |
| `src/frame_bridge.*` | SDK-independent bridge between a flat field-sequential CVBS sample buffer and the engine (level conversion, de-weave/re-weave, requantisation). |
| `tests/engine/` | The engine test suite (adjoint identity, lossless split, lock-in, colour, PAL lock-in/carrier, end-to-end bridge, ...). Hermetic — no test framework, see `tests/check.h`. |
| `docs/PAL.md` | How PAL support fits into the engine (effective-carrier formulation, standard dispatch points). |

The Python/NumPy reference material this engine was ported from
(`research/`, including its pinned `research/reference/` and
`research/reference-pal/` snapshots) is **not** in this directory — it lives
as its own sibling directory at the repository root, alongside `hvd-core/`
and `orc-plugin_hvd/`, since it's neither C++ engine code nor plugin/SDK
code. See `orc-plugin_hvd/docs/PORTING.md` for how it maps to `src/engine/`.

## Building standalone

Only needs cmake, a C++17 compiler, and FFTW3 **single precision** (`fftw3f`):

```bash
sudo apt-get install cmake ninja-build g++ libfftw3-dev   # Debian/Ubuntu
# Fedora: sudo dnf install cmake ninja-build gcc-c++ fftw-devel

cmake -S . -B build -G Ninja
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

This is the same suite the plugin's CI runs as its fast gate before the full
SDK matrix — see `orc-plugin_hvd/.github`-adjacent workflow at the repo root.

## Using it as a submodule

From a consuming project:

```bash
git submodule add <hvd-core repo URL> hvd-core
git submodule update --init --recursive
```

```cmake
add_subdirectory(hvd-core)
target_link_libraries(your_target PRIVATE hvd_core)
```

`hvd_core` exposes `src/` as a public include directory, so headers are
reached the same way whether you build this repo standalone or as a
submodule: `#include "engine/engine.h"`, `#include "frame_bridge.h"`, etc.

By default, `HVD_CORE_BUILD_TESTS` only turns on when this directory is the
top-level CMake project (a standalone checkout), so pulling it in via
`add_subdirectory()` doesn't duplicate the consumer's own test targets.
Force it on with `-DHVD_CORE_BUILD_TESTS=ON` if you want the engine suite
built alongside a consumer.

## License

GPL-3.0-or-later — see `LICENSE`.
