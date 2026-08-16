# hvd-cvbs-decoding

Holographic-variational NTSC/PAL Y/C separator: a decode-orc stage plugin
built on top of a standalone, SDK-free decoder engine.

This repository has three top-level parts:

| Directory | What | Depends on decode-orc SDK? |
|---|---|---|
| [`hvd-core/`](hvd-core) | The C++ decoder engine (git submodule) — numerical core, frame bridge, engine tests. | **No.** Buildable and testable entirely on its own. |
| [`orc-plugin_hvd/`](orc-plugin_hvd) | The decode-orc stage plugin: SDK glue, GUI/CLI-facing stage, video export, packaging. | Yes — this is the part the decode-orc host actually loads. |
| [`research/`](research) | The Python/NumPy research package and its pinned `reference/`, `reference-pal/` snapshots (the oracle `hvd-core` was ported from) — kept for porting-audit purposes. | No. |

`orc-plugin_hvd` consumes `hvd-core` via CMake's `add_subdirectory()`, resolved
by default at `../hvd-core` relative to the plugin (i.e. the submodule
checked out at the repository root).

## Getting the code

```bash
git clone --recurse-submodules https://github.com/vrunk11/hvd-cvbs-decoding.git
# or, if already cloned without --recurse-submodules:
git submodule update --init --recursive
```

## Building

* **Just the decoder engine** (no SDK, no plugin, fastest path — numerical
  work, PAL/NTSC changes): see [`hvd-core/README.md`](hvd-core/README.md).
* **The full plugin** (what the decode-orc host loads): see
  [`orc-plugin_hvd/BUILD.md`](orc-plugin_hvd/BUILD.md).

## Why the split

`hvd-core` has no decode-orc dependency at all — it's a plain C++17 static
library plus a hermetic test suite. Splitting it into its own repository
means:

* it can be built, tested, and iterated on (including by people not working
  on the decode-orc integration) without checking out the SDK at all;
* it can be reused by another host or a standalone CLI tool later without
  dragging plugin/SDK code along;
* the plugin repository (`orc-plugin_hvd/`) stays exactly what it says on the
  tin — decode-orc glue code — and nothing else.

See `orc-plugin_hvd/docs/PORTING.md` for the original engine/plugin boundary
this split follows (it already documented `src/engine/` and
`src/frame_bridge.*` as "no decode-orc dependency" before the two were
physically separated).

## License

GPL-3.0-or-later — see `LICENSE` in each directory.
