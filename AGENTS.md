# AI instructions for this repository

This repository is split into two parts with different rules — read the one
that applies to the files you're touching:

- **`hvd-core/`** (git submodule): the decoder engine. No decode-orc SDK
  dependency, no plugin concerns. Standard C++17 + FFTW/OpenMP hygiene. See
  `hvd-core/README.md`.
- **`orc-plugin_hvd/`**: the decode-orc stage plugin (SDK glue). Has its own
  strict SDK-boundary rules — see
  [`orc-plugin_hvd/AGENTS.md`](orc-plugin_hvd/AGENTS.md), which is the source
  of truth for any change under `orc-plugin_hvd/`.

Cross-cutting rule for both: `orc-plugin_hvd` consumes `hvd-core` only via
its public CMake target (`hvd_core`, from `add_subdirectory`) and its public
headers under `hvd-core/src/`. Never add a decode-orc SDK include or link
dependency inside `hvd-core/` — that boundary is the entire reason the two
are split into separate directories (and separate git repositories).
