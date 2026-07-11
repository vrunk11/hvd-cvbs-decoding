# AI Instructions for the Decode-Orc Plugin Skeleton

Use these instructions for all changes in this repository.

## Source of Truth
- Follow the latest decode-orc patterns from main:
  - https://github.com/simoninns/decode-orc/tree/main/orc/plugins
  - https://github.com/simoninns/decode-orc/tree/main/orc/plugins/stages
- For SDK usage and ABI contracts, use only public SDK headers under:
  - `orc/sdk/include/orc/plugin/`
  - `orc/sdk/include/orc/stage/`

## SDK Boundary (Hard Rule)
- This repository is an external plugin project.
- Only use the provided plugin SDK surface.
- Do not include private host/core headers from decode-orc internals.
- Do not link against host internal libraries or targets (for example `orc-core`, `orc-common`, `orc-metadata`, presenter/gui/cli targets).
- Do not call non-SDK helper APIs that are not part of the plugin SDK contract.

## Implementation Pattern Rules
- Match core plugin structure and naming patterns from decode-orc main.
- Keep plugin metadata in `plugin.h` and verify consistency in entrypoints.
- Keep stage behavior deterministic and compatible with SDK types.
- Prefer simple, portable implementations over host-internal shortcuts.

## CMake and Build Rules
- Build plugin targets through the SDK helper/SDK target only.
- For in-tree SDK use, treat decode-orc as a header/API source only; do not depend on internal compiled host libraries.
- For installed SDK use, rely on `find_package(decode-orc-plugin-sdk REQUIRED)`.
- Never add direct linkage to internal decode-orc targets just to satisfy convenience helpers.

## CI/CD Safety Rules
- Keep workflow SDK checkout pointed at decode-orc main by default.
- Do not pin `ORC_SDK_REF` to ephemeral or unknown refs.
- If pinning is required, use an existing branch/tag/commit and update references in the workflow intentionally.
- Preserve cross-platform matrix behavior (Linux/macOS/Windows) and packaging outputs.
- Preserve artifact names and expected output paths unless a coordinated CI change is made.

## Required Validation Before Finalizing Changes
- Configure and build with in-tree SDK:
  - `cmake -S . -B build -DORC_INTREE_SDK_DIR=<decode-orc checkout> -DBUILD_TESTS=ON`
  - `cmake --build build --parallel`
  - `ctest --test-dir build --output-on-failure -C Release`
- Ensure no private decode-orc include/link usage is introduced.
- Ensure workflow changes remain compatible with all matrix OS entries.

## Change Discipline
- Keep changes minimal and scoped.
- Do not introduce API/ABI assumptions that are not explicitly exposed by the SDK.
- If a desired optimization requires host-internal APIs, do not implement it here; keep the SDK-compliant approach.
