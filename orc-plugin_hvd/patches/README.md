# Host patches

These patches target **decode-orc itself** (the `orc/core/` host sources),
not this plugin. They are optional, not applied automatically by anything
in this repo, and not required to build or use the plugin — see the
"Preview modes" section of the main README for exactly what functionality
each one unlocks and what it costs if left unapplied.

Apply against a decode-orc checkout:

```bash
cd /path/to/decode-orc
git apply /path/to/orc-plugin_hvd/patches/0001-preview-renderer-merge-custom-with-capability.patch
```

## 0001-preview-renderer-merge-custom-with-capability.patch

**Problem** (verified against `orc/core/preview_renderer.cpp`, not
inferred): `get_available_outputs()` and `render_output()` both dispatch
with `if (capability_stage) {...} else if (custom_renderer) {...}`. Any
stage implementing `IStagePreviewCapability` — which is required to get
the working carrier-backed histogram and vectorscope views — never
reaches the `else if`, so its `IStageCustomPreviewRenderer` options are
silently dead code. This affects `SourceAlignStage` in the decode-orc tree
itself (its "Aligned Source N" options never appear), not just
third-party plugins like this one. The SDK explicitly documents the two
interfaces as meant to be implemented "alongside" each other
(`stage_custom_preview_renderer.h`), so this is a dispatch bug, not a
deliberate restriction — the interface contract and the dispatch code
disagree.

**Fix**: in the colour-domain branch of both functions, merge the
carrier's one hardcoded output with the stage's custom options (in
`get_available_outputs()`) instead of returning immediately, and route
`render_output()` to the custom renderer when `option_id` names one of
those extra views, falling through to the existing carrier path
otherwise (default/empty `option_id` — so every stage that currently
relies on the single carrier view keeps working identically). ~30 lines,
two functions, no signature or ABI changes, no changes to any plugin.

**Verified**: hand-traced against the exact quoted source (both hunks
matched verbatim before patching); the patched file was NOT compiled
against the full host (core/ pulls internal deps — `dag_executor.h`,
libpng — outside this environment's reach). Build and run it against a
real decode-orc checkout before relying on it; `orc-tests/core/unit/`
already has coverage for both functions that this patch should keep
green (add a merged-output case for a stage implementing both
interfaces, e.g. by extending `SourceAlignStage`'s own preview tests).

**Without this patch**, this plugin's `IStageCustomPreviewRenderer`
implementation (Field 1 / Field 2 / Full raster views) is implemented,
tested at the unit level (see `tests/`), and completely inert in the
GUI — only the default "Frame" carrier view is reachable, same as
before the PAL work. That is the current, honest state of the shipped
plugin; this patch is what turns the extra views on.
