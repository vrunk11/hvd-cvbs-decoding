#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-.}"
cd "$repo_root"

fail=0

say() {
  printf '%s\n' "$*"
}

run_search() {
  local pattern="$1"
  shift
  local files=("$@")

  if command -v rg >/dev/null 2>&1; then
    rg -n -e "$pattern" "${files[@]}" || true
  else
    grep -R -n -E "$pattern" "${files[@]}" || true
  fi
}

# Restrict scanning to source/build files where violations matter.
include_scan_paths=(src tests)
link_scan_paths=(CMakeLists.txt tests cmake)

say "[sdk-boundary] Scanning for private decode-orc include usage..."
include_hits="$(run_search '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<\"](orc/core/|.*/orc/core/)' "${include_scan_paths[@]}")"
if [[ -n "$include_hits" ]]; then
  say "$include_hits"
  say "[sdk-boundary] ERROR: Private host/core include path detected. Use only SDK headers under orc/sdk/include/orc/plugin/."
  fail=1
fi

say "[sdk-boundary] Scanning for private decode-orc target linkage..."
link_hits="$(run_search 'target_link_libraries[[:space:]]*\([^\)]*(orc::?(core|common|metadata|presenters)|orc-(core|common|metadata|presenters|gui|cli))' "${link_scan_paths[@]}")"
if [[ -n "$link_hits" ]]; then
  say "$link_hits"
  say "[sdk-boundary] ERROR: Private host target linkage detected. Link only via plugin SDK targets."
  fail=1
fi

say "[sdk-boundary] Scanning for direct private host library discovery..."
find_hits="$(run_search 'find_(library|package)[[:space:]]*\([^\)]*(orc[-_](core|common|metadata|presenters)|orc::(core|common|metadata|presenters))' "${link_scan_paths[@]}")"
if [[ -n "$find_hits" ]]; then
  say "$find_hits"
  say "[sdk-boundary] ERROR: Private host library discovery detected."
  fail=1
fi

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

say "[sdk-boundary] PASS: No private decode-orc include/link usage found."
