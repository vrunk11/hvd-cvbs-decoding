#!/usr/bin/env python3
"""Generate one platform's fragment of orc-plugin-manifest.yaml.

Run once per matrix job (linux/macos/windows), right after packaging that
platform's binary. The fragment carries everything known at that point —
file name, platform, host ABI, toolchain tag (all three read straight off the
just-built binary's own descriptor via hvd_print_build_info, never
re-derived) and the sha256 of the packaged asset — and travels inside the
same `dist/` directory that's already uploaded as a build artifact.

merge_manifest_fragments.py (run once, later, in the release job after all
three matrix jobs have finished) combines the three fragments into the final
orc-plugin-manifest.yaml. Splitting generation this way is necessary because
the three platform binaries are built in separate, parallel CI jobs that
can't see each other's output directly.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def parse_build_info(text: str) -> dict:
    info = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        info[key.strip()] = value.strip()
    return info


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", required=True, type=Path,
                     help="Path to the packaged plugin binary (e.g. dist/orc-plugin_hvd_chroma_decoder_linux.so)")
    ap.add_argument("--platform", required=True, choices=["linux", "macos", "windows"])
    ap.add_argument("--build-info-exe", type=Path, default=None,
                     help="Path to the hvd_print_build_info executable. "
                          "Mutually exclusive with --build-info-text.")
    ap.add_argument("--build-info-text", type=Path, default=None,
                     help="Path to a file already containing hvd_print_build_info's "
                          "stdout (useful on Windows, where invoking a freshly built "
                          ".exe from Python across shells is one more thing to get "
                          "wrong than just redirecting it once in the calling step).")
    ap.add_argument("--out", required=True, type=Path,
                     help="Where to write this platform's manifest fragment (JSON)")
    args = ap.parse_args()

    if bool(args.build_info_exe) == bool(args.build_info_text):
        ap.error("pass exactly one of --build-info-exe or --build-info-text")

    if args.build_info_exe:
        proc = subprocess.run([str(args.build_info_exe)], capture_output=True,
                              text=True, check=True)
        info = parse_build_info(proc.stdout)
    else:
        info = parse_build_info(args.build_info_text.read_text())

    for required in ("plugin_id", "host_abi", "toolchain_tag"):
        if required not in info:
            print(f"error: hvd_print_build_info output is missing '{required}': {info}",
                  file=sys.stderr)
            return 1

    if not args.artifact.is_file():
        print(f"error: artifact not found: {args.artifact}", file=sys.stderr)
        return 1

    fragment = {
        "file": args.artifact.name,
        "platform": args.platform,
        "abi": int(info["host_abi"]),
        "toolchain_tag": info["toolchain_tag"],
        "sha256": sha256_of(args.artifact),
        "plugin_id": info["plugin_id"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fragment, indent=2) + "\n")
    print(f"wrote {args.out}: {fragment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
