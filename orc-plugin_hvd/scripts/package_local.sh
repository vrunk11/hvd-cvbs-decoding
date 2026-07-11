#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <build-dir> [output-dir]"
    exit 1
fi

build_dir="$1"
output_dir="${2:-dist}"

mkdir -p "$output_dir"

case "$(uname -s)" in
    Linux)
        platform="linux"
        ext="so"
        ;;
    Darwin)
        platform="macos"
        ext="dylib"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        platform="windows"
        ext="dll"
        ;;
    *)
        echo "Unsupported platform"
        exit 1
        ;;
esac

plugin_name="orc-plugin_hvd_chroma_decoder_${platform}.${ext}"

candidate="$(find "$build_dir" -type f \( -name "*orc-stage-plugin-hvd-chroma-decoder*.${ext}" -o -name "orc-stage-plugin-hvd-chroma-decoder.${ext}" \) | head -n 1)"

if [[ -z "$candidate" ]]; then
    echo "Could not locate built plugin binary in $build_dir"
    exit 1
fi

cp "$candidate" "$output_dir/$plugin_name"

echo "Packaged: $output_dir/$plugin_name"
