#!/usr/bin/env python3
"""
hvd-decode — Holographic-Variational NTSC chroma decoder for ld-decode.

Drop-in-ish alternative to ld-chroma-decoder for NTSC .tbc sources:

    # PNG frames
    python3 hvd_decode.py input.tbc -s 0 -l 10 -o frames/

    # raw RGB48 stream on stdout (pipe into ffmpeg, like ld-chroma-decoder)
    python3 hvd_decode.py input.tbc --pipe | ffmpeg -f rawvideo \
        -pix_fmt rgb48le -s 760x484 -r 30000/1001 -i - out.mkv

Tuning knobs (the whole point of the variational formulation):
    --lambda-c   arbitration prior: higher = smoother chroma (less rainbow),
                 lower = sharper chroma (less dot crawl pushed into luma)
    --cg-iter    refinement iterations      (0 = pure holographic mode)
"""

import argparse
import os
import sys

import numpy as np

from hvd import TbcSource, DecoderConfig
from hvd.decoder import decode_sequence


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input .tbc file")
    ap.add_argument("--input-json", help="path to .tbc.json (default: <input>.json)")
    ap.add_argument("-s", "--start", type=int, default=0, help="first frame")
    ap.add_argument("-l", "--length", type=int, default=-1, help="number of frames (-1 = all)")
    ap.add_argument("-o", "--output", default="out", help="output directory for PNGs")
    ap.add_argument("--pipe", action="store_true",
                    help="write raw rgb48le frames to stdout instead of PNGs")
    ap.add_argument("--lambda-c", type=float, default=1.0,
                    help="arbitration knob: higher = smoother chroma / less rainbowing")
    ap.add_argument("--charbonnier-eps", type=float, default=0.5,
                    help="luma edge-preservation scale in IRE")
    ap.add_argument("--chroma-eps", type=float, default=1.0,
                    help="chroma edge-preservation scale in IRE")
    ap.add_argument("--structure-coupling", type=float, default=0.25,
                    help="parallel-level-sets Y->chroma edge coupling")
    ap.add_argument("--per-field", action="store_true",
                    help="decode fields separately (legacy; more combing)")
    ap.add_argument("--3d", dest="three_d", nargs="?", const=0.5, type=float,
                    default=0.0, metavar="NU",
                    help="enable motion-compensated temporal mode "
                         "(optional strength, default 0.5)")
    ap.add_argument("--temporal-eps", type=float, default=0.0,
                    help="motion gate scale in IRE (0 = auto-calibrated "
                         "to measured source noise)")
    ap.add_argument("--passes", type=int, default=2,
                    help="3D fixed-point refinement passes over the segment")
    ap.add_argument("--no-bidi", action="store_true",
                    help="3D: use only the past neighbor (causal)")
    ap.add_argument("--nr-anchor", type=float, default=1.0,
                    help="strength of the decode->NR->re-encode anchor "
                         "(0 disables; passes>=2 only)")
    ap.add_argument("--nr-eps", type=float, default=0.0,
                    help="NR blend robustness in IRE (0 = auto)")
    ap.add_argument("--nr-radius", type=int, default=2,
                    help="temporal NR radius in fields")
    ap.add_argument("--chunk-frames", type=int, default=6,
                    help="3D streaming window size (frames)")
    ap.add_argument("--chunk-overlap", type=int, default=2,
                    help="temporal context overlap (frames)")
    ap.add_argument("--fast", action="store_true",
                    help="fast mode: same algorithm with cheaper logistics "
                         "(shared motion cache, predicted+verified ME, "
                         "adaptive CG exit). Target: >=3x speed, "
                         "imperceptible difference")
    ap.add_argument("--diag-prior", type=float, default=0.0,
                    help="oriented +/-45deg chroma prior weight (0=off): "
                         "trades axis-aligned chroma sharpness for "
                         "diagonal cross-colour suppression")
    ap.add_argument("--ntsc-j", action="store_true",
                    help="NTSC-J source: black at 0 IRE (no 7.5 IRE setup) "
                         "— required for correct levels on Japanese discs")
    ap.add_argument("--no-acc", action="store_true",
                    help="disable Automatic Color Control (saturation "
                         "calibration from measured burst amplitude)")
    ap.add_argument("--drizzle", action="store_true",
                    help="[RECONSTRUCTION MODE — NOT PURE DECODING] "
                         "vertical 2x stacking of ~10 fields' measured "
                         "samples onto a fine grid (astronomy-style). "
                         "Output frames aggregate multiple fields.")
    ap.add_argument("--soft-output", action="store_true",
                    help="[FILTERING MODE — NOT PURE DECODING] allow the "
                         "anchored (temporally denoised) luma in the "
                         "output. The DEFAULT is pure decoding: output "
                         "satisfies Y + Re[chi c] = S exactly per pixel "
                         "(lossless split; neighbors only arbitrate the "
                         "Y/C ownership, as any 3D comb does).")
    ap.add_argument("--cg-iter", type=int, default=60)
    ap.add_argument("--chroma-gain", type=float, default=1.0)
    ap.add_argument("--monochrome", action="store_true")
    args = ap.parse_args()

    if args.drizzle or args.soft_output:
        mode = "drizzle (multi-field reconstruction)" if args.drizzle \
               else "soft-output (temporal filtering)"
        print(f"NOTE: {mode} enabled — output is beyond pure decoding. "
              f"Omit the flag for the lossless-split default.",
              file=sys.stderr)

    src = TbcSource.open(args.input, args.input_json)

    # --- decoding / reconstruction boundary notice -------------------
    # Default output is a PURE DECODE: Y + Re[chi e^{i phi}] = S holds
    # exactly (lossless split of the disc's composite). The two flags
    # below leave that regime and must never do so silently.
    if args.drizzle:
        print("NOTE: --drizzle output is a RECONSTRUCTION (multi-field "
              "stacking on a synthesized grid), not a pure decode.",
              file=sys.stderr)
    elif args.soft_output:
        print("NOTE: --soft-output luma is partially denoised; not a "
              "pure decode. Default mode is the lossless split.",
              file=sys.stderr)
    cfg = DecoderConfig(lambda_c=args.lambda_c,
                        charbonnier_eps=args.charbonnier_eps,
                        chroma_eps=args.chroma_eps,
                        structure_coupling=args.structure_coupling,
                        frame_decode=not args.per_field,
                        temporal_strength=args.three_d,
                        temporal_eps=args.temporal_eps,
                        passes=args.passes,
                        bidirectional=not args.no_bidi,
                        nr_anchor=args.nr_anchor,
                        nr_eps=args.nr_eps,
                        nr_radius=args.nr_radius,
                        chunk_frames=args.chunk_frames,
                        chunk_overlap=args.chunk_overlap,
                        output_fidelity=not args.soft_output,
                        drizzle=args.drizzle,
                        ntsc_j=args.ntsc_j,
                        diag_prior=args.diag_prior,
                        fast=args.fast,
                        acc=not args.no_acc,
                        cg_iterations=args.cg_iter,
                        chroma_gain=args.chroma_gain,
                        monochrome=args.monochrome)

    n = src.num_frames - args.start if args.length < 0 else args.length
    n = max(0, min(n, src.num_frames - args.start))

    if not args.pipe:
        os.makedirs(args.output, exist_ok=True)
        try:
            from PIL import Image
        except ImportError:
            Image = None

    for i, rgb in decode_sequence(src, args.start, n, cfg):
        if args.pipe:
            sys.stdout.buffer.write(rgb.astype("<u2").tobytes())
        else:
            path = os.path.join(args.output, f"frame_{i:06d}.png")
            if Image is not None:
                Image.fromarray((rgb >> 8).astype(np.uint8), "RGB").save(path)
            else:  # PPM fallback, 16-bit
                with open(path.replace(".png", ".ppm"), "wb") as f:
                    h, w, _ = rgb.shape
                    f.write(f"P6 {w} {h} 65535\n".encode())
                    f.write(rgb.astype(">u2").tobytes())
            print(f"frame {i} -> {path}  ({rgb.shape[1]}x{rgb.shape[0]})",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
