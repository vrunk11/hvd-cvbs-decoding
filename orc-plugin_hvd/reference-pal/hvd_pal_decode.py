#!/usr/bin/env python3
"""
hvd_pal_decode — Holographic-Variational PAL chroma decoder (research).

    # PNG frames
    python3 hvd_pal_decode.py input.tbc -s 0 -l 10 -o frames/

    # raw RGB48 stream on stdout (pipe into ffmpeg, ld-chroma-decoder style)
    python3 hvd_pal_decode.py input.tbc --pipe | ffmpeg -f rawvideo \
        -pix_fmt rgb48le -s 922x582 -r 25 -i - out.mkv

Knobs:
    --lambda-c   chroma smoothness vs luma plausibility arbitration
    --cg-iter    refinement iterations (0 = pure holographic mode)
    --baseline   decode with the delay-line reference instead (A/B tool)
"""

import argparse
import os
import sys

import numpy as np

from hvd_pal import TbcSource, DecoderConfig, decode_frame
from hvd_pal import decoder as dec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input PAL .tbc file")
    ap.add_argument("--input-json", help="path to .tbc.json (default: <input>.json)")
    ap.add_argument("-s", "--start", type=int, default=0)
    ap.add_argument("-l", "--length", type=int, default=-1)
    ap.add_argument("-o", "--output", default="out")
    ap.add_argument("--pipe", action="store_true",
                    help="write raw rgb48le frames to stdout")
    ap.add_argument("--lambda-c", type=float, default=1.0)
    ap.add_argument("--chroma-aniso", type=float, default=0.7)
    ap.add_argument("--chroma-eps", type=float, default=1.0)
    ap.add_argument("--cg-iter", type=int, default=60)
    ap.add_argument("--chroma-gain", type=float, default=1.0)
    ap.add_argument("--no-acc", action="store_true")
    ap.add_argument("--monochrome", action="store_true")
    ap.add_argument("--baseline", action="store_true",
                    help="decode with the delay-line reference decoder")
    ap.add_argument("--3d", dest="three_d", action="store_true",
                    help="temporal equations from same-parity fields "
                         "f+/-2 and f+/-4 (nu=0.5, swept default)")
    ap.add_argument("--temporal-strength", type=float, default=None,
                    help="override nu (implies 3D if > 0)")
    ap.add_argument("--passes", type=int, default=1,
                    help=">= 2 enables the decode->NR->re-encode anchor "
                         "loop (implies 3D benefit; heavy)")
    ap.add_argument("--soft-output", action="store_true",
                    help="deliver the softly-fidelity (Y, chi) of the "
                         "anchor pass instead of re-imposing the "
                         "lossless split (output NR)")
    ap.add_argument("--drizzle", action="store_true",
                    help="vertical 2x drizzle output (reconstruction, "
                         "not decoding: outside the purity contract)")
    ap.add_argument("--polsar-map", metavar="PNG",
                    help="write the cross-pol |chi_x| differential-phase "
                         "diagnostic map of the first frame and exit")
    args = ap.parse_args()

    src = TbcSource.open(args.input, args.input_json)
    nu = args.temporal_strength if args.temporal_strength is not None \
        else (0.5 if args.three_d else 0.0)
    cfg = DecoderConfig(lambda_c=args.lambda_c, chroma_aniso=args.chroma_aniso,
                        chroma_eps=args.chroma_eps, cg_iterations=args.cg_iter,
                        chroma_gain=args.chroma_gain, acc=not args.no_acc,
                        monochrome=args.monochrome, temporal_strength=nu,
                        passes=args.passes,
                        output_fidelity=not args.soft_output,
                        drizzle=args.drizzle)
    if args.drizzle and nu == 0.0:
        cfg.temporal_strength = nu = 0.5
    if args.passes >= 2 and nu == 0.0:
        cfg.temporal_strength = nu = 0.5  # anchor loop implies 3D

    if args.polsar_map:
        S, carrier, _ = dec.prepare_frame(src, args.start, cfg)
        co, cx = dec.polarimetric_maps(S, carrier, src.params, cfg)
        from PIL import Image
        m = np.clip(cx / 10.0, 0, 1)  # 10 IRE full scale
        Image.fromarray((m * 255).astype(np.uint8)).save(args.polsar_map)
        print(f"cross-pol map -> {args.polsar_map} "
              f"(median {np.median(cx[co > 15.0]) if (co > 15.0).any() else 0.0:.2f} IRE "
              f"on saturated chroma; ~0 = clean chain, "
              f"~|chi|*sin(delta) = differential phase)", file=sys.stderr)
        return

    n = src.num_frames - args.start if args.length < 0 else args.length
    if not args.pipe:
        os.makedirs(args.output, exist_ok=True)

    if (nu > 0.0 or args.passes >= 2) and not args.baseline:
        for k, (Y, U, V) in enumerate(
                dec.decode_sequence(src, args.start, n, cfg)):
            i = args.start + k
            rgb = dec.yuv_to_rgb16(Y, U, V)
            if args.pipe:
                sys.stdout.buffer.write(rgb.astype("<u2").tobytes())
            else:
                try:
                    from PIL import Image
                    Image.fromarray((rgb >> 8).astype(np.uint8)).save(
                        os.path.join(args.output, f"frame{i:06d}.png"))
                except ImportError:
                    rgb.astype("<u2").tofile(
                        os.path.join(args.output, f"frame{i:06d}.rgb48le"))
                print(f"frame {i} done", file=sys.stderr)
        return

    for i in range(args.start, args.start + n):
        if args.baseline:
            S, carrier, gain = dec.prepare_frame(src, i, cfg)
            Y, chi = dec.delayline_baseline(S, carrier, src.params, cfg)
            g = cfg.chroma_gain * gain
            U, V = -np.imag(chi) * g, np.real(chi) * g
        else:
            Y, U, V = decode_frame(src, i, cfg)
        rgb = dec.yuv_to_rgb16(Y, U, V)
        if args.pipe:
            sys.stdout.buffer.write(rgb.astype("<u2").tobytes())
        else:
            try:
                from PIL import Image
                Image.fromarray((rgb >> 8).astype(np.uint8)).save(
                    os.path.join(args.output, f"frame{i:06d}.png"))
            except ImportError:
                rgb.astype("<u2").tofile(
                    os.path.join(args.output, f"frame{i:06d}.rgb48le"))
            print(f"frame {i} done", file=sys.stderr)


if __name__ == "__main__":
    main()
