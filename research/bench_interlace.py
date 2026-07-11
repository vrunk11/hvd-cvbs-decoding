#!/usr/bin/env python3
"""True-interlace motion benchmark.

Fields are sampled at their real times (t, t+0.5): what a camera does.
Ground truth for evaluation = per-field (each decoded field compared to
the scene at its own instant), so decoder-induced inter-field
contamination shows up while legitimate interlace does not.

Metrics per zone: PSNR + inter-field chroma combing (line-alternating
chroma energy, which for a perfect decoder equals the scene's own
motion, measured on ground truth as reference).
"""
import sys, time
sys.path.insert(0, "/home/claude/hvd-decode")
import numpy as np
from hvd.tbc import VideoParameters, TbcSource
from hvd.encode import write_tbc
from hvd.decoder import DecoderConfig, decode_sequence

p = VideoParameters(); W, H = p.active_width, 484
N = 5; PAN = 3; OBJ_V = (6, 2)

def natural_texture(w, h, beta=1.5, seed=0):
    r = np.random.default_rng(seed); out = []
    for _ in range(3):
        z = r.normal(size=(h, w)); F = np.fft.fft2(z)
        fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None, :]
        f = np.sqrt(fx ** 2 + fy ** 2); f[0, 0] = 1
        img = np.real(np.fft.ifft2(F / f ** beta))
        img = (img - img.min()) / (np.ptp(img) + 1e-9); out.append(img)
    tex = np.stack(out, -1); m = tex.mean(-1, keepdims=True)
    return np.clip(0.35 * tex + 0.65 * m + 0.15 * (tex - 0.5), 0.05, 0.95)

BASE = natural_texture(W * 2, H, seed=3)
STATIC_ROWS = slice(0, H // 3)
PAN_ROWS = slice(H // 3, 2 * H // 3)
OBJ_ROWS = slice(2 * H // 3, H)

def scene(t):
    """Scene at continuous time t (frames)."""
    img = BASE[:, :W].copy()
    sh = int(round(t * PAN))
    img[PAN_ROWS] = BASE[PAN_ROWS, sh:sh + W]
    x = np.arange(W); detail = 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * x / 4.0))
    img[20:44] = detail[None, :, None]
    ox = 60 + int(round(t * OBJ_V[0])); oy = 2 * H // 3 + 30 + int(round(t * OBJ_V[1]))
    img[oy:oy + 50, ox:ox + 90] = [0.85, 0.15, 0.2]
    img[oy + 12:oy + 38, ox + 20:ox + 70] = [0.1, 0.5, 0.85]
    return img

def make_pairs():
    return [(scene(t), scene(t + 0.5)) for t in range(N)]

if __name__ == "__main__":
    pairs = make_pairs()
    write_tbc("motion_il.tbc", pairs, params=p, noise_ire=0.8)
    src = TbcSource.open("motion_il.tbc")

    # per-field ground truth woven back (row parity matches decoder output)
    pic = p.field_height - 21
    ysrc = (np.arange(2 * pic) * H // (2 * pic)).clip(0, H - 1)
    GT = []
    for t in range(N):
        g = np.zeros((2 * pic, W, 3))
        g[0::2] = pairs[t][0][ysrc][0::2]
        g[1::2] = pairs[t][1][ysrc][1::2]
        GT.append(g[:484])

    c = 12
    ms = np.zeros((484, W), bool); ms[STATIC_ROWS] = True
    mm = np.zeros((484, W), bool); mm[PAN_ROWS] = True; mm[OBJ_ROWS] = True
    for m in (ms, mm):
        m[:c] = m[-c:] = False; m[:, :c] = m[:, -c:] = False

    def psnr(a, b, m): return 10 * np.log10(1.0 / np.mean((a[m] - b[m]) ** 2))

    def combing(img, m):
        alt = img[1:-1] - 0.5 * (img[:-2] + img[2:])
        return np.sqrt(np.mean(alt[m[1:-1]] ** 2))

    gt_comb = np.mean([combing(g, mm) for g in GT])
    print(f"combing intrinseque du contenu entrelace (GT): {gt_comb:.4f}")

    def report(name, cfg):
        t0 = time.time()
        dec = [r.astype(np.float64) / 65535.0
               for _, r in decode_sequence(src, 0, N, cfg)]
        st = np.mean([psnr(d, g, ms) for d, g in zip(dec, GT)][1:])
        mv = np.mean([psnr(d, g, mm) for d, g in zip(dec, GT)][1:])
        cb = np.mean([combing(d, mm) for d in dec][1:])
        print(f"{name:30s} static {st:6.2f}  moving {mv:6.2f}  "
              f"combing {cb:.4f} (exces {cb - gt_comb:+.4f})  "
              f"({time.time() - t0:.0f}s)")

    report("2D (tisse)", DecoderConfig())
    report("3D defauts (tisse)", DecoderConfig(temporal_strength=0.5,
                                               passes=2, nr_anchor=1.0))

