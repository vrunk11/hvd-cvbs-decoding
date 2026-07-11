#!/usr/bin/env python3
"""Benchmark the 3D (temporal) mode on realistic moving content.

Scene: 1/f 'natural' colour texture with (a) a static region, (b) a
globally panning region, (c) an independently moving saturated object,
(d) fine luma detail (cross-colour bait). Ground truth is exact
(integer-pixel motion).

Decoders compared, per-region PSNR (static vs moving):
  - hvd 2D  (spatial only)
  - hvd 3D  (robust temporal priors, no motion detector)
  - naive 3D frame comb (inter-frame average/difference) — the
    conventional approach's failure mode on motion.
"""
import numpy as np
from PIL import Image

from hvd.tbc import VideoParameters, TbcSource
from hvd.encode import write_tbc
from hvd.decoder import (DecoderConfig, decode_sequence, decode_frame,
                         burst_lockin_phase, yuv_to_rgb16)

rng = np.random.default_rng(7)
p = VideoParameters()
W, H = p.active_width, 484
N_FRAMES = 5
PAN = 3          # px/frame horizontal pan in the moving band
OBJ_V = (6, 2)   # object velocity px/frame (x, y)


def natural_texture(w, h, beta=1.5, seed=0):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(3):
        z = r.normal(size=(h, w))
        F = np.fft.fft2(z)
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        f = np.sqrt(fx**2 + fy**2); f[0, 0] = 1
        img = np.real(np.fft.ifft2(F / f**beta))
        img = (img - img.min()) / (np.ptp(img) + 1e-9)
        out.append(img)
    tex = np.stack(out, -1)
    # correlate channels a bit (natural images have correlated RGB)
    m = tex.mean(-1, keepdims=True)
    return np.clip(0.35 * tex + 0.65 * m + 0.15 * (tex - 0.5), 0.05, 0.95)


BASE = natural_texture(W * 2, H, seed=3)          # wide for panning
STATIC_ROWS = slice(0, H // 3)                     # top third: static
PAN_ROWS = slice(H // 3, 2 * H // 3)               # middle third: pans
OBJ_ROWS = slice(2 * H // 3, H)                    # bottom: moving object

def make_frame(t):
    img = BASE[:, :W].copy()
    # panning band
    img[PAN_ROWS] = BASE[PAN_ROWS, t * PAN:t * PAN + W]
    # fine luma detail strip inside static zone (cross-colour bait)
    x = np.arange(W)
    detail = 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * x / 4.0))
    img[20:44] = detail[None, :, None]
    # moving saturated object over textured bottom
    ox = 60 + t * OBJ_V[0]
    oy = 2 * H // 3 + 30 + t * OBJ_V[1]
    img[oy:oy + 50, ox:ox + 90] = [0.85, 0.15, 0.2]
    img[oy + 12:oy + 38, ox + 20:ox + 70] = [0.1, 0.5, 0.85]
    return img


frames = [make_frame(t) for t in range(N_FRAMES)]
write_tbc("motion.tbc", frames, params=p, noise_ire=0.8)
src = TbcSource.open("motion.tbc")

# ground truth aligned to decoder row sampling
pic = p.field_height - 21
ys = (np.arange(2 * pic) * H // (2 * pic)).clip(0, H - 1)
GT = [f[ys][:484] for f in frames]

# region masks (crop 12 px borders everywhere)
c = 12
mask_static = np.zeros((484, W), bool); mask_static[STATIC_ROWS] = True
mask_moving = np.zeros((484, W), bool)
mask_moving[PAN_ROWS] = True; mask_moving[OBJ_ROWS] = True
for m in (mask_static, mask_moving):
    m[:c] = m[-c:] = False; m[:, :c] = m[:, -c:] = False


def psnr(a, b, m):
    return 10 * np.log10(1.0 / np.mean((a[m] - b[m]) ** 2))


def report(name, decoded):
    st = np.mean([psnr(d, g, mask_static) for d, g in zip(decoded, GT)][1:])
    mv = np.mean([psnr(d, g, mask_moving) for d, g in zip(decoded, GT)][1:])
    print(f"{name:34s} static {st:6.2f} dB   moving {mv:6.2f} dB")
    return decoded


def run(cfg):
    return [rgb.astype(np.float64) / 65535.0
            for _, rgb in decode_sequence(src, 0, N_FRAMES, cfg)]


# ---- naive 3D frame comb baseline -------------------------------------
def naive_3d_comb():
    a0, a1 = p.active_video_start, p.active_video_end
    fal = 21
    woven = []
    for i in range(N_FRAMES):
        (f0, _), (f1, _) = src.read_frame_fields(i)
        Ss, ths = [], []
        for fld in (f0, f1):
            ire = p.ire(fld)
            ths.append(burst_lockin_phase(ire, p)[fal:])
            Ss.append(ire[fal:, a0:a1])
        L = Ss[0].shape[0]
        S = np.zeros((2 * L, p.active_width)); th = np.zeros(2 * L)
        S[0::2], S[1::2] = Ss; th[0::2], th[1::2] = ths
        x = np.arange(p.field_width)[a0:a1]
        woven.append((S, th[:, None] + (np.pi / 2) * x[None, :]))
    out = []
    k = np.hanning(21); k /= k.sum()
    for i in range(N_FRAMES):
        S, phi = woven[i]
        Sn, _ = woven[min(i + 1, N_FRAMES - 1)] if i + 1 < N_FRAMES else woven[i - 1]
        # carrier flips 180 deg frame-to-frame:
        # static => luma = (S+Sn)/2, modulated chroma = (S-Sn)/2
        C = 0.5 * (S - Sn)
        Y = 0.5 * (S + Sn)
        z = C * np.exp(-1j * phi)
        zr = np.apply_along_axis(np.convolve, 1, z.real, k, 'same')
        zi = np.apply_along_axis(np.convolve, 1, z.imag, k, 'same')
        chi = 2 * (zr + 1j * zi)
        out.append(yuv_to_rgb16(Y, -np.imag(chi), np.real(chi))
                   .astype(np.float64) / 65535.0)
    return out


print(f"scene: static third / panning third ({PAN}px/f) / moving object "
      f"({OBJ_V}px/f), 1/f texture, 0.8 IRE noise, {N_FRAMES} frames")
print("(frame 0 excluded from averages: causal 3D warms up)\n")
report("naive 3D frame comb", naive_3d_comb())
report("hvd 2D (spatial only)", run(DecoderConfig()))
report("hvd 3D causal 1-pass",
       run(DecoderConfig(temporal_strength=0.5, bidirectional=False)))
report("hvd 3D bidi 1-pass",
       run(DecoderConfig(temporal_strength=0.5)))
report("hvd 3D bidi 2-pass",
       run(DecoderConfig(temporal_strength=0.5, passes=2)))
report("hvd 3D bidi 3-pass",
       run(DecoderConfig(temporal_strength=0.5, passes=3)))
report("hvd 3D bidi 2-pass nu=1.0",
       run(DecoderConfig(temporal_strength=1.0, passes=2)))
