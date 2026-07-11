"""Drizzle validation: vertically aliased fine scene, slow vertical
tilt providing extra sampling phases. Compare drizzle 2x against
linear 2x upscale of the plain decode, vs fine ground truth."""
import sys; sys.path.insert(0, "/home/claude/hvd-decode")
import numpy as np
from hvd.tbc import VideoParameters, TbcSource
from hvd.encode import write_tbc
from hvd.decoder import DecoderConfig, decode_sequence
from bench_interlace import natural_texture

p = VideoParameters(); W, H = p.active_width, 484
HF = 2 * H
FINE = natural_texture(W, HF + 40, seed=21)   # vertically rich scene
TILT = 1.25  # fine-rows per frame: sub-line vertical drift

def scene_rows(t):
    """Scene at time t, sampled on the coarse (H) grid the camera sees:
    row r of the coarse frame reads fine row 2r + drift (NO vertical
    prefilter — exactly how tube cameras and telecines alias)."""
    d = t * TILT
    i = int(np.floor(d)); f = d - i
    fine = (1 - f) * FINE[i:i + HF] + f * FINE[i + 1:i + 1 + HF]
    return fine[0::2], fine  # coarse view, fine truth

N = 6
pairs = []; fines = []
for t in range(N):
    c1, f1 = scene_rows(t)
    c2, f2 = scene_rows(t + 0.5)
    pairs.append((c1, c2)); fines.append((f1, f2))
write_tbc("/tmp/drz.tbc", pairs, params=p, noise_ire=0.8)
src = TbcSource.open("/tmp/drz.tbc")

pic = p.field_height - 21
# fine ground truth woven at 2x vertical: fine row q of output frame t
# is field parity (q//2)%2... build from per-field fine truths
def gt_fine(t):
    g = np.zeros((2 * 2 * pic, W, 3))
    # decoder maps field row m -> coarse frame rows via ysrc; replicate
    ysrc = (np.arange(2 * pic) * H // (2 * pic)).clip(0, H - 1)
    f1, f2 = fines[t]
    # coarse frame row r corresponds to fine rows 2r..2r+1
    for parity, fine in ((0, f1), (1, f2)):
        rows = ysrc[parity::2]              # coarse rows of this field
        for i, r in enumerate(rows):
            q = (2 * i + parity) * 2        # fine output row of that line
            g[q] = fine[2 * r]
            if q + 1 < g.shape[0]:
                g[q + 1] = fine[np.minimum(2 * r + 1, HF - 1)]
    return g[:4 * 484 // 2]

GT = [gt_fine(t) for t in range(N)]
c = 24
def psnr(a, b):
    n = min(a.shape[0], b.shape[0])
    return 10 * np.log10(1.0 / np.mean((a[c:n-c, c:-c] - b[c:n-c, c:-c]) ** 2))

base_cfg = DecoderConfig(temporal_strength=0.5, passes=2, nr_anchor=1.0)
plain = [r.astype(float) / 65535.0 for _, r in decode_sequence(src, 0, N, base_cfg)]
up = [np.repeat(d, 2, axis=0) * 0.5 + np.roll(np.repeat(d, 2, axis=0), -1, 0) * 0.5
      for d in plain]  # linear 2x vertical upscale
drz_cfg = DecoderConfig(temporal_strength=0.5, passes=2, nr_anchor=1.0, drizzle=True)
drz = [r.astype(float) / 65535.0 for _, r in decode_sequence(src, 0, N, drz_cfg)]

pu = np.mean([psnr(u, g) for u, g in zip(up, GT)][2:])
pd = np.mean([psnr(d, g) for d, g in zip(drz, GT)][2:])
print(f"upscale lineaire 2x : {pu:.2f} dB")
print(f"drizzle 2x          : {pd:.2f} dB   (delta {pd-pu:+.2f})")
