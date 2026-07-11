import sys, os; sys.path.insert(0, "/home/claude/hvd-decode")
import numpy as np
from hvd.tbc import VideoParameters, TbcSource
from hvd.encode import write_tbc
from hvd.decoder import DecoderConfig, decode_sequence
from bench_interlace import natural_texture

p = VideoParameters(); W, H = p.active_width, 484
BASE = natural_texture(W + 40, H, seed=11)
PAN = 1.5  # px/frame -> 0.75 px/field: pure sub-pixel regime

def scene(t):
    sh = t * PAN
    i = int(np.floor(sh)); f = sh - i
    img = (1-f)*BASE[:, i:i+W] + f*BASE[:, i+1:i+1+W]
    return img

N = 4
pairs = [(scene(t), scene(t+0.5)) for t in range(N)]
write_tbc("/tmp/subpel.tbc", pairs, params=p, noise_ire=0.8)
src = TbcSource.open("/tmp/subpel.tbc")
pic = p.field_height-21
ysrc = (np.arange(2*pic)*H//(2*pic)).clip(0, H-1)
GT = []
for t in range(N):
    g = np.zeros((2*pic, W, 3))
    g[0::2] = pairs[t][0][ysrc][0::2]
    g[1::2] = pairs[t][1][ysrc][1::2]
    GT.append(g[:484])
c = 16
def psnr(a, b): return 10*np.log10(1.0/np.mean((a[c:-c,c:-c]-b[c:-c,c:-c])**2))
cfg = DecoderConfig(temporal_strength=0.5, passes=2, nr_anchor=1.0)
dec = [r.astype(float)/65535.0 for _, r in decode_sequence(src, 0, N, cfg)]
v = np.mean([psnr(d, g) for d, g in zip(dec, GT)][1:])
mode = "entier" if os.environ.get("HVD_INT_BLEND") else "sous-pixel+parite"
print(f"pan 1.5px/frame, blend {mode:20s}: {v:.2f} dB")
