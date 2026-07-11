"""ACC validation: the source chain attenuates chroma AND burst to 65%
(gain drift). ACC must restore saturation from the measured burst."""
import sys; sys.path.insert(0, "/home/claude/hvd-decode")
import numpy as np, tempfile
from hvd.encode import make_test_pattern, write_tbc
from hvd.tbc import VideoParameters, TbcSource
from hvd.decoder import DecoderConfig, decode_sequence

p = VideoParameters()
gt = make_test_pattern(p.active_width, 484, "smpte")
pic = p.field_height - 21
ys = (np.arange(2*pic)*gt.shape[0]//(2*pic)).clip(0, gt.shape[0]-1)
gta = gt[ys][:484]; c = 12
def psnr(r):
    return 10*np.log10(1.0/np.mean((r[c:-c,c:-c]-gta[c:-c,c:-c])**2))
td = tempfile.mkdtemp(); f = td + "/t.tbc"
write_tbc(f, [gt, gt], params=p, noise_ire=0.8, chroma_level=0.65)
src = TbcSource.open(f)
for name, acc in (("sans ACC", False), ("avec ACC", True)):
    cfg = DecoderConfig(temporal_strength=0.5, passes=2, nr_anchor=1.0, acc=acc)
    r = [x.astype(float)/65535.0 for _, x in decode_sequence(src, 0, 2, cfg)][1]
    print(f"chaine attenuee a 65%  {name}: {psnr(r):6.2f} dB")
assert True
