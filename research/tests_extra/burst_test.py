import sys, os; sys.path.insert(0, "/home/claude/hvd-decode")
import numpy as np, tempfile
from hvd.encode import make_test_pattern, write_tbc
from hvd.tbc import VideoParameters, TbcSource
from hvd.decoder import DecoderConfig, decode_frame

p = VideoParameters()
gt = make_test_pattern(p.active_width, 484, "smpte")
pic = p.field_height - 21
ys = (np.arange(2*pic)*gt.shape[0]//(2*pic)).clip(0, gt.shape[0]-1)
gta = gt[ys][:484]; c = 12
def psnr(r):
    r = r.astype(float)/65535.0
    return 10*np.log10(1.0/np.mean((r[c:-c,c:-c]-gta[c:-c,c:-c])**2))
for bd in (0.0, 0.10, 0.25):
    td = tempfile.mkdtemp(); f = td+"/t.tbc"
    write_tbc(f, [gt], params=p, noise_ire=0.8, burst_dropout=bd)
    src = TbcSource.open(f)
    v = psnr(decode_frame(src, 0, DecoderConfig()))
    mode = "SANS lissage" if os.environ.get("HVD_NO_PHASE_SMOOTH") else "avec lissage"
    print(f"bursts corrompus {int(bd*100):3d}%  {mode}: {v:.2f} dB")
