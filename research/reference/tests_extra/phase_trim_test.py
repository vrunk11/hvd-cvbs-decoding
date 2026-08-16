"""Phase reference and level domain regression test.

Run with:  PYTHONPATH=. python3 tests_extra/phase_trim_test.py

Locks down three things that no existing test could see, because the
encoder and the decoder shared the same wrong conventions and cancelled
each other out:

  1. With chroma_phase_deg = 0 the decoder recovers the TRUE chroma from a
     burst synthesized the way a real encoder makes one (on the -U axis,
     i.e. -B sin(phi)). The old lock-in returned angle(z) + pi/2 where the
     correct relation is angle(z) - pi/2, so every decode was 180 deg out
     and needed a hardcoded 180 deg "correction" to look right.

  2. chroma_phase_deg is RELATIVE, not absolute. The rotation it produces
     must be identical whatever the source's own subcarrier phase happens
     to be, and it must not touch saturation. An absolute parameter (one
     that replaced the measured reference rather than adding to it) would
     give a different answer for every source phase -- this test would
     catch that immediately.

  3. On real ld-decode NTSC-M levels a nominal 20 IRE burst measures 20.0
     IRE and the ACC gain comes out at exactly 1.0. Under the old
     black-referenced scale it measured 21.62 and the ACC quietly applied
     a permanent 0.925, i.e. 7.5 % desaturation on every NTSC decode.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hvd.tbc import VideoParameters
from hvd.decoder import burst_lockin_phase, burst_amplitude_ire, phase_map

p = VideoParameters()
W, H = p.field_width, 24
x = np.arange(W)

U_TRUE, V_TRUE = -18.0, 25.0


def make_field(theta0, U, V, burst=20.0):
    """One field of TRUE-IRE composite.

    S = Y + U sin(phi) + V cos(phi), burst on the -U axis => -B sin(phi).
    """
    S = np.zeros((H, W))
    th = theta0 + np.pi * np.arange(H)  # 180 deg/line, the NTSC model
    phi = th[:, None] + (np.pi / 2.0) * x[None, :]
    b0, b1 = p.colour_burst_start, p.colour_burst_end
    S[:, b0:b1] = -burst * np.sin(phi[:, b0:b1])
    a0, a1 = p.active_video_start, p.active_video_end
    S[:, a0:a1] = 50.0 + U * np.sin(phi[:, a0:a1]) + V * np.cos(phi[:, a0:a1])
    return S


def recover_chi(S, theta, row=8):
    """chi = V - iU, demodulated against the decoder's own phase map."""
    a0, a1 = p.active_video_start, p.active_video_end
    phi = phase_map(theta, W)[row, a0:a1]
    seg = S[row, a0:a1]
    return 2.0 * np.mean((seg - seg.mean()) * np.exp(-1j * phi))


# --- 1. absolute correctness at trim = 0 ------------------------------
for theta0 in (0.3, 2.1, -1.4):
    S = make_field(theta0, U_TRUE, V_TRUE)
    chi = recover_chi(S, burst_lockin_phase(S, p, 0.0))
    assert abs(-chi.imag - U_TRUE) < 0.5, (theta0, -chi.imag)
    assert abs(chi.real - V_TRUE) < 0.5, (theta0, chi.real)
print("1. trim=0 recovers true chroma at any source phase           OK")

# --- 2. the trim is relative, and gain-neutral ------------------------
for trim in (0.0, 30.0, 90.0, -45.0, 180.0):
    rots, mags = [], []
    for theta0 in (0.3, 2.1, -1.4, 5.0):
        S = make_field(theta0, U_TRUE, V_TRUE)
        ref = recover_chi(S, burst_lockin_phase(S, p, 0.0))
        got = recover_chi(S, burst_lockin_phase(S, p, trim))
        rots.append(np.degrees(np.angle(got / ref)))
        mags.append(abs(got) / abs(ref))
    spread = max(rots) - min(rots)
    want = np.degrees(np.angle(np.exp(1j * np.deg2rad(trim))))
    assert spread < 0.05, f"NOT RELATIVE: rotation varies by {spread} deg"
    assert abs(np.degrees(np.angle(np.exp(1j * (np.deg2rad(np.mean(rots) - want)))))) < 0.05
    assert abs(np.mean(mags) - 1.0) < 1e-6, "trim must not touch saturation"
print("2. trim is relative (phase-independent) and gain-neutral      OK")

# --- 3. level domain / ACC -------------------------------------------
S = make_field(0.5, 0.0, 0.0, burst=20.0)
meas = burst_amplitude_ire(p.ire(p.raw(S)), p)
gain = float(np.clip(20.0 / max(meas, 1.0), 0.5, 2.0))
assert abs(p.codes_per_ire - 358.4) < 1e-6, p.codes_per_ire
assert abs(float(p.ire(np.array([float(p.black16bIre)]))[0]) - 7.5) < 1e-6
assert abs(float(p.ire(np.array([float(p.blanking16bIre)]))[0])) < 1e-9
assert abs(meas - 20.0) < 0.02, meas
assert abs(gain - 1.0) < 0.002, gain
print(f"3. nominal burst measures {meas:.4f} IRE -> ACC gain {gain:.4f}   OK")

print("phase_trim_test: ALL CHECKS PASSED")
