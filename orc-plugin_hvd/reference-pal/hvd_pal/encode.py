"""
hvd_pal.encode — Synthetic PAL composite encoder -> .tbc + .tbc.json

Ground-truth generator implementing the exact model the decoder
inverts:

    S(x, l) = Y + U*sin(phi) + s(l)*V*cos(phi)
            = Y + Re[ chi * c(x, l) ]

    chi     = V - i*U                     (single global chroma phasor)
    phi     = phi0(field) + (3*pi/2)*l + (pi/2)*x
    s(l)    = +/-1                        (PAL V-switch, alternates per line)
    c(x, l) = exp(i*phi)       on s = +1 lines
            = -exp(-i*phi)     on s = -1 lines   (the V-switch folded
                                                  into an EFFECTIVE
                                                  CARRIER — see decoder)

Geometry on the stored 1135-sample grid: 90 deg/sample horizontally,
270 deg/line vertically (283.75 cycles/line), 135 deg/field origin
advance => the 8-field PAL sequence (8 * 135 = 1080 = 0 mod 360).

Swinging burst: burst chroma is (U_b, V_b) = (-B/sqrt2, +B/sqrt2), so
through the same V-switch path the burst phasor swings +/-45 deg about
the -U axis line to line — the decoder's joint phase+parity estimator
relies on exactly this.

`diff_phase_deg` applies a DIFFERENTIAL PHASE error (chroma subcarrier
rotated relative to burst) — the classic transmission impairment PAL
was invented to survive. Used by the Hanover/hue torture test.
"""

from __future__ import annotations

import json
import numpy as np

from .tbc import VideoParameters, FSC_PAL, FS_4FSC_PAL

RGB_TO_YUV = np.array(
    [[0.299, 0.587, 0.114],
     [-0.14713, -0.28886, 0.436],
     [0.615, -0.51499, -0.10001]]
)

IRE_SYNC = -43.0        # PAL sync tip (-300 mV)
IRE_BLANK = 0.0
IRE_BLACK = 0.0         # PAL has NO setup pedestal (unlike NTSC-M's 7.5)
IRE_WHITE = 100.0
BURST_IRE = 21.43       # peak burst amplitude (+/-150 mV on 700 mV white)

LINE_ADV = 3.0 * np.pi / 2.0     # 270 deg/line on the stored grid
FIELD_ADV = 3.0 * np.pi / 4.0    # 135 deg/field origin advance (8-field seq)


def _phase_origin(field_phase_id: int) -> float:
    """Subcarrier phase at sample 0 / line 0 for field n of the
    8-field sequence."""
    return (field_phase_id - 1) * FIELD_ADV


def _vswitch_offset(field_phase_id: int) -> int:
    """V-switch line-parity offset for a field. Real PAL alternates
    because a field is 312.5 lines; a simple per-field alternation
    gives the correct woven-frame alternation (the decoder MEASURES
    parity from the swinging burst and never assumes this)."""
    return (field_phase_id - 1) % 2


def make_test_pattern(width: int, height: int, kind: str = "ebu") -> np.ndarray:
    """RGB float image in [0,1], shape (height, width, 3)."""
    img = np.zeros((height, width, 3), dtype=np.float64)
    if kind == "ebu":
        cols = np.array([
            [0.75, 0.75, 0.75],
            [0.75, 0.75, 0.00],
            [0.00, 0.75, 0.75],
            [0.00, 0.75, 0.00],
            [0.75, 0.00, 0.75],
            [0.75, 0.00, 0.00],
            [0.00, 0.00, 0.75],
        ])
        h1 = int(height * 0.6)
        for i in range(7):
            img[:h1, width * i // 7: width * (i + 1) // 7] = cols[i]
        h2 = int(height * 0.72)
        for i in range(7):
            img[h1:h2, width * i // 7: width * (i + 1) // 7] = cols[6 - i]
        ramp = np.linspace(0, 1, width)
        img[h2:, :, :] = ramp[None, :, None]
        h3 = int(height * 0.86)
        x = np.arange(width)
        # luma texture near fsc-equivalent: cross-colour bait
        detail = 0.5 + 0.45 * np.sign(np.sin(2 * np.pi * x / 4.0))
        img[h3:, :, :] = detail[None, :, None]
        img[h3 + 8:h3 + 28, width // 3:2 * width // 3] = [0.9, 0.1, 0.1]
    elif kind == "zoneplate":
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        r2 = ((xx - width / 2) ** 2 + (yy * 2 - height) ** 2)
        z = 0.5 + 0.5 * np.cos(r2 * np.pi / (width * 2.2))
        img[..., :] = z[..., None]
    else:
        raise ValueError(kind)
    return img


def encode_frame_to_fields(rgb, params: VideoParameters,
                           phase_ids=(1, 2), noise_ire: float = 0.0,
                           chroma_level: float = 1.0,
                           diff_phase_deg: float = 0.0,
                           rng: np.random.Generator | None = None):
    """Encode one frame into two PAL fields.

    `rgb`: single array (progressive) or pair (true interlace).
    Returns list of (field_uint16, phase_id, is_first_field).
    """
    rgb_pair = rgb if isinstance(rgb, (tuple, list)) else (rgb, rgb)
    p = params
    aw = p.active_width
    first_active_line = p.first_active_field_line

    pic_lines = p.field_height - first_active_line
    Ychi = []
    for r in rgb_pair:
        Hh, Ww = r.shape[:2]
        ys = (np.arange(2 * pic_lines) * Hh // (2 * pic_lines)).clip(0, Hh - 1)
        xs = (np.arange(aw) * Ww // aw).clip(0, Ww - 1)
        frame = r[ys][:, xs]
        yuv = frame @ RGB_TO_YUV.T
        Y = IRE_BLACK + yuv[..., 0] * (IRE_WHITE - IRE_BLACK)
        U = yuv[..., 1] * (IRE_WHITE - IRE_BLACK)
        V = yuv[..., 2] * (IRE_WHITE - IRE_BLACK)
        Ychi.append((Y, chroma_level * U, chroma_level * V))

    if rng is None:
        rng = np.random.default_rng(1234)

    delta = np.deg2rad(diff_phase_deg)
    x_all = np.arange(p.field_width)
    fields = []
    for fi, pid in enumerate(phase_ids):
        Y, U, V = Ychi[fi]
        S = np.full((p.field_height, p.field_width), IRE_BLANK)
        phi0 = _phase_origin(pid)
        voff = _vswitch_offset(pid)
        for line in range(p.field_height):
            phi = phi0 + LINE_ADV * line + (np.pi / 2.0) * x_all
            s = 1.0 if ((line + voff) % 2 == 0) else -1.0
            S[line, :60] = IRE_SYNC
            # swinging burst: (U_b, V_b) = (-B/sqrt2, +B/sqrt2), through
            # the SAME modulation path as picture chroma (no diff phase:
            # burst is the receiver's reference)
            b0, b1 = p.colour_burst_start, p.colour_burst_end
            Bs = chroma_level * BURST_IRE / np.sqrt(2.0)
            S[line, b0:b1] += (-Bs * np.sin(phi[b0:b1])
                               + s * Bs * np.cos(phi[b0:b1]))
            if line >= first_active_line:
                src_row = (line - first_active_line) * 2 + fi
                if src_row < Y.shape[0]:
                    a0, a1 = p.active_video_start, p.active_video_end
                    ph = phi[a0:a1] + delta   # differential phase error
                    S[line, a0:a1] = (Y[src_row]
                                      + U[src_row] * np.sin(ph)
                                      + s * V[src_row] * np.cos(ph))
        if noise_ire > 0:
            S = S + rng.normal(0.0, noise_ire, S.shape)
        scale = (p.white16bIre - p.black16bIre) / 100.0
        raw = np.clip(S * scale + p.black16bIre, 0, 65535).astype("<u2")
        fields.append((raw, pid, fi == 0))
    return fields


def write_tbc(path: str, rgb_frames, params: VideoParameters | None = None,
              noise_ire: float = 0.0, burst_dropout: float = 0.0,
              chroma_level: float = 1.0, diff_phase_deg: float = 0.0,
              rng: np.random.Generator | None = None):
    """Encode RGB frames to `path` (.tbc) + `path`.json (PAL layout)."""
    if params is None:
        params = VideoParameters()
    meta_fields = []
    seq = 1
    pid = 1
    with open(path, "wb") as f:
        for rgb in rgb_frames:
            flds = encode_frame_to_fields(
                rgb, params, phase_ids=(pid, (pid % 8) + 1),
                noise_ire=noise_ire, chroma_level=chroma_level,
                diff_phase_deg=diff_phase_deg, rng=rng)
            if burst_dropout > 0.0:
                if rng is None:
                    rng = np.random.default_rng(77)
                b0, b1 = params.colour_burst_start, params.colour_burst_end
                for raw, _, _ in flds:
                    bad = rng.random(raw.shape[0]) < burst_dropout
                    noise = rng.normal(params.black16bIre, 4000,
                                       (int(bad.sum()), b1 - b0))
                    raw[bad, b0:b1] = np.clip(noise, 0, 65535).astype("<u2")
            for raw, phase_id, is_first in flds:
                f.write(raw.tobytes())
                meta_fields.append({
                    "seqNo": seq,
                    "isFirstField": bool(is_first),
                    "fieldPhaseID": int(phase_id),
                })
                seq += 1
            pid = ((pid + 1) % 8) + 1  # advance 2 in the 8-field sequence

    meta = {
        "videoParameters": {
            "numberOfSequentialFields": len(meta_fields),
            "isSourcePal": True,
            "fsc": FSC_PAL,
            "fSC": FSC_PAL,
            "sampleRate": FS_4FSC_PAL,
            "fieldWidth": params.field_width,
            "fieldHeight": params.field_height,
            "activeVideoStart": params.active_video_start,
            "activeVideoEnd": params.active_video_end,
            "colourBurstStart": params.colour_burst_start,
            "colourBurstEnd": params.colour_burst_end,
            "black16bIre": params.black16bIre,
            "white16bIre": params.white16bIre,
            "firstActiveFieldLine": params.first_active_field_line,
        },
        "fields": meta_fields,
    }
    with open(path + ".json", "w") as f:
        json.dump(meta, f)
