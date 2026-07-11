"""
hvd.encode — Synthetic NTSC composite encoder → .tbc + .tbc.json

Purpose: generate ground-truth test material for the decoder without
hardware. It implements the same signal model the decoder inverts:

    S(x, y) = Y + Re[ chi * exp(i*phi(x,y)) ]        chi = V - iU

with phi advancing 90 deg per sample (4fsc sampling) and 180 deg per
line (227.5 cycles/line), plus the NTSC 4-field phase sequence, sync
tips, and a standard burst on the -U axis.

The output is byte-compatible with ld-decode's .tbc layout so the
decoder exercises the exact same code path as with real captures.
"""

from __future__ import annotations

import json
import numpy as np

from .tbc import VideoParameters, FSC_NTSC, FS_4FSC

# --- colour matrices (BT.601-ish YUV used by NTSC) ---------------------
RGB_TO_YUV = np.array(
    [[0.299, 0.587, 0.114],
     [-0.14713, -0.28886, 0.436],
     [0.615, -0.51499, -0.10001]]
)

IRE_SYNC = -40.0
IRE_BLANK = 0.0
IRE_BLACK = 7.5
IRE_WHITE = 100.0
BURST_IRE = 20.0  # peak amplitude


def _phase_origin(field_phase_id: int) -> float:
    """Subcarrier phase at sample 0 / line 0 of a field.

    NTSC has a 4-field sequence; consecutive fields are offset by
    262.5 * 227.5 cycles => 0.75 cycle => 270 deg steps.
    """
    return (field_phase_id - 1) * (3.0 * np.pi / 2.0)


def make_test_pattern(width: int, height: int, kind: str = "smpte") -> np.ndarray:
    """Return an RGB float image in [0,1], shape (height, width, 3)."""
    img = np.zeros((height, width, 3), dtype=np.float64)
    if kind == "smpte":
        cols = np.array([
            [0.75, 0.75, 0.75],  # grey
            [0.75, 0.75, 0.00],  # yellow
            [0.00, 0.75, 0.75],  # cyan
            [0.00, 0.75, 0.00],  # green
            [0.75, 0.00, 0.75],  # magenta
            [0.75, 0.00, 0.00],  # red
            [0.00, 0.00, 0.75],  # blue
        ])
        h1 = int(height * 0.6)
        for i in range(7):
            x0 = width * i // 7
            x1 = width * (i + 1) // 7
            img[:h1, x0:x1] = cols[i]
        # middle strip: reversed saturated bars
        h2 = int(height * 0.72)
        for i in range(7):
            x0 = width * i // 7
            x1 = width * (i + 1) // 7
            img[h1:h2, x0:x1] = cols[6 - i]
        # bottom: luma ramp + fine luma detail (dot-crawl / cross-colour trap)
        ramp = np.linspace(0, 1, width)
        img[h2:, :, :] = ramp[None, :, None]
        h3 = int(height * 0.86)
        x = np.arange(width)
        # high-frequency luma near fsc-equivalent: classic cross-colour bait
        detail = 0.5 + 0.45 * np.sign(np.sin(2 * np.pi * x / 4.0))
        img[h3:, :, :] = detail[None, :, None]
        # a saturated red box on top of the detail: dot-crawl bait
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
                           rng: np.random.Generator | None = None):
    """Encode a frame into two fields.

    `rgb` may be a single array (progressive: both fields sampled at
    the same instant) or a pair (rgb_at_t, rgb_at_t_plus_half): TRUE
    INTERLACE, each field sampled at its own time — essential for
    realistic motion testing, since real cameras do exactly this.

    Returns list of (field_uint16, phase_id, is_first_field).
    """
    if isinstance(rgb, (tuple, list)):
        rgb_pair = rgb
    else:
        rgb_pair = (rgb, rgb)
    p = params
    aw = p.active_width
    first_active_line = 21

    # Resize inputs to (2*picture_lines, aw) by nearest sampling
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
        # chroma_level simulates gain drift of the source chain (burst
        # scales identically below, which is what ACC exploits)
        Ychi.append((Y, chroma_level * (V - 1j * U)))

    if rng is None:
        rng = np.random.default_rng(1234)

    x_all = np.arange(p.field_width)
    fields = []
    for fi, pid in enumerate(phase_ids):
        Y, chi = Ychi[fi]
        S = np.full((p.field_height, p.field_width), IRE_BLANK)
        phi0 = _phase_origin(pid)
        for line in range(p.field_height):
            phi = phi0 + np.pi * line + (np.pi / 2.0) * x_all
            # sync tip (simplified: front 60 samples)
            S[line, :60] = IRE_SYNC
            # burst on -U axis: chroma = Re[(0 - i*(-A)) e^{i phi}] = A sin(phi)
            b0, b1 = p.colour_burst_start, p.colour_burst_end
            S[line, b0:b1] += chroma_level * BURST_IRE * np.sin(phi[b0:b1])
            if line >= first_active_line:
                src_row = (line - first_active_line) * 2 + fi
                if src_row < frame.shape[0]:
                    a0, a1 = p.active_video_start, p.active_video_end
                    S[line, a0:a1] = (
                        Y[src_row]
                        + np.real(chi[src_row] * np.exp(1j * phi[a0:a1]))
                    )
        if noise_ire > 0:
            S = S + rng.normal(0.0, noise_ire, S.shape)
        scale = (p.white16bIre - p.black16bIre) / 100.0
        raw = np.clip(S * scale + p.black16bIre, 0, 65535).astype("<u2")
        fields.append((raw, pid, fi == 0))
    return fields


def write_tbc(path: str, rgb_frames, params: VideoParameters | None = None,
              noise_ire: float = 0.0, burst_dropout: float = 0.0,
              chroma_level: float = 1.0,
              rng: np.random.Generator | None = None):
    """Encode a list of RGB frames to `path` (.tbc) + `path`.json.

    Each entry may be a single array (progressive) or a pair
    (field1_image, field2_image) for true interlace."""
    if params is None:
        params = VideoParameters()
    meta_fields = []
    seq = 1
    pid = 1
    with open(path, "wb") as f:
        for rgb in rgb_frames:
            flds = encode_frame_to_fields(
                rgb, params, phase_ids=(pid, ((pid) % 4) + 1),
                noise_ire=noise_ire, chroma_level=chroma_level)
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
            pid = ((pid + 1) % 4) + 1  # advance 2 in the 4-field sequence

    meta = {
        "videoParameters": {
            "numberOfSequentialFields": len(meta_fields),
            "isSourcePal": False,
            "fsc": FSC_NTSC,
            "fSC": FSC_NTSC,
            "sampleRate": FS_4FSC,
            "fieldWidth": params.field_width,
            "fieldHeight": params.field_height,
            "activeVideoStart": params.active_video_start,
            "activeVideoEnd": params.active_video_end,
            "colourBurstStart": params.colour_burst_start,
            "colourBurstEnd": params.colour_burst_end,
            "black16bIre": params.black16bIre,
            "white16bIre": params.white16bIre,
        },
        "fields": meta_fields,
    }
    with open(path + ".json", "w") as f:
        json.dump(meta, f)
