"""
hvd.tbc — Reader for ld-decode .tbc files (NTSC, 4fsc).

A .tbc file is raw 16-bit unsigned little-endian composite video,
time-base corrected, sampled at 4×fsc (14 318 181.8 Hz for NTSC),
stored field-by-field. The companion .tbc.json describes geometry
(fieldWidth, fieldHeight, active area, burst window, IRE scaling)
and per-field metadata (fieldPhaseID, isFirstField, ...).

Only what the decoder needs is parsed; unknown keys are ignored so
files from any ld-decode / vhs-decode revision should load.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

FSC_NTSC = 315e6 / 88.0          # 3 579 545.45... Hz
FS_4FSC = 4.0 * FSC_NTSC          # 14 318 181.8 Hz


@dataclass
class VideoParameters:
    field_width: int = 910
    field_height: int = 263
    active_video_start: int = 134
    active_video_end: int = 894
    colour_burst_start: int = 78
    colour_burst_end: int = 110
    black16bIre: int = 15360
    white16bIre: int = 51200
    first_active_field_line: int = 21
    last_active_field_line: int = 0   # 0 = field_height
    is_source_pal: bool = False
    sample_rate: float = FS_4FSC
    fsc: float = FSC_NTSC

    @property
    def active_width(self) -> int:
        return self.active_video_end - self.active_video_start

    def ire(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw 16-bit samples to IRE units (float)."""
        scale = (self.white16bIre - self.black16bIre) / 100.0
        return (raw.astype(np.float64) - self.black16bIre) / scale


@dataclass
class FieldMeta:
    seq_no: int
    is_first_field: bool
    field_phase_id: int  # 1..4 for NTSC
    extra: dict = field(default_factory=dict)


@dataclass
class TbcSource:
    tbc_path: str
    params: VideoParameters
    fields: List[FieldMeta]
    _file: Optional[object] = None

    # ---------------------------------------------------------------- I/O

    @classmethod
    def open(cls, tbc_path: str, json_path: Optional[str] = None) -> "TbcSource":
        if json_path is None:
            json_path = tbc_path + ".json"
        with open(json_path, "r") as f:
            meta = json.load(f)

        vp_raw = meta.get("videoParameters", {})
        params = VideoParameters(
            field_width=vp_raw.get("fieldWidth", 910),
            field_height=vp_raw.get("fieldHeight", 263),
            active_video_start=vp_raw.get("activeVideoStart", 134),
            active_video_end=vp_raw.get("activeVideoEnd", 894),
            colour_burst_start=vp_raw.get("colourBurstStart", 78),
            colour_burst_end=vp_raw.get("colourBurstEnd", 110),
            black16bIre=vp_raw.get("black16bIre", 15360),
            white16bIre=vp_raw.get("white16bIre", 51200),
            # ld-decode ships the ACTUAL active field-line range; the
            # historical hardcoded 21 breaks on sources with different
            # vertical geometry (portability audit item)
            first_active_field_line=vp_raw.get("firstActiveFieldLine", 21),
            last_active_field_line=vp_raw.get("lastActiveFieldLine", 0),
            is_source_pal=vp_raw.get("isSourcePal", False),
            sample_rate=vp_raw.get("sampleRate", FS_4FSC),
            fsc=vp_raw.get("fSC", FSC_NTSC),
        )
        if params.is_source_pal:
            raise ValueError("This decoder is NTSC-only (isSourcePal is true).")

        fields = []
        for i, fm in enumerate(meta.get("fields", [])):
            fields.append(
                FieldMeta(
                    seq_no=fm.get("seqNo", i + 1),
                    is_first_field=fm.get("isFirstField", (i % 2 == 0)),
                    field_phase_id=fm.get("fieldPhaseID", (i % 4) + 1),
                    extra=fm,
                )
            )

        src = cls(tbc_path=tbc_path, params=params, fields=fields)

        # Sanity: infer field count from file size if JSON lacks fields
        fsize = os.path.getsize(tbc_path)
        fld_bytes = params.field_width * params.field_height * 2
        n_in_file = fsize // fld_bytes
        if not fields:
            src.fields = [
                FieldMeta(seq_no=i + 1, is_first_field=(i % 2 == 0),
                          field_phase_id=(i % 4) + 1)
                for i in range(n_in_file)
            ]
        return src

    @property
    def num_fields(self) -> int:
        return len(self.fields)

    @property
    def num_frames(self) -> int:
        return self.num_fields // 2

    def read_field(self, index: int) -> np.ndarray:
        """Return one field as a (field_height, field_width) uint16 array."""
        p = self.params
        n = p.field_width * p.field_height
        offset = index * n * 2
        with open(self.tbc_path, "rb") as f:
            f.seek(offset)
            buf = f.read(n * 2)
        if len(buf) != n * 2:
            raise EOFError(f"Field {index} truncated in {self.tbc_path}")
        return np.frombuffer(buf, dtype="<u2").reshape(p.field_height, p.field_width)

    def read_frame_fields(self, frame_index: int):
        """Return (first_field, second_field) raw arrays + their metadata."""
        i0 = frame_index * 2
        f0, f1 = self.read_field(i0), self.read_field(i0 + 1)
        m0, m1 = self.fields[i0], self.fields[i0 + 1]
        if (not m0.is_first_field) and m1.is_first_field:
            f0, f1, m0, m1 = f1, f0, m1, m0
        return (f0, m0), (f1, m1)
