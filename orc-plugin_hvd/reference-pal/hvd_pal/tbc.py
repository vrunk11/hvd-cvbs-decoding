"""
hvd_pal.tbc — Reader for ld-decode .tbc files (PAL, 4fsc).

PAL 4fsc geometry (ld-decode conventions):
    fsc         = 4 433 618.75 Hz
    sample rate = 4 * fsc = 17 734 475 Hz
    fieldWidth  = 1135 samples/line   (the true line is 1135.0064
                  samples; the TBC stores 1135, so on the stored grid
                  the subcarrier advances EXACTLY 283.75 cycles/line
                  => 270 deg/line, and the residual 25 Hz offset shows
                  up as a slow phase drift that the burst lock-in
                  measures instead of assuming)
    fieldHeight = 313 lines

This is a deliberately separate architecture from the NTSC reader:
PAL-only, refuses NTSC sources.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

FSC_PAL = 4433618.75
FS_4FSC_PAL = 4.0 * FSC_PAL          # 17 734 475 Hz


@dataclass
class VideoParameters:
    field_width: int = 1135
    field_height: int = 313
    active_video_start: int = 185
    active_video_end: int = 1107
    colour_burst_start: int = 98
    colour_burst_end: int = 138
    black16bIre: int = 16384
    white16bIre: int = 54016
    first_active_field_line: int = 22
    last_active_field_line: int = 0   # 0 = field_height
    is_source_pal: bool = True
    sample_rate: float = FS_4FSC_PAL
    fsc: float = FSC_PAL

    @property
    def active_width(self) -> int:
        return self.active_video_end - self.active_video_start

    def ire(self, raw: np.ndarray) -> np.ndarray:
        scale = (self.white16bIre - self.black16bIre) / 100.0
        return (raw.astype(np.float64) - self.black16bIre) / scale


@dataclass
class FieldMeta:
    seq_no: int
    is_first_field: bool
    field_phase_id: int  # 1..8 for PAL (8-field sequence)
    extra: dict = field(default_factory=dict)


@dataclass
class TbcSource:
    tbc_path: str
    params: VideoParameters
    fields: List[FieldMeta]

    @classmethod
    def open(cls, tbc_path: str, json_path: Optional[str] = None) -> "TbcSource":
        if json_path is None:
            json_path = tbc_path + ".json"
        with open(json_path, "r") as f:
            meta = json.load(f)

        vp_raw = meta.get("videoParameters", {})
        params = VideoParameters(
            field_width=vp_raw.get("fieldWidth", 1135),
            field_height=vp_raw.get("fieldHeight", 313),
            active_video_start=vp_raw.get("activeVideoStart", 185),
            active_video_end=vp_raw.get("activeVideoEnd", 1107),
            colour_burst_start=vp_raw.get("colourBurstStart", 98),
            colour_burst_end=vp_raw.get("colourBurstEnd", 138),
            black16bIre=vp_raw.get("black16bIre", 16384),
            white16bIre=vp_raw.get("white16bIre", 54016),
            first_active_field_line=vp_raw.get("firstActiveFieldLine", 22),
            last_active_field_line=vp_raw.get("lastActiveFieldLine", 0),
            is_source_pal=vp_raw.get("isSourcePal", True),
            sample_rate=vp_raw.get("sampleRate", FS_4FSC_PAL),
            fsc=vp_raw.get("fSC", FSC_PAL),
        )
        if not params.is_source_pal:
            raise ValueError("This decoder is PAL-only (isSourcePal is false); "
                             "use the NTSC decoder for this source.")

        fields = []
        for i, fm in enumerate(meta.get("fields", [])):
            fields.append(FieldMeta(
                seq_no=fm.get("seqNo", i + 1),
                is_first_field=fm.get("isFirstField", (i % 2 == 0)),
                field_phase_id=fm.get("fieldPhaseID", (i % 8) + 1),
                extra=fm,
            ))

        src = cls(tbc_path=tbc_path, params=params, fields=fields)

        fsize = os.path.getsize(tbc_path)
        fld_bytes = params.field_width * params.field_height * 2
        n_in_file = fsize // fld_bytes
        if not fields:
            src.fields = [
                FieldMeta(seq_no=i + 1, is_first_field=(i % 2 == 0),
                          field_phase_id=(i % 8) + 1)
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
        i0 = frame_index * 2
        f0, f1 = self.read_field(i0), self.read_field(i0 + 1)
        m0, m1 = self.fields[i0], self.fields[i0 + 1]
        if (not m0.is_first_field) and m1.is_first_field:
            f0, f1, m0, m1 = f1, f0, m1, m0
        return (f0, m0), (f1, m1)
