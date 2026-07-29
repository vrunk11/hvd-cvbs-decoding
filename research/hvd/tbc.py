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
    # ld-decode 16-bit domain levels (CVBS 10-bit x 64).
    #
    # WATCH THE NAMES. ld-decode's JSON stores PICTURE BLACK in
    # `black16bIre` -- on NTSC-M that is the 7.5 IRE setup pedestal,
    # 282 x 64 = 18048, NOT the 0 IRE blanking reference (240 x 64 =
    # 15360). The old default here was 15360, i.e. the blanking value
    # under the black name, so synthetic files (which use the default)
    # were accidentally right while every real ld-decode capture came out
    # on a wrong scale. PAL and NTSC-J carry no setup, so black ==
    # blanking there.
    black16bIre: int = 18048        # picture black (7.5 IRE on NTSC-M)
    white16bIre: int = 51200        # 100 IRE
    blanking16bIre: int = 15360     # 0 IRE reference
    setup_ire: float = 7.5          # 7.5 for NTSC-M, 0 for NTSC-J / PAL
    first_active_field_line: int = 21
    last_active_field_line: int = 0   # 0 = field_height
    is_source_pal: bool = False
    sample_rate: float = FS_4FSC
    fsc: float = FSC_NTSC

    @property
    def active_width(self) -> int:
        return self.active_video_end - self.active_video_start

    @property
    def codes_per_ire(self) -> float:
        """16-bit codes per 1 IRE. Referenced to BLANKING, not black."""
        return (self.white16bIre - self.blanking16bIre) / 100.0

    def ire(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw 16-bit samples to TRUE IRE units (float).

        0 IRE is the blanking level and 100 IRE is peak white; the setup
        pedestal is part of the luma signal, not part of the scale. This
        used to divide by (white - black) / 100 and offset by black, which
        on NTSC-M made one unit 331.5 codes instead of the true 358.4 --
        every value 1.081x too large -- and put picture black at 0 IRE
        instead of 7.5.

        The error was invisible in any round trip (encode.py made the same
        mistake, so it cancelled) and invisible on the synthetic files
        (whose black16bIre default was really the blanking value). It bit
        only where an ABSOLUTE IRE reference is used, which is exactly the
        two places that matter: burst_amplitude_ire feeding the ACC (a
        nominal 20 IRE burst measured 21.62, so the ACC applied a permanent
        0.925 gain = 7.5 % desaturation), and yuv_to_rgb16, whose
        IRE_BLACK / IRE_WHITE constants have always assumed true IRE.
        """
        return (raw.astype(np.float64) - self.blanking16bIre) / self.codes_per_ire

    def raw(self, ire: np.ndarray) -> np.ndarray:
        """Inverse of ire(): TRUE IRE -> raw 16-bit sample values."""
        return np.asarray(ire) * self.codes_per_ire + self.blanking16bIre


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
    def open(cls, tbc_path: str, json_path: Optional[str] = None,
             setup_ire: Optional[float] = None) -> "TbcSource":
        """Open a .tbc + .tbc.json pair.

        `setup_ire` overrides the assumed pedestal used to derive the
        blanking (0 IRE) reference from the JSON's picture-black level:
        7.5 for standard NTSC-M, 0.0 for NTSC-J. ld-decode's JSON does not
        record which one applies, so the caller has to say (hvd_decode.py
        passes it from --ntsc-j). Defaults to 7.5.
        """
        if json_path is None:
            json_path = tbc_path + ".json"
        with open(json_path, "r") as f:
            meta = json.load(f)

        vp_raw = meta.get("videoParameters", {})

        # Derive the 0 IRE blanking reference from the picture-black level
        # ld-decode records. Formula from decode-orc's own
        # cvbs_signal_constants.h:
        #     blanking = black - setup * (white - black) / (100 - setup)
        # A file that explicitly carries a blanking level wins over the
        # derivation.
        _black = vp_raw.get("black16bIre", 18048)
        _white = vp_raw.get("white16bIre", 51200)
        _setup = 0.0 if vp_raw.get("isSourcePal", False) else (
            7.5 if setup_ire is None else float(setup_ire))
        _blanking = vp_raw.get("blanking16bIre")
        if _blanking is None:
            _blanking = _black - _setup * (_white - _black) / (100.0 - _setup)
        _blanking = float(_blanking)

        params = VideoParameters(
            field_width=vp_raw.get("fieldWidth", 910),
            field_height=vp_raw.get("fieldHeight", 263),
            active_video_start=vp_raw.get("activeVideoStart", 134),
            active_video_end=vp_raw.get("activeVideoEnd", 894),
            colour_burst_start=vp_raw.get("colourBurstStart", 78),
            colour_burst_end=vp_raw.get("colourBurstEnd", 110),
            black16bIre=_black,
            white16bIre=_white,
            blanking16bIre=_blanking,
            setup_ire=_setup,
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
