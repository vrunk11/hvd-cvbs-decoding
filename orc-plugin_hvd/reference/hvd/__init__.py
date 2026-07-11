"""hvd — Holographic-Variational Decoder for ld-decode NTSC .tbc files."""

from .tbc import TbcSource, VideoParameters
from .decoder import DecoderConfig, decode_frame

__version__ = "0.1.0"
__all__ = ["TbcSource", "VideoParameters", "DecoderConfig", "decode_frame"]
