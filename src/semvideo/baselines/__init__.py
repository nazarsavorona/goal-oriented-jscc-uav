"""Classical separated source-channel baselines for SNR-sweep comparisons.

This package wires real ffmpeg-based source codecs (H.264 / H.265 / optional BPG)
to a real LDPC channel code and a real wireless channel (AWGN, erasure, barrage
jam). It complements the threshold-LDPC approximation used in
``scripts/run_classical_clssim_sweep.py``.

Public API (lazy-imported to avoid pulling torch / cv2 at package load time):

    encode_decode_video   – ffmpeg H.264/H.265 round-trip on a frame list
    bpsk_qpsk_ldpc        – BPSK / QPSK + pyldpc encode / channel / decode
    classical_pipeline    – ffmpeg source -> LDPC -> channel -> LDPC -> ffmpeg

Install prerequisites (see ``INSTALL.md`` in this package for a fuller note)::

    pip install imageio-ffmpeg av pyldpc

A system ffmpeg binary is also detected if present; otherwise we fall back to
the binary shipped by ``imageio-ffmpeg``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "encode_decode_video",
    "bpsk_qpsk_ldpc",
    "bytes_to_bits",
    "bits_to_bytes",
    "classical_pipeline",
]


# Lazy attribute access so simply importing ``semvideo.baselines`` does not
# pull torch / cv2 / pyldpc into the process (helpful for tests that only
# exercise the byte-bit utilities, and for environments where the baseline
# stack isn't fully installed).
def __getattr__(name: str):
    if name == "encode_decode_video":
        from semvideo.baselines.ffmpeg_codec import encode_decode_video as _f

        return _f
    if name == "bpsk_qpsk_ldpc":
        from semvideo.baselines.ldpc_real import bpsk_qpsk_ldpc as _f

        return _f
    if name == "bytes_to_bits":
        from semvideo.baselines.ldpc_real import bytes_to_bits as _f

        return _f
    if name == "bits_to_bytes":
        from semvideo.baselines.ldpc_real import bits_to_bytes as _f

        return _f
    if name == "classical_pipeline":
        from semvideo.baselines.codec_under_channel import classical_pipeline as _f

        return _f
    raise AttributeError(f"module 'semvideo.baselines' has no attribute {name!r}")


if TYPE_CHECKING:  # for IDE / mypy
    from semvideo.baselines.codec_under_channel import (  # noqa: F401
        classical_pipeline,
    )
    from semvideo.baselines.ffmpeg_codec import encode_decode_video  # noqa: F401
    from semvideo.baselines.ldpc_real import (  # noqa: F401
        bits_to_bytes,
        bpsk_qpsk_ldpc,
        bytes_to_bits,
    )
