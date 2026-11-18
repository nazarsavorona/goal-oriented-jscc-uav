"""Real H.264 / H.265 round-trip via ffmpeg for classical SNR-sweep baselines.

Runs a real ``libx264`` / ``libx265`` encode-decode pass under a configurable CRF
(instead of an OpenCV `mp4v` fallback).

The encode is performed by spawning a single ``ffmpeg`` subprocess pipe and the
decode by reading the produced MP4 back through OpenCV's VideoCapture. The
bitstream bytes are also returned so that downstream code can feed them into a
real LDPC + channel simulation (see ``codec_under_channel.py``).

Notes on ffmpeg discovery:
    1. We honour a system-wide ``ffmpeg`` on PATH first (lets the user pick the
       build with libx265 enabled, hardware encoders, etc.).
    2. Else, we fall back to ``imageio_ffmpeg.get_ffmpeg_exe()``. ``imageio-ffmpeg``
       ships a static Linux/Win/macOS binary with libx264 + libvpx, which is
       enough for the H.264 baseline; H.265 will only work on the system binary
       in that case.

Tensor / array conventions:
    - frames are passed as a ``list[np.ndarray]`` of shape (H, W, 3), RGB, uint8.
    - bytes-used count includes container overhead (MP4 atoms / Matroska EBML).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FfmpegResult:
    """Holds the round-trip output of ``encode_decode_video``.

    Attributes:
        frames: Decoded frames (RGB uint8) – may be shorter than input on heavy
            corruption; missing frames are NOT padded here so callers can decide
            on the fail-mode (cliff frame vs interpolation).
        bitstream: Raw encoded MP4 container bytes.
        bytes_used: ``len(bitstream)`` – kept as a separate field so callers can
            compute bits-per-pixel without materialising the bytestring.
        encoder: ``"libx264"`` or ``"libx265"`` actually selected.
    """

    frames: list[np.ndarray]
    bitstream: bytes
    bytes_used: int
    encoder: str


def _resolve_ffmpeg() -> str:
    """Return the ffmpeg binary path, preferring a system-wide install.

    Raises:
        RuntimeError: if neither a system ffmpeg nor ``imageio-ffmpeg`` is
            available. The error message includes the install hint.
    """
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg not on PATH and 'imageio-ffmpeg' not installed. Run: pip install imageio-ffmpeg"
        ) from exc


def encode_decode_video(
    frames: list[np.ndarray],
    codec: str = "h264",
    crf: int = 23,
    fps: int = 30,
    preset: str = "medium",
    keyint: int | None = None,
) -> FfmpegResult:
    """Round-trip RGB frames through a real H.264 / H.265 encoder + decoder.

    Args:
        frames: Non-empty list of RGB uint8 frames of identical shape (H, W, 3).
        codec: ``"h264"`` (libx264) or ``"h265"``/``"hevc"`` (libx265).
        crf: Constant Rate Factor – lower is better quality (libx264 default 23,
            libx265 default 28; we use the libx264 default for both for
            apples-to-apples sweep comparability).
        fps: Frame rate baked into the MP4 timing.
        preset: ffmpeg speed/quality preset (ultrafast..veryslow). ``medium``
            matches the paper's default baseline.
        keyint: Force keyframe interval (GOP size). If None, uses ffmpeg default
            (libx264: 250; libx265: 250). Set to 1 for all-intra coding.

    Returns:
        FfmpegResult with decoded frames, raw bitstream, bytes used, encoder name.

    Raises:
        RuntimeError: if ffmpeg is unavailable or the subprocess fails.
        ValueError: if ``frames`` is empty or shapes inconsistent.
    """
    if not frames:
        raise ValueError("encode_decode_video: empty frame list")
    h, w, c = frames[0].shape
    if c != 3:
        raise ValueError(f"encode_decode_video: expected 3 channels, got {c}")
    for i, f in enumerate(frames):
        if f.shape != (h, w, 3):
            raise ValueError(f"encode_decode_video: frame {i} shape {f.shape} != {(h, w, 3)}")
        if f.dtype != np.uint8:
            raise ValueError(f"encode_decode_video: frame {i} dtype {f.dtype} != uint8")

    codec_norm = codec.lower().replace(".", "")
    if codec_norm == "h264":
        encoder = "libx264"
    elif codec_norm in ("h265", "hevc"):
        encoder = "libx265"
    else:
        raise ValueError(f"codec must be 'h264' or 'h265', got {codec!r}")

    ff = _resolve_ffmpeg()

    tmp_dir = Path(tempfile.mkdtemp(prefix="semvideo_ff_"))
    out_path = tmp_dir / "encoded.mp4"

    cmd = [
        ff,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",  # raw frames on stdin
        "-c:v",
        encoder,
        "-crf",
        str(int(crf)),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",  # standard playback-friendly
    ]
    if keyint is not None:
        # libx264 and libx265 both accept -g for GOP size
        cmd += ["-g", str(int(keyint))]
    cmd += [str(out_path)]

    raw = b"".join(f.tobytes() for f in frames)
    try:
        proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=300, check=False)
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"ffmpeg encode failed ({encoder}, rc={proc.returncode}): {err}")
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced no output")

        bitstream = out_path.read_bytes()
        decoded = _decode_mp4(out_path)
    finally:
        try:
            if out_path.exists():
                out_path.unlink()
            os.rmdir(tmp_dir)
        except OSError:
            # Leave the temp dir – the OS will reap it; not worth crashing on.
            pass

    return FfmpegResult(
        frames=decoded,
        bitstream=bitstream,
        bytes_used=len(bitstream),
        encoder=encoder,
    )


def _decode_mp4(path: Path) -> list[np.ndarray]:
    """Decode an MP4 to RGB frames via OpenCV.

    OpenCV's VideoCapture transparently uses ffmpeg under the hood on most
    builds, so we don't need a second ffmpeg invocation. Returns whatever
    frames decoded successfully – truncation is the caller's signal that the
    bitstream was corrupted past recovery.
    """
    cap = cv2.VideoCapture(str(path))
    out: list[np.ndarray] = []
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return out


def decode_bitstream(bitstream: bytes) -> list[np.ndarray]:
    """Decode a possibly-corrupted MP4 byte buffer back to frames.

    Used by ``codec_under_channel.py`` after the bitstream has been mangled by
    the channel + LDPC pipeline. Returns whatever frames decoded; missing
    frames are the caller's problem (typically pad with cliff frames).

    Args:
        bitstream: Raw MP4 container bytes (possibly corrupted).

    Returns:
        Decoded RGB frames (may be empty / truncated).
    """
    tmp = Path(tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name)
    try:
        tmp.write_bytes(bitstream)
        return _decode_mp4(tmp)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
