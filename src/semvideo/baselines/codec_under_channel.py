"""End-to-end classical separated-coding baseline: ffmpeg + LDPC + channel.

The paper's classical separated-coding baseline. Pipeline per clip:

    raw frames
        → ffmpeg H.264 / H.265 encode (CRF q)              [source coding]
        → bytes → bit-stream
        → LDPC encode (rate R, BPSK/QPSK)                   [channel coding]
        → channel (AWGN @ SNR | Bernoulli erasure @ p | BarrageJam @ JNR)
        → LDPC belief-propagation decode
        → bits → bytes
        → ffmpeg decode (corrupted bitstream)               [source decoding]
        → padded with cliff frames if decode truncated      [fail-safe]
        → metrics: PSNR / SSIM / [LPIPS, cls_sim if requested]

This module is intentionally presentation-quality (paper baseline); the per-frame
loop is in pure Python because that is what makes the fail-modes legible.
Throughput is ~1-5 fps at 256x256 on a laptop CPU.

Public function::

    classical_pipeline(frames, channel="awgn", snr_db=10, codec="h264", ...)

Returns ``ClassicalResult`` with decoded frames + metrics + bytes-used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from semvideo.baselines.ffmpeg_codec import (
    FfmpegResult,
    decode_bitstream,
    encode_decode_video,
)
from semvideo.baselines.ldpc_real import (
    bits_to_bytes,
    bpsk_qpsk_ldpc,
    bytes_to_bits,
)
from semvideo.metrics.psnr_ssim import psnr as psnr_fn
from semvideo.metrics.psnr_ssim import ssim as ssim_fn


@dataclass
class ClassicalResult:
    """Round-trip output of ``classical_pipeline``.

    Attributes:
        frames: Decoded RGB uint8 frames – padded to input length on truncation.
        per_frame_psnr: PSNR per frame in dB (np.inf if identical).
        per_frame_ssim: SSIM per frame in [0, 1].
        ber: Pre-source-decode channel BER measured by the LDPC decoder.
        bytes_source: Bytes used by the source codec (ffmpeg encoded MP4).
        bytes_channel: Bits per channel use × n_channel_uses / 8.
        decode_ok: True if every encoded frame was recovered (no truncation).
        encoder: ``"libx264"`` / ``"libx265"``.
        notes: Free-form diagnostic strings (rate-limited).
    """

    frames: list[np.ndarray]
    per_frame_psnr: list[float]
    per_frame_ssim: list[float]
    ber: float
    bytes_source: int
    bytes_channel: int
    decode_ok: bool
    encoder: str
    notes: list[str] = field(default_factory=list)

    @property
    def mean_psnr(self) -> float:
        finite = [p for p in self.per_frame_psnr if np.isfinite(p)]
        return float(np.mean(finite)) if finite else float("nan")

    @property
    def mean_ssim(self) -> float:
        return float(np.mean(self.per_frame_ssim)) if self.per_frame_ssim else float("nan")


def _cliff_frame(shape: tuple[int, int, int]) -> np.ndarray:
    """Gray fill (= LDPC decode-failure / source-decoder gives up)."""
    return np.full(shape, 128, dtype=np.uint8)


def _per_frame_metrics(
    orig: list[np.ndarray], rec: list[np.ndarray]
) -> tuple[list[float], list[float]]:
    psnrs: list[float] = []
    ssims: list[float] = []
    for a, b in zip(orig, rec, strict=False):
        ta = torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tb = torch.from_numpy(b).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        psnrs.append(psnr_fn(tb, ta))
        ssims.append(ssim_fn(tb, ta))
    return psnrs, ssims


def _apply_channel(
    bits: np.ndarray,
    channel: str,
    snr_db: float,
    erasure_p: float,
    jnr_db: float,
    modulation: str,
    code_rate: str,
    bp_iter: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Run the LDPC + physical-channel + LDPC-decode round-trip on a bit vector.

    For AWGN the LDPC module already integrates the noise. For erasure / jam we
    pre-corrupt the bit vector OR add extra interference before calling
    ``bpsk_qpsk_ldpc`` at the configured SNR.

    Returns ``(decoded_bits, channel_ber)``.
    """
    if channel == "awgn":
        res = bpsk_qpsk_ldpc(
            bits,
            snr_db=snr_db,
            modulation=modulation,
            code_rate=code_rate,
            bp_iter=bp_iter,
            seed=seed,
        )
        return res.decoded_bits, res.ber

    if channel == "erasure":
        # Bernoulli bit-flip with prob = erasure_p, *then* LDPC at high SNR.
        # This isolates the erasure effect from background noise.
        rng = np.random.default_rng(seed + 1)
        flip = rng.random(size=bits.size) < erasure_p
        corrupted = bits.copy()
        corrupted[flip] ^= 1
        res = bpsk_qpsk_ldpc(
            corrupted,
            snr_db=snr_db,
            modulation=modulation,
            code_rate=code_rate,
            bp_iter=bp_iter,
            seed=seed,
        )
        return res.decoded_bits, res.ber

    if channel == "barrage_jam":
        # BarrageJam = AWGN at effective SNR = SNR - JNR after combining.
        # Effective_SNR_lin = SNR_lin * NoisePower / (NoisePower + JamPower).
        # In terms of dB: SNR_eff = -10 log10(10^(-SNR/10) + 10^(JNR/10 - SNR/10))
        # Simplification: power ratio of (signal) : (noise + jammer at JNR over noise)
        snr_lin = 10.0 ** (snr_db / 10.0)
        jnr_lin = 10.0 ** (jnr_db / 10.0)
        # Jammer power relative to signal: jnr_lin / snr_lin (since JNR is jammer/noise)
        # Effective signal-to-(noise+jam) ratio:
        eff_snr_lin = snr_lin / (1.0 + jnr_lin)
        eff_snr_db = 10.0 * np.log10(max(eff_snr_lin, 1e-12))
        res = bpsk_qpsk_ldpc(
            bits,
            snr_db=float(eff_snr_db),
            modulation=modulation,
            code_rate=code_rate,
            bp_iter=bp_iter,
            seed=seed,
        )
        return res.decoded_bits, res.ber

    raise ValueError(f"channel must be awgn / erasure / barrage_jam, got {channel!r}")


def classical_pipeline(
    frames: list[np.ndarray],
    codec: str = "h264",
    crf: int = 23,
    channel: str = "awgn",
    snr_db: float = 10.0,
    erasure_p: float = 0.0,
    jnr_db: float = 10.0,
    modulation: str = "bpsk",
    code_rate: str = "1/2",
    bp_iter: int = 30,
    fps: int = 30,
    keyint: int | None = None,
    seed: int = 0,
    max_bits: int | None = None,
) -> ClassicalResult:
    """Run a real classical separated source-channel coding baseline.

    Args:
        frames: List of RGB uint8 frames (H, W, 3).
        codec: ``"h264"`` (libx264) or ``"h265"`` (libx265).
        crf: ffmpeg constant rate factor (lower = higher quality).
        channel: ``"awgn"``, ``"erasure"``, or ``"barrage_jam"``.
        snr_db: Eb/N0 for the AWGN portion of the channel.
        erasure_p: per-bit erasure (XOR-flip after channel) probability.
        jnr_db: jammer-to-noise ratio for ``barrage_jam``.
        modulation: ``"bpsk"`` or ``"qpsk"``.
        code_rate: LDPC code rate (``"1/2"`` or ``"3/4"``).
        bp_iter: max belief-propagation iterations.
        fps: video FPS for the source codec.
        keyint: GOP / keyframe interval (defaults to ffmpeg default).
        seed: master seed for all PRNGs.
        max_bits: if set, only push the first ``max_bits`` of the bitstream
            through the channel (the rest is assumed received correctly). This
            is a *speed* knob for very long clips at high CRF; defaults to
            ``None`` (whole bitstream through LDPC).

    Returns:
        ClassicalResult with frames + metrics + diagnostics.
    """
    if not frames:
        raise ValueError("classical_pipeline: empty frame list")
    shape = frames[0].shape
    n = len(frames)

    # ── 1) Source coding: ffmpeg encode + immediate decode for round-trip MP4 ──
    enc: FfmpegResult = encode_decode_video(frames, codec=codec, crf=crf, fps=fps, keyint=keyint)
    bitstream = enc.bitstream
    bits = bytes_to_bits(bitstream)

    # ── 2) Channel coding: LDPC encode + channel + LDPC decode ──
    if max_bits is not None and bits.size > max_bits:
        head = bits[:max_bits]
        tail = bits[max_bits:]
        decoded_head, ber = _apply_channel(
            head,
            channel,
            snr_db,
            erasure_p,
            jnr_db,
            modulation,
            code_rate,
            bp_iter,
            seed,
        )
        decoded_bits = np.concatenate([decoded_head, tail])
    else:
        decoded_bits, ber = _apply_channel(
            bits,
            channel,
            snr_db,
            erasure_p,
            jnr_db,
            modulation,
            code_rate,
            bp_iter,
            seed,
        )

    decoded_bytes = bits_to_bytes(decoded_bits)
    bytes_channel = (decoded_bits.size + 7) // 8

    # ── 3) Source decoding (may fail / truncate on heavy corruption) ──
    decoded_frames = decode_bitstream(decoded_bytes)
    notes: list[str] = []
    if len(decoded_frames) < n:
        notes.append(
            f"ffmpeg decode truncated: {len(decoded_frames)}/{n} frames recovered "
            f"(BER={ber:.2e}); padding with cliff frames."
        )
        decoded_frames = decoded_frames + [
            _cliff_frame(shape) for _ in range(n - len(decoded_frames))
        ]
    elif len(decoded_frames) > n:
        decoded_frames = decoded_frames[:n]

    # ── 4) Per-frame metrics ──
    psnrs, ssims = _per_frame_metrics(frames, decoded_frames)

    return ClassicalResult(
        frames=decoded_frames,
        per_frame_psnr=psnrs,
        per_frame_ssim=ssims,
        ber=ber,
        bytes_source=enc.bytes_used,
        bytes_channel=bytes_channel,
        decode_ok=(len(notes) == 0),
        encoder=enc.encoder,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────
# Optional: optional LPIPS / DINOv2 hooks (lazy, best-effort)
# ─────────────────────────────────────────────────────────────────────
def extra_perceptual_metrics(
    orig: list[np.ndarray], rec: list[np.ndarray], device: str = "cpu"
) -> dict[str, float | None]:
    """Compute LPIPS + DINOv2 cls_sim averaged over the clip.

    Returns ``{"lpips": float | None, "cls_sim": float | None}``. Each metric is
    ``None`` if its backbone is not available – this matches the demo's
    fail-soft behaviour and lets the sweep CSV stay schema-stable.
    """
    out: dict[str, float | None] = {"lpips": None, "cls_sim": None}

    # LPIPS
    try:
        from semvideo.metrics.lpips_fid import LPIPSMetric

        lp = LPIPSMetric(device=device)
        vals = []
        for a, b in zip(orig, rec, strict=False):
            ta = torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            tb = torch.from_numpy(b).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            vals.append(float(lp(ta, tb).mean().item()))
        out["lpips"] = float(np.mean(vals)) if vals else None
    except Exception:  # noqa: BLE001
        pass

    # DINOv2 cls_sim (mid frame only – matches demo's averaging convention)
    try:
        import torch.nn.functional as F

        dn = (
            torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitb14",
                verbose=False,
                trust_repo=True,
            )
            .to(device)
            .eval()
        )
        mid = len(orig) // 2

        def _emb(img: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            t = F.interpolate(t, size=(518, 518), mode="bicubic", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            with torch.no_grad():
                return dn((t - mean) / std)

        out["cls_sim"] = float(F.cosine_similarity(_emb(orig[mid]), _emb(rec[mid])).item())
    except Exception:  # noqa: BLE001
        pass

    return out


def _summarise(
    result: ClassicalResult, extras: dict[str, float | None] | None = None
) -> dict[str, Any]:
    """Flatten a ClassicalResult + optional extras into a CSV-friendly dict."""
    bpp = (
        8
        * result.bytes_source
        / (result.frames[0].shape[0] * result.frames[0].shape[1] * len(result.frames))
        if result.frames
        else 0.0
    )
    row: dict[str, Any] = {
        "encoder": result.encoder,
        "psnr_mean": result.mean_psnr,
        "ssim_mean": result.mean_ssim,
        "ber": result.ber,
        "bytes_source": result.bytes_source,
        "bytes_channel": result.bytes_channel,
        "bpp": bpp,
        "decode_ok": int(result.decode_ok),
        "n_frames": len(result.frames),
    }
    if extras:
        row.update({k: (v if v is not None else "") for k, v in extras.items()})
    return row
