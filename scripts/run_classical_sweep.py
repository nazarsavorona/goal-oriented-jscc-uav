"""Run the real classical-baseline SNR sweep.

For every (codec, channel, SNR) cell:
    real ffmpeg encode  -> real LDPC encode  -> channel  -> LDPC decode  -> ffmpeg decode
    -> per-frame PSNR/SSIM (+ optional LPIPS / DINOv2 cls_sim)

Writes ``metrics.csv`` with one row per (codec, channel, snr_db, crf, code_rate).

Example::

    python scripts/run_classical_sweep.py \
        --frames-dir data/frames/val \
        --max-frames 32 \
        --codecs h264 h265 \
        --channels awgn erasure barrage_jam \
        --snr-grid 0 5 10 15 20 \
        --crf 23 35 \
        --code-rate 1/2 \
        --out experiments/baselines_real/metrics.csv

Install once::

    pip install imageio-ffmpeg av pyldpc
    # H.265 requires a system ffmpeg with libx265 (the imageio-ffmpeg bundled
    # binary often lacks libx265). On Windows: install ffmpeg from gyan.dev or
    # via `winget install ffmpeg` and ensure it is on PATH.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make ``src/semvideo/...`` importable when running from project root
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from semvideo.baselines.codec_under_channel import (  # noqa: E402
    ClassicalResult,
    classical_pipeline,
    extra_perceptual_metrics,
)


def _load_frames(frames_dir: Path, max_frames: int, size: int | None) -> list[np.ndarray]:
    """Load JPEG/PNG frames from a directory, sorted lexicographically.

    Args:
        frames_dir: Directory containing image files.
        max_frames: Maximum number of frames to load (taken in sort order).
        size: Optional square side – if set, frames are centre-cropped + resized.

    Returns:
        List of RGB uint8 arrays of consistent shape.
    """
    paths: list[Path] = []
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        paths.extend(sorted(frames_dir.rglob(f"*{ext}")))
    paths.sort()
    paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No image frames found under {frames_dir}")
    frames: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if size:
            h, w = img.shape[:2]
            side = min(h, w)
            y0, x0 = (h - side) // 2, (w - side) // 2
            img = img[y0:y0 + side, x0:x0 + side]
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(img.astype(np.uint8))
    return frames


def _run_one(
    frames: list[np.ndarray],
    codec: str,
    crf: int,
    channel: str,
    snr_db: float,
    erasure_p: float,
    jnr_db: float,
    code_rate: str,
    modulation: str,
    bp_iter: int,
    seed: int,
    max_bits: int | None,
    perceptual: bool,
    device: str,
) -> dict:
    t0 = time.time()
    res: ClassicalResult = classical_pipeline(
        frames,
        codec=codec, crf=crf,
        channel=channel, snr_db=snr_db,
        erasure_p=erasure_p, jnr_db=jnr_db,
        modulation=modulation, code_rate=code_rate,
        bp_iter=bp_iter, seed=seed, max_bits=max_bits,
    )
    extras = (
        extra_perceptual_metrics(frames, res.frames, device=device)
        if perceptual else {}
    )
    wall = time.time() - t0
    row = {
        "codec": codec,
        "encoder": res.encoder,
        "crf": crf,
        "channel": channel,
        "snr_db": snr_db,
        "erasure_p": erasure_p,
        "jnr_db": jnr_db,
        "code_rate": code_rate,
        "modulation": modulation,
        "psnr_mean": res.mean_psnr,
        "ssim_mean": res.mean_ssim,
        "ber": res.ber,
        "bytes_source": res.bytes_source,
        "bytes_channel": res.bytes_channel,
        "decode_ok": int(res.decode_ok),
        "n_frames": len(frames),
        "wall_s": wall,
        "lpips": extras.get("lpips", ""),
        "cls_sim": extras.get("cls_sim", ""),
        "notes": " | ".join(res.notes),
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Real H.264/H.265 + LDPC + channel SNR sweep."
    )
    ap.add_argument("--frames-dir", type=Path, required=True,
                    help="Directory of input frames (jpg/png) – sorted lexically.")
    ap.add_argument("--max-frames", type=int, default=32,
                    help="Maximum frames to load (ffmpeg+LDPC is CPU-bound).")
    ap.add_argument("--size", type=int, default=256,
                    help="Square side after centre-crop+resize (None = native).")
    ap.add_argument("--codecs", nargs="+", default=["h264"],
                    choices=["h264", "h265"])
    ap.add_argument("--channels", nargs="+", default=["awgn"],
                    choices=["awgn", "erasure", "barrage_jam"])
    ap.add_argument("--snr-grid", nargs="+", type=float,
                    default=[-5, 0, 5, 10, 15, 20])
    ap.add_argument("--erasure-grid", nargs="+", type=float, default=[0.05],
                    help="Bit-flip probabilities (used when channel=erasure).")
    ap.add_argument("--jnr-grid", nargs="+", type=float, default=[10.0],
                    help="Jammer-to-noise ratios (used when channel=barrage_jam).")
    ap.add_argument("--crf", nargs="+", type=int, default=[23, 35])
    ap.add_argument("--code-rate", default="1/2", choices=["1/2", "3/4"])
    ap.add_argument("--modulation", default="bpsk", choices=["bpsk", "qpsk"])
    ap.add_argument("--bp-iter", type=int, default=30)
    ap.add_argument("--max-bits", type=int, default=200000,
                    help="Cap on bits actually pushed through LDPC – speed knob "
                         "(remainder assumed received correctly). 0 = whole stream.")
    ap.add_argument("--perceptual", action="store_true",
                    help="Also compute LPIPS + DINOv2 cls_sim per cell (slower).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output CSV path (parent dirs created).")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames = _load_frames(args.frames_dir, max_frames=args.max_frames, size=args.size)
    print(f"[sweep] loaded {len(frames)} frames at {frames[0].shape} from {args.frames_dir}")
    print(f"[sweep] codecs={args.codecs} channels={args.channels} "
          f"snrs={args.snr_grid} crfs={args.crf} rate={args.code_rate}")
    max_bits = args.max_bits if args.max_bits > 0 else None

    rows: list[dict] = []
    total = 0
    for codec in args.codecs:
        for crf in args.crf:
            for channel in args.channels:
                # Per-channel sweep axes
                if channel == "erasure":
                    main_axis = args.erasure_grid
                    snr_axis = [args.snr_grid[-1]]                # high SNR, isolate erasure
                elif channel == "barrage_jam":
                    main_axis = args.jnr_grid
                    snr_axis = args.snr_grid
                else:                                              # awgn
                    main_axis = [0.0]
                    snr_axis = args.snr_grid

                for axis_val in main_axis:
                    for snr in snr_axis:
                        try:
                            row = _run_one(
                                frames,
                                codec=codec, crf=crf,
                                channel=channel, snr_db=float(snr),
                                erasure_p=float(axis_val) if channel == "erasure" else 0.0,
                                jnr_db=float(axis_val) if channel == "barrage_jam" else 0.0,
                                code_rate=args.code_rate,
                                modulation=args.modulation,
                                bp_iter=args.bp_iter,
                                seed=args.seed,
                                max_bits=max_bits,
                                perceptual=args.perceptual,
                                device=args.device,
                            )
                            rows.append(row)
                            total += 1
                            print(f"  [{total:3d}] {codec} crf={crf} ch={channel} "
                                  f"snr={snr:>5.1f} axis={axis_val:>5.2f} "
                                  f"psnr={row['psnr_mean']:>6.2f} ssim={row['ssim_mean']:.3f} "
                                  f"ber={row['ber']:.2e} ok={row['decode_ok']}")
                        except Exception as e:                     # noqa: BLE001
                            print(f"  [skip] {codec}/{channel} snr={snr}: {e}")
                            rows.append({
                                "codec": codec, "crf": crf, "channel": channel,
                                "snr_db": snr, "error": str(e),
                            })

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[sweep] wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
