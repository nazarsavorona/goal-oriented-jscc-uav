"""End-to-end latency profile for SOTA codecs at 512x512 on RTX 3060 Laptop.

Measures the *operator pipeline* (1080 -> 512 downsample + JSCC encode +
channel + JSCC decode + 1080 bicubic upsample) for P12, P13, P15.
Reports median ms/frame and fps over 100 runs after 10 warmup runs.

Output:
  - Console table
  - experiments/latency_512_2026_05_02/results.csv

Run (from the repository root; SEMVIDEO_SOURCE_VIDEO names a 1080p source video):
    python -u scripts/latency_profile_512.py
"""
from __future__ import annotations

import csv
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch

from semvideo.channel.awgn import AWGNChannel
from semvideo.models.deep_jscc import DeepJSCC


DEVICE = "cuda"
# The 1080p source video is not public (see README); point SEMVIDEO_SOURCE_VIDEO at your own.
SOURCE_VIDEO = os.environ.get("SEMVIDEO_SOURCE_VIDEO", "data/source_video.mp4")
OUT_DIR = Path("experiments/latency_512_2026_05_02")
OUT_DIR.mkdir(parents=True, exist_ok=True)


CHECKPOINTS = [
    ("P12", "experiments/p12_deepjscc_mixed_512_*/best.pt"),
    ("P13", "experiments/p13_deepjscc_yolofeat_512_*/best.pt"),
    ("P15", "experiments/p15_deepjscc_ch3_visdronefeat_512_*/best.pt"),
]


def load_codec(glob_pattern: str):
    matches = sorted(glob.glob(glob_pattern))
    if not matches:
        return None
    ck = torch.load(matches[-1], map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    m = DeepJSCC(channel_ratio=cfg.get("channel_ratio", 1/6),
                 base_channels=64).to(DEVICE)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    n_params = sum(p.numel() for p in m.parameters())
    return m, n_params


def profile_codec(model, frame_1080: np.ndarray, n_warmup: int = 10,
                   n_iters: int = 100, snr_db: float = 10.0) -> dict:
    """Return median, p10, p90 timings in ms for the operator pipeline."""
    h, w = frame_1080.shape[:2]
    ch = AWGNChannel(snr_db=float(snr_db)).to(DEVICE)
    times_full = []      # full pipeline
    times_codec = []     # codec only

    for _ in range(n_warmup):
        # Warm caches and JIT
        small = cv2.resize(frame_1080, (512, 512), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
        with torch.no_grad():
            rec = model(x, channel=ch, snr_db=float(snr_db)).clamp(0, 1)
        rec_np = (rec.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rec_bgr = cv2.cvtColor(rec_np, cv2.COLOR_RGB2BGR)
        _ = cv2.resize(rec_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    torch.cuda.synchronize()

    for _ in range(n_iters):
        # Full pipeline timer
        t0 = time.time()
        small = cv2.resize(frame_1080, (512, 512), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
        torch.cuda.synchronize()
        t_codec_start = time.time()
        with torch.no_grad():
            rec = model(x, channel=ch, snr_db=float(snr_db)).clamp(0, 1)
        torch.cuda.synchronize()
        t_codec_end = time.time()
        rec_np = (rec.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rec_bgr = cv2.cvtColor(rec_np, cv2.COLOR_RGB2BGR)
        _ = cv2.resize(rec_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
        t1 = time.time()
        times_full.append((t1 - t0) * 1000)
        times_codec.append((t_codec_end - t_codec_start) * 1000)

    return {
        "full_median_ms": float(np.median(times_full)),
        "full_p10_ms":    float(np.percentile(times_full, 10)),
        "full_p90_ms":    float(np.percentile(times_full, 90)),
        "codec_median_ms": float(np.median(times_codec)),
        "codec_p10_ms":   float(np.percentile(times_codec, 10)),
        "codec_p90_ms":   float(np.percentile(times_codec, 90)),
        "fps_full":       1000.0 / float(np.median(times_full)),
        "fps_codec":      1000.0 / float(np.median(times_codec)),
        "n_iters":        n_iters,
    }


def main():
    # Take a real frame of the source video for realistic conditions
    cap = cv2.VideoCapture(SOURCE_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 11045)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("Failed to read source frame")
    print(f"Test frame: {frame.shape[1]}x{frame.shape[0]}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    rows = []
    for label, ck_glob in CHECKPOINTS:
        loaded = load_codec(ck_glob)
        if loaded is None:
            print(f"  {label}: SKIP (no checkpoint)")
            continue
        model, n_params = loaded
        print(f"\n--- {label} ({n_params:,} params) ---")
        stats = profile_codec(model, frame)
        row = {"model": label, "params": n_params, **stats}
        rows.append(row)
        print(f"  Full pipeline (1080 down + codec + 1080 up): "
              f"{stats['full_median_ms']:.2f} ms (p10={stats['full_p10_ms']:.2f}, "
              f"p90={stats['full_p90_ms']:.2f}) -> {stats['fps_full']:.1f} fps")
        print(f"  Codec only:                                  "
              f"{stats['codec_median_ms']:.2f} ms -> {stats['fps_codec']:.1f} fps")
        del model
        torch.cuda.empty_cache()

    # CSV
    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as fp:
        if rows:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\nSaved -> {csv_path}")

    # Summary table
    print("\n=== Summary table ===")
    print(f"{'Model':<6} | {'Params':>11} | {'Full pipe ms':>12} | {'Codec only ms':>13} | {'Full fps':>10} | {'Codec fps':>11}")
    print("-" * 80)
    for r in rows:
        print(f"  {r['model']:<4} | {r['params']:>11,} | {r['full_median_ms']:>12.2f} | "
              f"{r['codec_median_ms']:>13.2f} | {r['fps_full']:>10.1f} | {r['fps_codec']:>11.1f}")


if __name__ == "__main__":
    main()
