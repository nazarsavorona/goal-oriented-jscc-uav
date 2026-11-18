"""Classical (JPEG/H.264 + LDPC) baseline under barrage / pulsed / single-tone jamming.

Extends the BER-threshold model of the separated-coding baseline to jammed channels and
produces the data for the comparison of JSCC against classical separated coding under jamming.

Mathematical model (consistent with the BER-threshold cliff of the AWGN baseline):

  - Effective signal-to-(noise+jammer) ratio:
        SINR_eff = SNR - 10 log10(1 + JNR_linear)
    where the jammer bandwidth equals the signal bandwidth (barrage), so the
    jammer power is ADDed to the AWGN floor at the receiver.

  - For pulsed jamming with duty-cycle d, the channel is "clean" for fraction
    (1 - d) of symbols and "jammed" for fraction d. With long-block LDPC, the
    code can correct iff the AVERAGE BER stays below the threshold:
        BER_avg = (1 - d) * BER(SNR) + d * BER(SINR_eff)
    The threshold model: if BER_avg < threshold → recover, else cliff.

  - For single-tone jamming, only a fraction f of the modulation alphabet is
    affected (the bin overlapping the tone). Modeled as effective SINR over
    the affected fraction. Conservatively we apply the SAME jammer-power add
    at receiver (worst case) since BPSK does not have native frequency
    selectivity. This is a deliberately pessimistic model for classical.

  - Output: CSV with (codec, jammer_type, jnr_db, ber_threshold, snr_db,
    psnr, ssim, recovered) for plotting against our JSCC codecs at matched
    SNR/jammer parameters.

Usage:
  python scripts/baseline_under_jamming.py \
      --data-dir data/dji/frames_512_v1/val \
      --n-frames 30 \
      --quality 70 \
      --out experiments/baseline_jamming_2026_05_03

Target: a "JSCC vs classical under jamming" plot showing that the
cliff shifts up by ~10 dB under barrage JNR=10 dB while JSCC stays graceful.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMVIDEO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = SEMVIDEO_ROOT / "experiments"
sys.path.insert(0, str(SEMVIDEO_ROOT / "src"))


def bpsk_ber(snr_db: float) -> float:
    snr_lin = 10.0 ** (snr_db / 10.0)
    return 0.5 * math.erfc(math.sqrt(snr_lin))


def jpeg_encode(img_uint8: np.ndarray, quality: int) -> bytes:
    bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def jpeg_decode(jpeg_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def cliff_frame(shape):
    return np.full(shape, 128, dtype=np.uint8)


def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return 99.0
    return 10 * math.log10(255.0 ** 2 / mse)


def effective_sinr_db(snr_db: float, jnr_db: float, duty_cycle: float = 1.0) -> float:
    """Per-symbol effective SINR when the jammer is ON.

    Model: jammer power adds to noise at the receiver. JNR is jammer power
    relative to the AWGN noise floor. With JNR linear j, total interference is
    (1 + j) * noise, so SINR_eff = SNR - 10 log10(1 + j).

    Pulsed handling done separately by averaging BER across off/on symbols.
    """
    jnr_lin = 10.0 ** (jnr_db / 10.0)
    return snr_db - 10.0 * math.log10(1.0 + jnr_lin)


def average_ber_under_jamming(
    snr_db: float, jnr_db: float, jammer_type: str, duty_cycle: float = 0.5
) -> float:
    """Average BER experienced by an LDPC-protected stream under jamming."""
    sinr_jammed = effective_sinr_db(snr_db, jnr_db)
    ber_clean = bpsk_ber(snr_db)
    ber_jammed = bpsk_ber(sinr_jammed)
    if jammer_type == "barrage":
        return ber_jammed
    if jammer_type == "pulsed":
        return (1.0 - duty_cycle) * ber_clean + duty_cycle * ber_jammed
    if jammer_type == "single_tone":
        # CW jammer: pessimistic – apply same SINR_eff (BPSK has no notch)
        return ber_jammed
    raise ValueError(f"unknown jammer_type {jammer_type}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str,
                    default=str(REPO_ROOT / "data" / "dji" / "frames_512_v1" / "val"))
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--quality", type=int, default=70,
                    help="JPEG quality for the source codec")
    ap.add_argument("--snrs", type=int, nargs="+", default=[-5, 0, 5, 10, 15, 20, 25])
    ap.add_argument("--jnrs", type=int, nargs="+", default=[0, 5, 10, 15],
                    help="Jammer-to-noise ratios in dB to sweep")
    ap.add_argument("--jammers", nargs="+",
                    default=["none", "barrage", "pulsed", "single_tone"])
    ap.add_argument("--ber-thresholds", type=str, default="rate12=1e-3,rate34=1e-4")
    ap.add_argument("--out", type=str,
                    default=str(EXPERIMENTS_DIR / "baseline_jamming_2026_05_03"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load eval frames
    paths = sorted(glob(os.path.join(args.data_dir, "*.jpg")))[: args.n_frames]
    frames = []
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is not None:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    print(f"[data] {len(frames)} frames from {args.data_dir}")

    # Pre-encode once
    blobs = [jpeg_encode(f, quality=args.quality) for f in frames]
    avg_size_bytes = float(np.mean([len(b) for b in blobs]))
    h, w = frames[0].shape[:2]
    bpp = 8 * avg_size_bytes / (h * w)
    print(f"[src ] JPEG q={args.quality}: avg {avg_size_bytes:.0f} bytes / "
          f"frame, bpp={bpp:.4f}")

    # Parse thresholds
    thresholds: dict[str, float] = {}
    for kv in args.ber_thresholds.split(","):
        k, v = kv.split("=")
        thresholds[k.strip()] = float(v)
    print(f"[thr ] {thresholds}")

    rows: list[dict] = []
    for jammer in args.jammers:
        for jnr in (args.jnrs if jammer != "none" else [0]):
            for snr in args.snrs:
                if jammer == "none":
                    ber = bpsk_ber(snr)
                else:
                    ber = average_ber_under_jamming(snr, jnr, jammer)
                for thr_name, thr_val in thresholds.items():
                    recovered = ber < thr_val
                    psnrs = []
                    for img, blob in zip(frames, blobs):
                        if recovered:
                            dec = jpeg_decode(blob)
                            if dec is None:
                                dec = cliff_frame(img.shape)
                        else:
                            dec = cliff_frame(img.shape)
                        psnrs.append(psnr_uint8(img, dec))
                    psnr_mean = float(np.mean(psnrs))
                    rows.append({
                        "codec": "JPEG",
                        "quality": args.quality,
                        "ldpc": thr_name,
                        "jammer": jammer,
                        "jnr_db": jnr,
                        "snr_db": snr,
                        "ber": f"{ber:.3e}",
                        "recovered": int(recovered),
                        "psnr_mean": round(psnr_mean, 3),
                        "bpp": round(bpp, 5),
                    })

    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="") as fp:
        if rows:
            wr = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
    print(f"[saved] {csv_path}")

    # Print headline cliff shifts
    print("\nCliff SNR (smallest SNR at which recovery happens):")
    for jammer in args.jammers:
        if jammer == "none":
            jnr_iter = [0]
        else:
            jnr_iter = args.jnrs
        for jnr in jnr_iter:
            for thr_name in thresholds:
                rs = [r for r in rows
                      if r["jammer"] == jammer and r["jnr_db"] == jnr
                      and r["ldpc"] == thr_name and r["recovered"] == 1]
                if rs:
                    smallest = min(r["snr_db"] for r in rs)
                    print(f"  {jammer:>11s} JNR={jnr:>2d} {thr_name}: SNR>= {smallest:>3d} dB")
                else:
                    print(f"  {jammer:>11s} JNR={jnr:>2d} {thr_name}: never recovers in tested range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
