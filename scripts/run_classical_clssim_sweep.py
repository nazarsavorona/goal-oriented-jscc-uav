"""Classical separated-coding baselines on the DINOv2 cls-sim metric.

Every new model is compared against JPEG+LDPC and H.264+LDPC. This is the
*goal-oriented* (cls-sim) form of that separated-coding baseline. It is intentionally
self-contained so it runs CPU-only inside the ``semvideo:latest`` Docker image,
which lacks ``pyldpc`` / ``av`` / a system ``ffmpeg`` (so the heavier
``run_classical_sweep.py`` real-LDPC pipeline cannot run there).

Pipeline per (codec, SNR, code-rate) cell (separated source and channel coding)::

    raw frame (CBR ~1/8 matched bpp)
        -> source codec (JPEG at matched quality | H.264 via cv2 if available)
        -> THRESHOLD-LDPC channel model (see below)
        -> if recovered: source-decode the bitstream  -> reconstructed frame
        -> if lost     : "cliff" mid-grey frame (receiver has nothing usable)
        -> push BOTH original and reconstruction through the SAME DINOv2-small
           cls-token cosine used by ``eval_r1_paired.py`` (facebook/dinov2-small,
           518 px, per-frame) -> cls_sim

THRESHOLD-LDPC MODELLING ASSUMPTION (as stated in the paper):
    Uncoded BPSK BER over AWGN:  BER(SNR) = 0.5 * erfc( sqrt(10^(SNR/10)) ).
    A strong capacity-approaching outer code (long-block LDPC) is modelled as a
    hard cliff at a fixed post-channel BER threshold for its rate:
        BER  < threshold  ==>  outer code recovers ALL bits  ==> decode source
        BER >= threshold  ==>  outer code FAILS               ==> cliff frame
    This is the standard optimistic separated-coding abstraction (DeepJSCC-l,
    Bourtsoulatze et al. 2019, Sec. III): it gives the classical scheme the
    maximum credit a strong code can provide and still exposes the characteristic
    "cliff effect" / "levelling-off" that JSCC removes. Two thresholds are swept:
        rate 1/2 long-block LDPC : BER_threshold = 1e-3
        rate 3/4 strong code     : BER_threshold = 1e-4
    The SNR at which BER crosses the threshold is the *only* quantity that drives
    the cliff; absolute BD-rate is not claimed.

Usage (CPU, inside Docker, alongside busy GPUs)::

    python scripts/run_classical_clssim_sweep.py \
        --frames-dir /data/dji/frames_512_v1/val \
        --label dji \
        --max-frames 200 --size 512 --cbr 0.125 \
        --snr-grid 0 5 10 15 20 \
        --codecs jpeg h264 \
        --out experiments/classical_clssim/dji

Outputs under ``--out``::
    metrics.csv      one row per (codec, code_rate/threshold, snr_db)
    summary.txt      human-readable cls-sim-vs-SNR table with the cliff
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# DINOv2-small loaded exactly like scripts/eval_r1_paired.py (transformers AutoModel)
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Channel model (threshold LDPC)
# ---------------------------------------------------------------------------


def bpsk_ber(snr_db: float) -> float:
    """Uncoded BPSK bit-error-rate over AWGN at the given SNR (dB)."""
    snr_lin = 10.0 ** (snr_db / 10.0)
    return 0.5 * math.erfc(math.sqrt(snr_lin))


# Threshold LDPC operating points: (label, code_rate, BER threshold).
LDPC_OPERATING_POINTS = [
    ("LDPC_r1/2", "1/2", 1e-3),
    ("LDPC_r3/4", "3/4", 1e-4),
]


def cliff_frame(shape: tuple[int, int, int]) -> np.ndarray:
    """The decoder-failed output: mid-grey (128) frame.

    Mid-grey caps the PSNR a failed classical codec can deliver and, for the
    goal-oriented metric, drives cls-sim toward the DINOv2 self-similarity of a
    flat field (typically low but non-zero)."""
    return np.full(shape, 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Source codecs
# ---------------------------------------------------------------------------


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


def jpeg_bpp(jpeg_bytes: bytes, h: int, w: int) -> float:
    return 8.0 * len(jpeg_bytes) / (h * w)


def pick_jpeg_quality_for_bpp(img_uint8: np.ndarray, target_bpp: float) -> tuple[int, bytes, float]:
    """Choose the JPEG quality whose source bpp is closest to ``target_bpp``.

    Returns ``(quality, encoded_bytes, achieved_bpp)``. Searches the standard
    quality ladder; JPEG's floor bpp (~0.2 at q=10 on 512px aerial frames) may
    exceed very small targets, in which case the smallest-bpp quality is used
    and the achieved bpp is reported honestly (no fabrication)."""
    h, w = img_uint8.shape[:2]
    best = None
    for q in (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95):
        b = jpeg_encode(img_uint8, q)
        bpp = jpeg_bpp(b, h, w)
        d = abs(bpp - target_bpp)
        if best is None or d < best[3]:
            best = (q, b, bpp, d)
    assert best is not None
    return best[0], best[1], best[2]


def h264_available() -> bool:
    """Probe whether cv2.VideoWriter can actually open an H.264 (avc1) writer."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    try:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        w = cv2.VideoWriter(tmp.name, fourcc, 30, (64, 64))
        ok = w.isOpened()
        w.release()
        return ok
    except Exception:  # noqa: BLE001
        return False
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def h264_roundtrip(img_uint8: np.ndarray, fourcc_tag: str) -> tuple[np.ndarray | None, int]:
    """Single-frame H.264/MP4 encode + decode via cv2. Returns (frame, bytes)."""
    h, w = img_uint8.shape[:2]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
        writer = cv2.VideoWriter(tmp.name, fourcc, 30, (w, h))
        if not writer.isOpened():
            return None, 0
        writer.write(cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        writer.release()
        nbytes = os.path.getsize(tmp.name)
        cap = cv2.VideoCapture(tmp.name)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None, nbytes
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), nbytes
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# DINOv2 cls-sim -- same conventions as scripts/eval_r1_paired.py
# ---------------------------------------------------------------------------


def load_dinov2(device: str) -> torch.nn.Module:
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    model.to(device).eval()
    return model


@torch.no_grad()
def dinov2_cls(dinov2: torch.nn.Module, x: torch.Tensor, target_size: int = 518) -> torch.Tensor:
    """cls-token embedding. x: (B,3,H,W) float in [0,1]. Matches eval_r1_paired.py."""
    x = F.interpolate(x.clamp(0, 1), size=(target_size, target_size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    out = dinov2(pixel_values=x)
    return out.last_hidden_state[:, 0]


def to_tensor(imgs_uint8: list[np.ndarray], device: str) -> torch.Tensor:
    arr = np.stack(imgs_uint8).astype(np.float32) / 255.0  # (B,H,W,3)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)


@torch.no_grad()
def cls_sim_batch(
    dinov2: torch.nn.Module, orig_uint8: list[np.ndarray], rec_uint8: list[np.ndarray], device: str
) -> tuple[float, float]:
    """Mean +/- std cls-sim over a list of (orig, rec) frame pairs."""
    z_o = dinov2_cls(dinov2, to_tensor(orig_uint8, device))
    z_r = dinov2_cls(dinov2, to_tensor(rec_uint8, device))
    sims = F.cosine_similarity(z_r, z_o, dim=-1)
    return float(sims.mean().item()), float(sims.std().item())


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------


def load_frames(frames_dir: Path, max_frames: int, size: int | None) -> list[np.ndarray]:
    paths: list[Path] = []
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        paths.extend(frames_dir.rglob(f"*{ext}"))
    paths = sorted(set(paths))[:max_frames]
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


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Classical JPEG/H.264 + threshold-LDPC on DINOv2 cls-sim.")
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--label", required=True, help="Dataset label for the CSV (e.g. dji / visdrone).")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--cbr", type=float, default=0.125,
                    help="Channel-bandwidth ratio (real symbols / source pixel-dims) to match. "
                         "Source bpp target = cbr * bits/symbol * 3(RGB) * code_rate. With BPSK "
                         "(1 bit/symbol) and rate 1/2: target_bpp = 0.125*1*3*0.5 = ~0.1875 floor; "
                         "we report the JPEG quality whose bpp is closest and the achieved bpp.")
    ap.add_argument("--snr-grid", nargs="+", type=float, default=[0, 5, 10, 15, 20])
    ap.add_argument("--codecs", nargs="+", default=["jpeg"], choices=["jpeg", "h264"])
    ap.add_argument("--batch-size", type=int, default=8, help="DINOv2 forward batch size.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frames = load_frames(args.frames_dir, args.max_frames, args.size)
    h, w = frames[0].shape[:2]
    print(f"[clssim] {args.label}: loaded {len(frames)} frames at {frames[0].shape} from {args.frames_dir}")

    # Target source bpp for CBR matching, assuming BPSK (1 bit/symbol), rate 1/2 nominal.
    # CBR = symbols / (3*H*W); source_bits = symbols * rate; bpp = 8*source_bytes/(H*W).
    # target_bpp = cbr * 3 * bits_per_symbol * rate. Use rate 1/2, BPSK.
    target_bpp = args.cbr * 3.0 * 1.0 * 0.5
    print(f"[clssim] CBR={args.cbr} -> target source bpp ~ {target_bpp:.4f} (BPSK, rate 1/2)")

    dinov2 = load_dinov2(args.device)
    print("[clssim] DINOv2-small (facebook/dinov2-small) loaded on", args.device)

    have_h264 = h264_available() if "h264" in args.codecs else False
    h264_tag = "avc1" if have_h264 else "mp4v"
    if "h264" in args.codecs and not have_h264:
        print("[clssim] WARNING: cv2 cannot open an H.264 (avc1) writer; H.264 path UNAVAILABLE.")

    # ------- Pre-encode each codec at the matched bpp; cache decoded frames -------
    codec_recon: dict[str, list[np.ndarray]] = {}
    codec_bpp: dict[str, float] = {}
    codec_q: dict[str, int] = {}
    codec_feasible: dict[str, bool] = {}

    # JPEG: pick a single quality matched to target bpp using the first frame, then
    # encode all frames at that quality (consistent operating point).
    if "jpeg" in args.codecs:
        q, _, _ = pick_jpeg_quality_for_bpp(frames[0], target_bpp)
        recon, bpps = [], []
        for img in frames:
            b = jpeg_encode(img, q)
            bpps.append(jpeg_bpp(b, h, w))
            d = jpeg_decode(b)
            recon.append(d if d is not None else cliff_frame(img.shape))
        codec_recon["jpeg"] = recon
        codec_bpp["jpeg"] = float(np.mean(bpps))
        codec_q["jpeg"] = q
        codec_feasible["jpeg"] = True
        print(f"[clssim] JPEG q={q} achieved mean bpp={codec_bpp['jpeg']:.4f} (target {target_bpp:.4f})")

    if "h264" in args.codecs:
        if have_h264:
            recon, bpps = [], []
            ok_any = False
            for img in frames:
                d, nbytes = h264_roundtrip(img, h264_tag)
                if d is not None:
                    ok_any = True
                    if d.shape != img.shape:
                        d = cv2.resize(d, (w, h))
                    recon.append(d)
                    bpps.append(8.0 * nbytes / (h * w))
                else:
                    recon.append(cliff_frame(img.shape))
            codec_recon["h264"] = recon
            codec_bpp["h264"] = float(np.mean(bpps)) if bpps else float("nan")
            codec_q["h264"] = -1
            codec_feasible["h264"] = ok_any
            print(f"[clssim] H.264 ({h264_tag}) feasible={ok_any} mean bpp={codec_bpp['h264']:.4f}")
        else:
            codec_feasible["h264"] = False

    # ------- Clean (channel-perfect) cls-sim per codec, computed once -------
    def batched_cls(orig: list[np.ndarray], rec: list[np.ndarray]) -> tuple[float, float]:
        means, allsims = [], []
        for i in range(0, len(orig), args.batch_size):
            z_o = dinov2_cls(dinov2, to_tensor(orig[i:i + args.batch_size], args.device))
            z_r = dinov2_cls(dinov2, to_tensor(rec[i:i + args.batch_size], args.device))
            sims = F.cosine_similarity(z_r, z_o, dim=-1)
            allsims.extend(sims.tolist())
        arr = np.asarray(allsims)
        return float(arr.mean()), float(arr.std())

    cliff_recon = [cliff_frame(f.shape) for f in frames]
    cliff_mean, cliff_std = batched_cls(frames, cliff_recon)
    print(f"[clssim] cliff (mid-grey) cls-sim = {cliff_mean:.4f} +/- {cliff_std:.4f}  "
          f"(this is the floor below the LDPC cliff)")

    clean_cls: dict[str, tuple[float, float]] = {}
    for codec in codec_recon:
        if codec_feasible.get(codec, False):
            clean_cls[codec] = batched_cls(frames, codec_recon[codec])
            print(f"[clssim] {codec} clean (channel-perfect) cls-sim = "
                  f"{clean_cls[codec][0]:.4f} +/- {clean_cls[codec][1]:.4f}")

    # ------- Sweep: codec x LDPC operating point x SNR -------
    rows: list[dict] = []
    for codec in args.codecs:
        if not codec_feasible.get(codec, False):
            rows.append({
                "dataset": args.label, "codec": codec, "feasible": 0,
                "note": "codec unavailable in this environment (no ffmpeg / no avc1 writer)",
            })
            continue
        clean_m, clean_s = clean_cls[codec]
        for op_label, code_rate, ber_thresh in LDPC_OPERATING_POINTS:
            for snr in args.snr_grid:
                ber = bpsk_ber(float(snr))
                ok = ber < ber_thresh
                if ok:
                    cls_m, cls_s = clean_m, clean_s  # recovered -> source-decoded recon
                else:
                    cls_m, cls_s = cliff_mean, cliff_std  # cliff
                rows.append({
                    "dataset": args.label,
                    "codec": codec,
                    "jpeg_q": codec_q.get(codec, -1),
                    "source_bpp": round(codec_bpp.get(codec, float("nan")), 5),
                    "cbr": args.cbr,
                    "ldpc": op_label,
                    "code_rate": code_rate,
                    "ber_threshold": ber_thresh,
                    "snr_db": float(snr),
                    "ber": ber,
                    "decode_ok": int(ok),
                    "cls_sim_mean": round(cls_m, 5),
                    "cls_sim_std": round(cls_s, 5),
                    "n_frames": len(frames),
                    "feasible": 1,
                })

    # ------- Write CSV -------
    csv_path = args.out / "metrics.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)

    # ------- Human-readable summary -------
    lines: list[str] = []
    lines.append(f"Classical separated-coding baselines on DINOv2 cls-sim  (dataset={args.label})")
    lines.append(f"frames={len(frames)} size={args.size} cbr={args.cbr} target_bpp~{target_bpp:.4f}")
    lines.append(f"cliff (mid-grey) cls-sim floor = {cliff_mean:.4f}")
    lines.append("")
    for codec in args.codecs:
        if not codec_feasible.get(codec, False):
            lines.append(f"[{codec}] UNAVAILABLE in this environment (no ffmpeg / no avc1 writer)")
            continue
        cm, _ = clean_cls[codec]
        lines.append(f"[{codec}] q={codec_q.get(codec)} bpp={codec_bpp.get(codec):.4f} "
                     f"clean cls-sim={cm:.4f}")
        for op_label, code_rate, ber_thresh in LDPC_OPERATING_POINTS:
            lines.append(f"  {op_label} (rate {code_rate}, BER<{ber_thresh:g}):")
            lines.append("    SNR(dB) |   BER    | OK | cls-sim")
            for snr in args.snr_grid:
                ber = bpsk_ber(float(snr))
                ok = ber < ber_thresh
                cls_m = cm if ok else cliff_mean
                lines.append(f"    {snr:>6.1f} | {ber:>.2e} | {int(ok)}  | {cls_m:.4f}")
            lines.append("")
    summary = "\n".join(lines)
    (args.out / "summary.txt").write_text(summary)
    print("\n" + summary)
    print(f"[clssim] wrote {len(rows)} rows -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
