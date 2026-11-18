"""Erasure-channel cross-corpus eval for SwinJSCC codecs (P49/P68/P80) -- the
erasure half of the threat-specificity 2x2 (reactive half = eval_jamming_cross_corpus.py).

Mirrors eval_jamming_cross_corpus's monkey-patch approach: replaces SwinJSCC's
built-in Gaussian channel with a Bernoulli ERASURE on the (B,N,C) latent (symbols
dropped to zero with prob p; iid or burst). Per-tensor unit-power normalisation is
preserved exactly as in the jamming script so the two are directly comparable.

Outputs per_frame.csv (cls_sim per image) + summary.csv, same schema/columns as the
jamming script. Question: does the erasure-aug codec (P68) survive erasure better than
the reactive-aug codec (P80), confirming each specialist wins its own threat?

Usage (run from the repository root; ABSOLUTE --out + --ckpt, the script chdir's to SwinJSCC):
    python scripts/eval_erasure_cross_corpus.py --ckpt /work/experiments/.../best.pt \\
        --label P68 --p-grid 0.1 0.3 0.5 --snr 20 --n-dji 375 \\
        --dji-val-dir /data/dji/frames_512_v1/val --out /work/experiments/erasure_P68
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMVIDEO_ROOT = Path(__file__).resolve().parents[1]
# Upstream SwinJSCC code is not distributed with this repository (see README, "Third-party
# code"). Set SWINJSCC_DIR, or place it at <parent of this repository>/third_party/SwinJSCC
# (/third_party/SwinJSCC when the repository is mounted at /workspace in the Docker image).
SWINJSCC_DIR = Path(os.environ.get("SWINJSCC_DIR", REPO_ROOT / "third_party" / "SwinJSCC"))
sys.path.insert(0, str(SEMVIDEO_ROOT / "scripts"))
sys.path.insert(0, str(SEMVIDEO_ROOT / "src"))
sys.path.insert(0, str(SWINJSCC_DIR))
os.chdir(str(SWINJSCC_DIR))

from eval_swinjscc_dinov2 import build_swinjscc  # noqa: E402
from eval_benchmark_sweep import DINOv2Extractor, cossim, list_images, psnr  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ErasurePatcher:
    """Monkey-patch net.feature_pass_channel to apply Bernoulli erasure to the latent."""

    def __init__(self, net):
        self.net = net
        self._orig = net.feature_pass_channel
        self.p = 0.0
        self.mode = "iid"
        self.mbl = 16

    def _patched(self, feature, chan_param, avg_pwr=False):
        if self.p <= 0.0:
            return self._orig(feature, chan_param, avg_pwr)
        # unit-power normalise (mirror SwinJSCC Channel + the jamming patcher)
        if avg_pwr:
            tx = feature / torch.sqrt(avg_pwr * 2); scale = torch.sqrt(avg_pwr * 2)
        else:
            pwr = torch.mean(feature ** 2) * 2
            tx = feature / torch.sqrt(pwr); scale = torch.sqrt(pwr)
        shp = tx.shape
        flat = tx.reshape(-1)
        n = flat.numel()
        if self.mode == "burst":
            # vectorised: one Bernoulli(p) per block of mbl symbols -> bursty erasure,
            # expected erased fraction = p. (No per-symbol Python loop.)
            nblocks = (n + self.mbl - 1) // self.mbl
            block_keep = (torch.rand(nblocks, device=flat.device) >= self.p).float()
            mask = block_keep.repeat_interleave(self.mbl)[:n]
        else:  # iid
            mask = (torch.rand(n, device=flat.device) >= self.p).float()
        noisy = (flat * mask).reshape(shp)
        return noisy * scale

    def __enter__(self):
        self.net.feature_pass_channel = self._patched
        return self

    def __exit__(self, *a):
        self.net.feature_pass_channel = self._orig
        return False


def center_square_resize(bgr, size):
    h, w = bgr.shape[:2]
    s = min(h, w); y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(bgr[y0:y0 + s, x0:x0 + s], (size, size), interpolation=cv2.INTER_AREA)


@torch.no_grad()
def reconstruct(net, crop_bgr, snr_db):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    rec, *_ = net(x, given_SNR=snr_db, given_rate=None)
    rec = rec.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor((rec * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--C", type=int, default=96)
    ap.add_argument("--codec-size", type=int, default=512)
    ap.add_argument("--snr", type=int, default=20)
    ap.add_argument("--p-grid", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    ap.add_argument("--erasure-mode", choices=["iid", "burst"], default="iid")
    ap.add_argument("--mbl", type=int, default=16)
    ap.add_argument("--dji-val-dir", default=str(REPO_ROOT / "data" / "dji" / "frames_512_v1" / "val"))
    ap.add_argument("--bench-label", default="DJI_val")
    ap.add_argument("--n-dji", type=int, default=375)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    paths = sorted(p for p in Path(args.dji_val_dir).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.n_dji > 0 and len(paths) > args.n_dji:
        idx = np.linspace(0, len(paths) - 1, args.n_dji, dtype=int)
        paths = [paths[int(i)] for i in idx]
    print(f"[setup] {args.label} on {args.bench_label}: {len(paths)} imgs, snr={args.snr}, p_grid={args.p_grid}", flush=True)

    dino = DINOv2Extractor()
    net = build_swinjscc(Path(args.ckpt), image_dims=(3, args.codec_size, args.codec_size), C=args.C)
    patcher = ErasurePatcher(net)
    patcher.mode = args.erasure_mode; patcher.mbl = args.mbl

    crops, refs = {}, {}
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        c = center_square_resize(bgr, args.codec_size)
        crops[p.stem] = c; refs[p.stem] = dino.features(c)[0]  # cls token only
    print(f"[setup] cached {len(refs)} refs", flush=True)

    pf = open(out_dir / "per_frame.csv", "w", newline="")
    pw = csv.writer(pf)
    pw.writerow(["label", "bench", "channel", "p", "snr", "image", "cls_sim", "psnr"])
    summ = []
    for p in args.p_grid:
        patcher.p = p
        cls_vals, psnr_vals = [], []
        with patcher:
            for stem, crop in crops.items():
                rec = reconstruct(net, crop, args.snr)
                c = cossim(refs[stem], dino.features(rec)[0])
                ps = psnr(crop, rec)
                cls_vals.append(c); psnr_vals.append(ps)
                pw.writerow([args.label, args.bench_label, "erasure", f"{p:.2f}", args.snr, stem, round(c, 5), round(ps, 3)])
        m = float(np.mean(cls_vals)); sd = float(np.std(cls_vals))
        out = sum(1 for x in cls_vals if x < 0.85) / len(cls_vals)
        summ.append((p, m, sd, out, len(cls_vals)))
        print(f"  p={p:.2f}  cls={m:.4f}+-{sd:.4f}  outage[<0.85]={out:.3f}  PSNR={np.mean(psnr_vals):.2f}", flush=True)
    pf.close()
    with open(out_dir / "summary.csv", "w", newline="") as f:
        sw = csv.writer(f)
        sw.writerow(["label", "bench", "channel", "p", "snr", "n", "cls_mean", "cls_std", "outage"])
        for p, m, sd, out, n in summ:
            sw.writerow([args.label, args.bench_label, "erasure", f"{p:.2f}", args.snr, n, f"{m:.5f}", f"{sd:.5f}", f"{out:.4f}"])
    print(f"[saved] {out_dir}/per_frame.csv + summary.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
