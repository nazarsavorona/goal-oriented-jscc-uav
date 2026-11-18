"""Per-frame cls-sim for the four supervisor-ablation codecs in one invocation.

The supervisor ablation compares four detector supervisors (YOLOv8m, v8x, v11m,
v10m). Independent per-arm intervals cannot separate arms whose means differ by a
small margin. All four arms share one validation split, so the honest test is a
paired per-frame bootstrap of the difference -- which needs per-frame output.

This runs every arm over the SAME frame list in one process and writes a tidy
per-frame CSV, so downstream analysis can pair by image.

Loader and preprocessing are copied verbatim from eval_deepjscc_cls.py (the
proven path for these DeepJSCC checkpoints) so the numbers stay comparable to
the published table.

Usage:
    python eval_supablation_paired.py --arms NAME=CKPT [NAME=CKPT ...] \
        --val-dir DIR --n 375 --snr 10 [--codec-size 256] --out OUT.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from transformers import AutoModel  # noqa: E402

from semvideo.channel.awgn import AWGNChannel  # noqa: E402
from semvideo.models.deep_jscc import DeepJSCC  # noqa: E402

DINO_HF = "facebook/dinov2-small"
DINO_RES = 518


def build_arm(ckpt: str) -> DeepJSCC:
    net = DeepJSCC(channel_ratio=1 / 6, base_channels=64).cuda().eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = sd.get("model_state_dict", sd.get("model_state", sd)) if isinstance(sd, dict) else sd
    missing, _ = net.load_state_dict(state, strict=False)
    assert len(missing) < 0.3 * len(net.state_dict()), f"arch mismatch: {len(missing)} missing"
    return net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="NAME=/path/to/best.pt, one per supervisor arm")
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--n", type=int, default=375)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--codec-size", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.val_dir + "/*.jpg") + glob.glob(args.val_dir + "/*.png"))
    if not paths:
        raise SystemExit(f"no frames under {args.val_dir}")
    idx = np.linspace(0, len(paths) - 1, min(args.n, len(paths)), dtype=int)
    paths = [paths[int(i)] for i in idx]
    print(f"[setup] {len(paths)} frames, SNR={args.snr}, codec_size={args.codec_size}")

    dino = AutoModel.from_pretrained(DINO_HF).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    @torch.no_grad()
    def cls_token(x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x.clamp(0, 1), (DINO_RES, DINO_RES),
                          mode="bilinear", align_corners=False)
        return dino(pixel_values=(x - mean) / std).last_hidden_state[:, 0]

    channel = AWGNChannel(snr_db=args.snr).cuda().eval()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["arm", "image", "snr", "codec_size", "cls_sim"])
        for spec in args.arms:
            name, _, ckpt = spec.partition("=")
            net = build_arm(ckpt)
            sims = []
            with torch.no_grad():
                for p in paths:
                    bgr = cv2.imread(p)
                    if bgr is None:
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
                    t = F.interpolate(t, (args.codec_size, args.codec_size),
                                      mode="bilinear", align_corners=False)
                    rec = net(t, channel)
                    sim = F.cosine_similarity(cls_token(t), cls_token(rec.clamp(0, 1))).item()
                    sims.append(sim)
                    writer.writerow([name, Path(p).stem, args.snr, args.codec_size, f"{sim:.6f}"])
            arr = np.array(sims)
            print(f"[arm] {name:8s} n={len(arr)} cls_mean={arr.mean():.5f} "
                  f"cls_std={arr.std():.5f}")
            del net
            torch.cuda.empty_cache()
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
