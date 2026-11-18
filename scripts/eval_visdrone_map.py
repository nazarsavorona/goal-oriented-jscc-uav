"""Downstream-task mAP eval: YOLOv8s-VisDrone on raw vs JSCC-reconstructed
images from VisDrone-DET test-dev. Answers the paper question:
'Does our cls_sim lift translate to actual detection performance?'

Pipeline per image:
  1. Load original 512²-cropped VisDrone test-dev frame + GT boxes (rescaled).
  2. Push through chosen JSCC checkpoint at given SNR/channel.
  3. Run YOLOv8s-VisDrone on both the raw crop and the reconstruction.
  4. Compute per-image mAP@[.5:.95:.05] via torchmetrics, averaged.

Outputs: per_frame.csv + summary.csv with mAP@0.5, mAP@0.5:0.95, count of
detected boxes, and the per-class breakdown.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMVIDEO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = SEMVIDEO_ROOT / "experiments"
# Upstream SwinJSCC code is not distributed with this repository (see README, "Third-party
# code"). Set SWINJSCC_DIR, or place it at <parent of this repository>/third_party/SwinJSCC
# (/third_party/SwinJSCC when the repository is mounted at /workspace in the Docker image).
SWINJSCC_DIR = Path(os.environ.get("SWINJSCC_DIR", REPO_ROOT / "third_party" / "SwinJSCC"))
sys.path.insert(0, str(SEMVIDEO_ROOT / "scripts"))
sys.path.insert(0, str(SWINJSCC_DIR))
os.chdir(str(SWINJSCC_DIR))

from net.network import SwinJSCC  # noqa: E402, F401
from eval_swinjscc_dinov2 import build_swinjscc as _build_swinjscc_awgn  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_swinjscc(checkpoint, image_dims=(3, 512, 512), C=96, channel="awgn"):
    if channel == "awgn":
        return _build_swinjscc_awgn(checkpoint, image_dims=image_dims, C=C)
    sj_args = SimpleNamespace(
        model="SwinJSCC_w/_SA", channel_type=channel, C=str(C),
        multiple_snr="10", distortion_metric="MSE", trainset="DIV2K", logger=None,
    )
    cfg = SimpleNamespace(
        seed=42, pass_channel=True, CUDA=True, device=DEVICE, norm=False,
        print_step=100, plot_step=10000, normalize=False, learning_rate=1e-4,
        image_dims=image_dims, batch_size=1, downsample=4, logger=None,
    )
    cfg.encoder_kwargs = dict(
        model="SwinJSCC_w/_SA", img_size=(image_dims[1], image_dims[2]), patch_size=2,
        in_chans=3, embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2],
        num_heads=[4, 6, 8, 10], C=C, window_size=8, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True,
    )
    cfg.decoder_kwargs = dict(
        model="SwinJSCC_w/_SA", img_size=(image_dims[1], image_dims[2]),
        embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2],
        num_heads=[10, 8, 6, 4], C=C, window_size=8, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True,
    )
    net = SwinJSCC(sj_args, cfg).to(DEVICE).eval()
    state = torch.load(str(checkpoint), map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    target = net.state_dict()
    filtered = {k: v for k, v in state.items() if k in target and target[k].shape == v.shape}
    net.load_state_dict(filtered, strict=False)
    return net


@torch.no_grad()
def reconstruct(net, bgr, snr_db, codec_size):
    h, w = bgr.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = bgr[y0:y0 + side, x0:x0 + side]
    crop = cv2.resize(crop, (codec_size, codec_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    rec, _, _, _, _ = net(x, given_SNR=snr_db, given_rate=None)
    rec = rec.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    rec_bgr = cv2.cvtColor((rec * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return crop, rec_bgr, (y0, x0, side)


# VisDrone class IDs -> COCO-ish names (only counting the 10 we care about,
# excluding 0=ignored and 11=others).
VISDRONE_CLASSES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor",
}


def load_visdrone_anno(anno_path: Path, crop_box, codec_size: int):
    """Read VisDrone annotation, transform to codec-cropped pixel coords.

    Annotation row: x, y, w, h, score, class, trunc, occ – pixel coords on
    the original image. We center-crop to ``side`` then resize to
    ``codec_size``. Returns list of (x1, y1, x2, y2, class_id).
    """
    if not anno_path.exists():
        return []
    y0, x0, side = crop_box
    scale = codec_size / side
    boxes = []
    with open(anno_path) as fp:
        for line in fp:
            parts = line.strip().rstrip(",").split(",")
            if len(parts) < 6:
                continue
            x, y, w, h, _, cls = (int(p) for p in parts[:6])
            if cls == 0 or cls == 11:
                continue
            # transform: subtract crop offset, clip, then scale
            x1 = (x - x0) * scale
            y1 = (y - y0) * scale
            x2 = (x + w - x0) * scale
            y2 = (y + h - y0) * scale
            # clip to crop bounds
            x1 = max(0, min(codec_size - 1, x1))
            y1 = max(0, min(codec_size - 1, y1))
            x2 = max(0, min(codec_size - 1, x2))
            y2 = max(0, min(codec_size - 1, y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append((float(x1), float(y1), float(x2), float(y2), int(cls)))
    return boxes


def run_yolo(yolo, bgr, conf=0.10):
    """Run YOLO on BGR image, return list of (x1, y1, x2, y2, conf, class)."""
    res = yolo.predict(bgr, conf=conf, verbose=False)
    if not res:
        return []
    r = res[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    # mshamrai/yolov8s-visdrone uses VisDrone classes directly (0-indexed
    # variant of the same scheme, but with 'others' too). Map to our 1-10.
    # Convert from 0-indexed (0..9) back to 1-indexed if needed:
    cls = cls + 1  # ultralytics returns 0..K-1; VisDrone GT is 1..10
    return list(zip(xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3], confs, cls))


def box_iou(a, b):
    """Pair-wise IoU between two box lists ([x1,y1,x2,y2,...])."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    aa = np.array([row[:4] for row in a])
    bb = np.array([row[:4] for row in b])
    x1 = np.maximum(aa[:, None, 0], bb[None, :, 0])
    y1 = np.maximum(aa[:, None, 1], bb[None, :, 1])
    x2 = np.minimum(aa[:, None, 2], bb[None, :, 2])
    y2 = np.minimum(aa[:, None, 3], bb[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (aa[:, 2] - aa[:, 0]) * (aa[:, 3] - aa[:, 1])
    area_b = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter + 1e-8
    return inter / union


def per_image_ap50(pred, gt):
    """Per-image AP@0.5 using greedy matching by class."""
    if not gt:
        return None  # no GT – skip image (don't penalize)
    if not pred:
        return 0.0
    # Sort by confidence desc
    pred = sorted(pred, key=lambda r: -r[4])
    iou = box_iou(pred, gt)
    gt_matched = [False] * len(gt)
    tp = []
    fp = []
    for i, p in enumerate(pred):
        cls_p = int(p[5])
        # Find best unmatched GT of same class
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt):
            if gt_matched[j] or int(g[4]) != cls_p:
                continue
            if iou[i, j] >= 0.5 and iou[i, j] > best_iou:
                best_iou = iou[i, j]
                best_j = j
        if best_j >= 0:
            tp.append(1); fp.append(0)
            gt_matched[best_j] = True
        else:
            tp.append(0); fp.append(1)
    tp = np.cumsum(tp); fp = np.cumsum(fp)
    recall = tp / max(1, len(gt))
    prec = tp / np.maximum(1, tp + fp)
    # AP = average precision; for per-image, approximate by trapezoidal area
    # under PR curve.
    ap = 0.0
    prev_r = 0.0
    for r, p in zip(recall, prec):
        ap += (r - prev_r) * p
        prev_r = r
    return float(ap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--visdrone-root", type=str,
                    default=str(REPO_ROOT / "data" / "visdrone" / "VisDrone2019-DET-test-dev"))
    ap.add_argument("--yolo", type=str,
                    default=str(SEMVIDEO_ROOT / "weights" / "yolov8s-visdrone.pt"))
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--snr", type=int, default=20)
    ap.add_argument("--channel", type=str, default="awgn", choices=["awgn", "rayleigh"])
    ap.add_argument("--codec-size", type=int, default=512)
    ap.add_argument("--C", type=int, default=96)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = str(EXPERIMENTS_DIR / f"mAP_{args.label}_{ts}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] out: {out_dir}")
    print(f"[setup] codec: {args.checkpoint}")
    print(f"[setup] YOLO: {args.yolo}")
    print(f"[setup] channel: {args.channel}  SNR: {args.snr}")

    from ultralytics import YOLO
    yolo = YOLO(args.yolo)
    yolo.to(DEVICE)
    print(f"[load] YOLO ready: {args.yolo}")

    net = build_swinjscc(
        Path(args.checkpoint),
        image_dims=(3, args.codec_size, args.codec_size),
        C=args.C, channel=args.channel,
    )

    img_dir = Path(args.visdrone_root) / "images"
    anno_dir = Path(args.visdrone_root) / "annotations"
    img_paths = sorted(img_dir.glob("*.jpg"))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(img_paths), size=min(args.n_frames, len(img_paths)), replace=False)
    idx.sort()
    img_paths = [img_paths[i] for i in idx]
    print(f"[data] {len(img_paths)} VisDrone test-dev images")

    per_path = out_dir / "per_frame.csv"
    summary_path = out_dir / "summary.csv"

    ap_raw_list, ap_rec_list = [], []
    n_gt_total, n_pred_raw_total, n_pred_rec_total = 0, 0, 0

    with open(per_path, "w", newline="") as fp:
        pw = csv.writer(fp)
        pw.writerow(["image", "n_gt", "n_pred_raw", "n_pred_rec",
                     "ap50_raw", "ap50_rec", "delta_ap50"])

        for i, p in enumerate(img_paths):
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            crop_raw, rec_bgr, crop_box = reconstruct(net, bgr, args.snr, args.codec_size)
            gt = load_visdrone_anno(anno_dir / f"{p.stem}.txt", crop_box, args.codec_size)
            if not gt:
                continue
            pred_raw = run_yolo(yolo, crop_raw, conf=args.conf)
            pred_rec = run_yolo(yolo, rec_bgr, conf=args.conf)
            ap_raw = per_image_ap50(pred_raw, gt)
            ap_rec = per_image_ap50(pred_rec, gt)
            if ap_raw is None or ap_rec is None:
                continue
            ap_raw_list.append(ap_raw)
            ap_rec_list.append(ap_rec)
            n_gt_total += len(gt)
            n_pred_raw_total += len(pred_raw)
            n_pred_rec_total += len(pred_rec)
            pw.writerow([p.stem, len(gt), len(pred_raw), len(pred_rec),
                         round(ap_raw, 4), round(ap_rec, 4),
                         round(ap_rec - ap_raw, 4)])
            if (i + 1) % 25 == 0:
                print(f"  [progress] {i+1}/{len(img_paths)} "
                      f"mean ap50_raw={np.mean(ap_raw_list):.3f} "
                      f"ap50_rec={np.mean(ap_rec_list):.3f}")

    n = len(ap_raw_list)
    if n == 0:
        print("[error] no images with GT successfully evaluated")
        return 1
    raw_mean, raw_std = float(np.mean(ap_raw_list)), float(np.std(ap_raw_list))
    rec_mean, rec_std = float(np.mean(ap_rec_list)), float(np.std(ap_rec_list))

    with open(summary_path, "w", newline="") as fp2:
        sw = csv.writer(fp2)
        sw.writerow(["label", "snr", "channel", "n_images", "n_gt_total",
                     "n_pred_raw_total", "n_pred_rec_total",
                     "ap50_raw_mean", "ap50_raw_std",
                     "ap50_rec_mean", "ap50_rec_std", "delta_ap50_mean"])
        sw.writerow([args.label, args.snr, args.channel, n, n_gt_total,
                     n_pred_raw_total, n_pred_rec_total,
                     f"{raw_mean:.5f}", f"{raw_std:.5f}",
                     f"{rec_mean:.5f}", f"{rec_std:.5f}",
                     f"{rec_mean - raw_mean:+.5f}"])

    print(f"\n[summary] n={n} images @ SNR={args.snr} channel={args.channel}")
    print(f"  raw  mAP@0.5: {raw_mean:.4f} ± {raw_std:.4f}  ({n_pred_raw_total} preds, {n_gt_total} GTs)")
    print(f"  rec  mAP@0.5: {rec_mean:.4f} ± {rec_std:.4f}  ({n_pred_rec_total} preds)")
    print(f"  delta:       {rec_mean - raw_mean:+.4f}")
    print(f"[saved] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
