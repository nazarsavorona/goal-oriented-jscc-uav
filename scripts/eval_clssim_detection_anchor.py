"""Anchor the cls-sim gate to a task outcome: detection agreement per frame.

The 0.85 "usable" gate is currently justified by a false-alarm argument and a
threshold-insensitivity sweep, but not by a downstream task outcome: a global
cosine may be dominated by background, and cross-checks confirm ordering but not
that the metric tracks the operational goal.

This script produces the missing curve. For each frame it computes, on the SAME
frame and the SAME reconstruction:
  * cls-sim (DINOv2 ViT-S/14 CLS cosine, the paper's metric), and
  * detection agreement between the detector's output on the original and on the
    reconstruction -- an F1 over greedily IoU-matched boxes, i.e. how much of the
    operator-relevant content survived transmission.
Sweeping SNR (and optionally a jammer) spreads frames across the cls-sim range,
so the two can be correlated and the gate located in task terms.

Detection agreement is measured detector-vs-detector, NOT against ground truth:
the question is whether transmission preserved what the detector would have seen,
which is the deployed quantity and needs no labels.

Usage:
    python eval_clssim_detection_anchor.py --ckpt CKPT --detector DET.pt \
        --val-dir DIR --n 200 --snrs 0 5 10 20 --codec-size 512 --out OUT.csv
"""

from __future__ import annotations

import argparse
import contextlib
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

DINO_HF = "facebook/dinov2-small"
DINO_RES = 518
IOU_MATCH = 0.5
CONF = 0.25


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy box sets."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = (a[:, i][:, None] for i in range(4))
    bx1, by1, bx2, by2 = (b[:, i][None, :] for i in range(4))
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def agreement(ref: np.ndarray, hyp: np.ndarray) -> tuple[float, int, int, int]:
    """Greedy IoU matching -> (F1, matched, n_ref, n_hyp)."""
    n_ref, n_hyp = len(ref), len(hyp)
    if n_ref == 0 and n_hyp == 0:
        return 1.0, 0, 0, 0  # nothing to see either way: perfect agreement
    if n_ref == 0 or n_hyp == 0:
        return 0.0, 0, n_ref, n_hyp
    iou = iou_matrix(ref, hyp)
    matched = 0
    used_r, used_h = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for r, h in order:
        if iou[r, h] < IOU_MATCH:
            break
        if r in used_r or h in used_h:
            continue
        used_r.add(int(r)); used_h.add(int(h)); matched += 1
    precision = matched / n_hyp
    recall = matched / n_ref
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return f1, matched, n_ref, n_hyp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="SwinJSCC checkpoint")
    ap.add_argument("--detector", required=True, help="YOLO .pt trained on this domain")
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10, 20])
    ap.add_argument("--codec-size", type=int, default=512)
    ap.add_argument("--C", type=int, default=96)
    ap.add_argument("--jammer", default=None,
                    help="optional jammer mode (barrage/pulsed/single_tone/swept/reactive); "
                         "when set, --snrs is held fixed and --jnrs is swept instead")
    ap.add_argument("--jnrs", type=float, nargs="+", default=None,
                    help="JNR grid, used only with --jammer")
    ap.add_argument("--erasures", type=float, nargs="+", default=None,
                    help="symbol-erasure probabilities to sweep at a fixed --snrs[0]; "
                         "gives the erasure result a downstream-task anchor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=CONF,
                    help="Detector confidence threshold.")
    ap.add_argument("--tile", type=int, default=1,
                    help="Tiled inference: split each frame into TILE x TILE tiles and run "
                         "the detector on each at native size. Aerial frames put vehicles at "
                         "~19 px across, well under what a COCO detector resolves in one "
                         "512 px pass, so TILE>1 is what makes the task measurable at all.")
    args = ap.parse_args()
    if args.jammer and not args.jnrs:
        raise SystemExit("--jammer requires --jnrs")
    if args.tile < 1:
        raise SystemExit("--tile must be >= 1")

    from ultralytics import RTDETR, YOLO

    sys.path.insert(0, "scripts")
    from eval_swinjscc_dinov2 import build_swinjscc  # proven SwinJSCC loader

    # RT-DETR is a transformer detector outside the YOLO family; using it as a third
    # judge tests whether the cls-sim / detector-agreement link depends on that family.
    det = (RTDETR if 'rtdetr' in Path(args.detector).name.lower() else YOLO)(args.detector)
    dino = AutoModel.from_pretrained(DINO_HF).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    @torch.no_grad()
    def cls_token(x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x.clamp(0, 1), (DINO_RES, DINO_RES),
                          mode="bilinear", align_corners=False)
        return dino(pixel_values=(x - mean) / std).last_hidden_state[:, 0]

    def _predict(img: np.ndarray) -> np.ndarray:
        res = det.predict(img, conf=args.conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return res.boxes.xyxy.cpu().numpy().astype(np.float32)

    def boxes_of(bgr: np.ndarray) -> np.ndarray:
        if args.tile == 1:
            return _predict(bgr)
        # Tiled pass: each tile is upsampled back to the full frame size so small
        # objects reach a scale the detector was trained on; boxes are mapped back
        # to full-frame coordinates. Overlapping duplicates are not merged because
        # the tiles partition the frame without overlap.
        h, w = bgr.shape[:2]
        th, tw = h // args.tile, w // args.tile
        out = []
        for ty in range(args.tile):
            for tx in range(args.tile):
                y0, x0 = ty * th, tx * tw
                y1 = h if ty == args.tile - 1 else y0 + th
                x1 = w if tx == args.tile - 1 else x0 + tw
                tile = bgr[y0:y1, x0:x1]
                up = cv2.resize(tile, (w, h), interpolation=cv2.INTER_CUBIC)
                b = _predict(up)
                if len(b):
                    sx, sy = (x1 - x0) / w, (y1 - y0) / h
                    b = b * np.array([sx, sy, sx, sy], dtype=np.float32)
                    b += np.array([x0, y0, x0, y0], dtype=np.float32)
                    out.append(b)
        return np.concatenate(out, 0) if out else np.zeros((0, 4), dtype=np.float32)

    paths = sorted(glob.glob(args.val_dir + "/*.jpg") + glob.glob(args.val_dir + "/*.png"))
    if not paths:
        raise SystemExit(f"no frames under {args.val_dir}")
    idx = np.linspace(0, len(paths) - 1, min(args.n, len(paths)), dtype=int)
    paths = [paths[int(i)] for i in idx]
    print(f"[setup] {len(paths)} frames x {len(args.snrs)} SNR points")

    # The patcher carries both the jammers and the erasure channel, so the import must not
    # sit inside the jammer branch - an --erasures run needs it too.
    if args.jammer or args.erasures:
        from eval_jamming_cross_corpus import JammingPatcher

    # When a jammer is requested, sweep JNR at the first --snrs value instead.
    if args.jammer:
        sweep = [("jnr", v) for v in args.jnrs]
        fixed_snr = args.snrs[0]
        print(f"[setup] jammer={args.jammer}, JNR grid {args.jnrs} at fixed SNR={fixed_snr}")
    elif args.erasures:
        fixed_snr = args.snrs[0]
        sweep = [("erasure", v) for v in args.erasures]
        print(f"[setup] erasure sweep {args.erasures} at fixed SNR={fixed_snr}")
    else:
        sweep = [("snr", v) for v in args.snrs]
        fixed_snr = None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "snr", "jammer", "jnr", "cls_sim", "det_f1",
                    "matched", "n_ref", "n_hyp"])
        for kind, value in sweep:
            snr = fixed_snr if kind in ("jnr", "erasure") else value
            jnr = value if kind in ("jnr", "erasure") else ""
            net = build_swinjscc(args.ckpt, image_dims=(3, args.codec_size, args.codec_size),
                                 C=args.C)
            patcher = JammingPatcher(net) if (args.jammer or kind == "erasure") else None
            if patcher and kind == "erasure":
                ctx = patcher(jammer_type="erasure", jnr_db=0.0, erasure_p=value)
            elif patcher:
                ctx = patcher(jammer_type=args.jammer, jnr_db=jnr)
            else:
                ctx = contextlib.nullcontext()
            rows = []
            with torch.no_grad(), ctx:
                for p in paths:
                    bgr = cv2.imread(p)
                    if bgr is None:
                        continue
                    side = min(bgr.shape[:2])
                    y0, x0 = (bgr.shape[0] - side) // 2, (bgr.shape[1] - side) // 2
                    crop = cv2.resize(bgr[y0:y0 + side, x0:x0 + side],
                                      (args.codec_size, args.codec_size),
                                      interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
                    rec, *_ = net(t, given_SNR=snr, given_rate=None)
                    rec = rec.clamp(0, 1)
                    sim = F.cosine_similarity(cls_token(t), cls_token(rec)).item()
                    rec_bgr = cv2.cvtColor((rec.squeeze(0).permute(1, 2, 0).cpu().numpy()
                                            * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    f1, matched, n_ref, n_hyp = agreement(boxes_of(crop), boxes_of(rec_bgr))
                    w.writerow([Path(p).stem, snr, args.jammer or (
                        "erasure" if kind == "erasure" else ""), jnr,
                                f"{sim:.6f}", f"{f1:.6f}", matched, n_ref, n_hyp])
                    rows.append((sim, f1))
            arr = np.array(rows)
            tag = f"jnr={jnr:5.1f}" if kind == "jnr" else f"snr={snr:5.1f}"
            print(f"[{tag}] cls_mean={arr[:, 0].mean():.4f} "
                  f"det_f1_mean={arr[:, 1].mean():.4f} n={len(arr)}")
            del net
            torch.cuda.empty_cache()
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
