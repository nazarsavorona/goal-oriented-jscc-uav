"""Score the stored predictions of eval_visdrone_gt_detection.py against VisDrone ground truth.

AP@0.5 per class with all-point interpolation (area under the monotone precision envelope),
matching per image in descending confidence to the highest-IoU unmatched ground-truth box of the
same class (IoU >= 0.5). Macro mAP@0.5 averages the classes that have ground truth in the image
set. Intervals are image-level bootstrap percentiles (fixed seed); a codec difference is paired:
both codecs are scored on the same resampled image multiset in every replicate.

Usage:  python analyse_visdrone_gt_detection.py <run_dir> [--boot 2000] [--seed 0]
Writes <run_dir>/summary.json and prints a table.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = list(range(1, 11))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """preds (P,5)=[x1,y1,x2,y2,conf] of one class in one image, gts (G,4). Returns tp flags (P,)."""
    tp = np.zeros(len(preds), dtype=np.int8)
    if len(preds) == 0 or len(gts) == 0:
        return tp
    order = np.argsort(-preds[:, 4], kind="stable")
    iou = iou_matrix(preds[order, :4], gts)
    used = np.zeros(len(gts), dtype=bool)
    for r, i in enumerate(order):
        cand = np.where((iou[r] >= 0.5) & ~used)[0]
        if len(cand):
            j = cand[np.argmax(iou[r, cand])]
            used[j] = True
            tp[i] = 1
    return tp


def ap_from(conf, tp, img, weights, npos):
    """All-point AP with per-image multiplicity weights; conf/tp/img presorted by conf desc."""
    if npos <= 0:
        return np.nan
    w = weights[img]
    tpc = np.cumsum(tp * w)
    fpc = np.cumsum((1 - tp) * w)
    keep = w > 0
    if not keep.any():
        return 0.0
    tpc, fpc = tpc[keep], fpc[keep]
    rec = tpc / npos
    prec = tpc / np.maximum(tpc + fpc, 1e-12)
    mpre = np.maximum.accumulate(prec[::-1])[::-1]
    drec = np.diff(np.concatenate([[0.0], rec]))
    return float(np.sum(drec * mpre))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run_dir)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    n_img = meta["n_images"]
    gt = np.load(run / "gt.npy")  # rows: idx, x1, y1, x2, y2, cls
    preds = json.loads((run / "preds.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    n_img = int(max(n_img if gt.size == 0 else 0, gt[:, 0].max() + 1 if gt.size else 0,
                    max((p[2] for p in preds), default=-1) + 1))

    gt_by = defaultdict(list)
    gt_count = np.zeros((n_img, 11), dtype=np.int64)
    for r in gt:
        i, c = int(r[0]), int(r[5])
        gt_by[(i, c)].append(r[1:5])
        gt_count[i, c] += 1
    pr_by = defaultdict(lambda: defaultdict(list))
    for cond, codec, i, x1, y1, x2, y2, conf, c in preds:
        pr_by[(cond, codec)][(int(i), int(c))].append((x1, y1, x2, y2, conf))

    # per (cond, codec, class): arrays sorted by confidence
    scored = {}
    for key, cells in pr_by.items():
        per_cls = {}
        for c in CLASSES:
            confs, tps, imgs = [], [], []
            for i in range(n_img):
                p = np.array(cells.get((i, c), []), dtype=np.float64).reshape(-1, 5)
                g = np.array(gt_by.get((i, c), []), dtype=np.float64).reshape(-1, 4)
                if len(p):
                    t = match(p, g)
                    confs.append(p[:, 4]); tps.append(t); imgs.append(np.full(len(p), i))
            if confs:
                conf = np.concatenate(confs); tp = np.concatenate(tps); img = np.concatenate(imgs)
                o = np.argsort(-conf, kind="stable")
                per_cls[c] = (conf[o], tp[o].astype(np.float64), img[o])
            else:
                per_cls[c] = (np.zeros(0), np.zeros(0), np.zeros(0, dtype=int))
        scored[key] = per_cls

    def macro(key, weights):
        aps = []
        for c in CLASSES:
            npos = float(np.sum(weights * gt_count[:, c]))
            if npos <= 0:
                continue
            conf, tp, img = scored[key][c]
            aps.append(ap_from(conf, tp, img, weights, npos) if len(conf) else 0.0)
        return float(np.mean(aps)) if aps else np.nan

    ones = np.ones(n_img)
    rng = np.random.default_rng(args.seed)
    boots = [np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(np.float64)
             for _ in range(args.boot)]
    keys = sorted(scored)
    point = {f"{k[0]}|{k[1]}": macro(k, ones) for k in keys}
    dist = {f"{k[0]}|{k[1]}": np.array([macro(k, w) for w in boots]) for k in keys}

    out = {"n_images": n_img, "boot": args.boot, "seed": args.seed, "conf_threshold": meta["args"]["conf"],
           "interpolation": "all-point", "iou": 0.5, "classes_with_gt": int((gt_count[:, 1:].sum(0) > 0).sum()),
           "macro_map50": {k: {"point": point[k], "ci95": [float(np.percentile(dist[k], 2.5)), float(np.percentile(dist[k], 97.5))]} for k in point},
           "paired": {}}
    conds = sorted({k[0] for k in keys if k[0] != "clean"})
    codecs = sorted({k[1] for k in keys if k[0] != "clean"})
    for cond in conds:
        for a, b in itertools.combinations(codecs, 2):
            ka, kb = f"{cond}|{a}", f"{cond}|{b}"
            if ka in dist and kb in dist:
                d = dist[kb] - dist[ka]
                out["paired"][f"{cond}|{b}-{a}"] = {"point": point[kb] - point[ka],
                                                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        for c in codecs:
            kc = f"{cond}|{c}"
            if kc in dist and "clean|raw" in dist:
                d = dist[kc] - dist["clean|raw"]
                out["paired"][f"{cond}|{c}-raw"] = {"point": point[kc] - point["clean|raw"],
                                                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    # surrogate check: per-(cond, codec) means of the representation metrics next to mAP
    agg = defaultdict(lambda: defaultdict(list))
    per_img = defaultdict(lambda: defaultdict(dict))  # key -> field -> image idx -> value
    for m in metrics:
        k = f"{m['cond']}|{m['codec']}"
        for f in ("dino_cls", "clip_cls", "psnr"):
            if f in m:
                agg[k][f].append(m[f])
                per_img[k][f][int(m["idx"])] = m[f]
    out["means"] = {k: {f: float(np.mean(v)) for f, v in fs.items()} for k, fs in agg.items()}

    # paired per-image differences of the representation metrics, resampled with the same
    # image multisets as the mAP intervals (one bootstrap, every quantity)
    out["paired_metrics"] = {}
    for cond in conds:
        for a, b in itertools.combinations(codecs, 2):
            ka, kb = f"{cond}|{a}", f"{cond}|{b}"
            for f in ("dino_cls", "clip_cls", "psnr"):
                va, vb = per_img[ka].get(f, {}), per_img[kb].get(f, {})
                common = sorted(set(va) & set(vb))
                if not common:
                    continue
                d = np.zeros(n_img)
                present = np.zeros(n_img)
                for i in common:
                    d[i] = vb[i] - va[i]
                    present[i] = 1.0
                pt = float(d[present > 0].mean())
                bs = np.array([float(np.sum(w * d) / max(np.sum(w * present), 1e-12)) for w in boots])
                out["paired_metrics"][f"{cond}|{f}|{b}-{a}"] = {
                    "n": len(common), "point": pt,
                    "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    # does the surrogate rank the (condition, codec) cells as the task measure does?
    cells = [k for k in point if not k.startswith("clean|") and k in out["means"]]
    if len(cells) >= 3:
        from scipy.stats import spearmanr
        out["cell_rank_agreement"] = {}
        maps = [point[k] for k in cells]
        for f in ("dino_cls", "clip_cls", "psnr"):
            xs = [out["means"][k].get(f) for k in cells]
            if None in xs:
                continue
            rho, pv = spearmanr(xs, maps)
            out["cell_rank_agreement"][f] = {"n_cells": len(cells), "spearman": float(rho), "p": float(pv)}
    (run / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"n_images={n_img}  classes_with_gt={out['classes_with_gt']}  boot={args.boot}")
    for k in sorted(out["macro_map50"]):
        v = out["macro_map50"][k]; mm = out["means"].get(k, {})
        print(f"{k:28s} mAP50 {v['point']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]  "
              + "  ".join(f"{f} {x:.4f}" for f, x in mm.items()))
    for k, v in out["paired"].items():
        print(f"  {k:34s} {v['point']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    for k, v in out["paired_metrics"].items():
        print(f"  {k:44s} n={v['n']} {v['point']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    for f, v in out.get("cell_rank_agreement", {}).items():
        print(f"  cells ranked by {f} vs mAP50: Spearman {v['spearman']:+.3f} (p={v['p']:.2g}, {v['n_cells']} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
