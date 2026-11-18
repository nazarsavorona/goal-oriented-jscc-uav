"""Ground-truth detection on VisDrone2019-DET test-dev for the SwinJSCC-capacity codecs.

Why this exists: the SwinJSCC-capacity codec chain carries a frozen
DINOv2 ViT-B/14 feature term in its training loss, while the paper's reporting metric is a
DINOv2 ViT-S/14 class-token cosine, so the key findings are validated here with an
independent metric and with task performance against ground-truth labels. This script scores
reconstructions against the official VisDrone test-dev annotations, which no model in the paper
was trained on, with a detector that was not used in any training loss (YOLOv8s fine-tuned on
VisDrone train). Alongside it records two representation metrics per image (DINOv2 ViT-S/14 and
CLIP ViT-B/32 class-embedding cosine) and PSNR, so the surrogate can be checked against the task.

Design choices, fixed before the run:
  * every test-dev image (1610) unless --n-frames limits it; centre square crop, 512 px;
  * the same noise / erasure / jammer realisation for every codec within an (image, condition)
    cell (seed derived from the image name and the condition), so codec differences are paired;
  * predictions are stored per image with confidence >= --conf, so AP@0.5 and paired bootstrap
    intervals are computed offline by analyse_visdrone_gt_detection.py without re-running;
  * GT classes 1-10 only; VisDrone 'ignored regions' (0) and 'others' (11) are dropped, and
    detections are not masked by ignored regions (applied identically to every condition).

Conditions are strings: "clean" (raw crop, no codec), "awgn:SNR", "erasure:P:SNR" (zero-fill
erasure of the coded latent, no residual AWGN, as in the paper's erasure table), and
"barrage|pulsed|swept|single_tone|reactive:JNR:SNR" (latent-domain jammers with the paper's
settings), "<jammer>_excise:JNR:SNR" (that jammer with the paper's training-free MAD excision
front-end, k = 3, on the received symbols; barrage_excise:-100:SNR is excision with no jammer), and "rayleigh:SNR" (the codec rebuilt with
SwinJSCC's Rayleigh channel, as in the paper's Rayleigh cells).

Usage (from the repository root, with a GPU):
    python scripts/eval_visdrone_gt_detection.py --out <dir> \
        --codec orig=<SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model> \
        --codec p49=<experiments/p49_p46_awgn_finetune/best.pt> \
        --codec p68=<experiments/p68_p49_erasure_aug_p40/best.pt> \
        --cond awgn:20 awgn:10 ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ORIG_CWD = Path.cwd()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SEMVIDEO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SEMVIDEO_ROOT / "scripts"))
sys.path.insert(0, str(SEMVIDEO_ROOT / "src"))

# Both imports below chdir into third_party/SwinJSCC; every path is resolved before use.
from eval_visdrone_map import load_visdrone_anno, run_yolo  # noqa: E402
from eval_jamming_cross_corpus import JammingPatcher  # noqa: E402
# the six-cell harness's loader: same build recipe as the paper's AWGN and Rayleigh cells,
# and it prints the missing/unexpected key counts of every non-AWGN load
from eval_benchmark_sweep import DINOv2Extractor, CLIPExtractor, build_swinjscc, cossim  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (ORIG_CWD / q).resolve()


def cell_seed(image: str, cond: str) -> int:
    """Deterministic seed shared by every codec in one (image, condition) cell."""
    return int(hashlib.sha1(f"{image}|{cond}".encode()).hexdigest()[:8], 16)


def centre_crop(bgr: np.ndarray, size: int):
    h, w = bgr.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = cv2.resize(bgr[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)
    return crop, (y0, x0, side)


@torch.no_grad()
def reconstruct(net, patcher: JammingPatcher, crop_bgr: np.ndarray, cond: str) -> np.ndarray:
    kind, *vals = cond.split(":")
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    if kind in ("awgn", "rayleigh"):
        # the channel type is fixed when the net is built; main() hands in the matching net
        rec, *_ = net(x, given_SNR=float(vals[0]), given_rate=None)
    elif kind == "erasure":
        p, snr = float(vals[0]), float(vals[1])
        with patcher("erasure", jnr_db=0.0, erasure_p=p):
            rec, *_ = net(x, given_SNR=snr, given_rate=None)
    elif kind.replace("_excise", "") in ("barrage", "reactive", "pulsed", "swept", "single_tone"):
        # for <jammer>_excise main() hands in the patcher built with rx_defense="excise"
        jnr, snr = float(vals[0]), float(vals[1])
        with patcher(kind.replace("_excise", ""), jnr_db=jnr):
            rec, *_ = net(x, given_SNR=snr, given_rate=None)
    else:
        raise ValueError(f"unknown condition {cond}")
    rec = rec.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor((rec * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def _save_atomic(path: Path, write, binary: bool = False) -> None:
    """Write to a temporary file next to ``path`` and rename it over ``path`` once complete."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})) as f:
        write(f)
    os.replace(tmp, path)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--visdrone-root", default=str(REPO_ROOT / "data" / "visdrone" / "VisDrone2019-DET-test-dev"))
    ap.add_argument("--yolo", default=str(SEMVIDEO_ROOT / "weights" / "yolov8s-visdrone.pt"))
    ap.add_argument("--codec", action="append", required=True, help="name=checkpoint (repeatable)")
    ap.add_argument("--cond", nargs="+", required=True)
    ap.add_argument("--n-frames", type=int, default=0, help="0 = every test-dev image")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--C", type=int, default=96)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--clip", action="store_true", help="also record CLIP ViT-B/32 cosine")
    ap.add_argument("--restrict", action="append", default=[],
                    help="name=prefix1,prefix2: run this codec only on conditions starting with a prefix")
    ap.add_argument("--images-dir", default=None,
                    help="score the images of this directory instead of <visdrone-root>/images (needs --no-gt "
                         "when the directory carries no VisDrone annotations, e.g. DJI val)")
    ap.add_argument("--no-gt", action="store_true",
                    help="representation metrics and PSNR only: no detector, no annotations")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run in --out: keep every image whose cells are all stored")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = _abs(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = _abs(args.visdrone_root)
    if args.images_dir:
        idir = _abs(args.images_dir)
        imgs = sorted(list(idir.glob("*.jpg")) + list(idir.glob("*.png")))
    else:
        imgs = sorted((root / "images").glob("*.jpg"))
    if args.n_frames > 0:
        imgs = imgs[: args.n_frames]
    codecs = {}
    for spec in args.codec:
        name, ck = spec.split("=", 1)
        codecs[name] = _abs(ck)
    restrict = {}
    for spec in args.restrict:
        name, prefixes = spec.split("=", 1)
        restrict[name] = tuple(prefixes.split(","))

    def runs(name: str, cond: str) -> bool:
        return name not in restrict or cond.startswith(restrict[name])

    meta = {"args": vars(args), "codecs": {k: str(v) for k, v in codecs.items()},
            "n_images": len(imgs), "device": str(DEVICE), "torch": torch.__version__,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    gt_rows, pred_rows, metric_rows = [], [], []
    start = 0
    if args.resume and (out / "metrics.json").exists():
        old_meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert old_meta["args"]["cond"] == args.cond and old_meta["codecs"] == meta["codecs"],             "resume needs the same conditions and codecs"
        meta["resumed_from"] = old_meta.get("started")
        # rows of cells the current --restrict excludes are dropped, so every kept cell covers
        # the same images
        m_old = [r for r in json.loads((out / "metrics.json").read_text(encoding="utf-8"))
                 if runs(r["codec"], r["cond"])]
        per_img = {}
        for r in m_old:
            per_img[r["idx"]] = per_img.get(r["idx"], 0) + 1
        need = sum(1 for c in args.cond for n in codecs if runs(n, c))
        start = 0
        while per_img.get(start, 0) == need:
            start += 1
        metric_rows = [r for r in m_old if r["idx"] < start]
        pred_rows = [tuple(r) for r in json.loads((out / "preds.json").read_text(encoding="utf-8"))
                     if r[2] < start and (r[0] == "clean" or runs(r[1], r[0]))]
        gt_rows = [tuple(r) for r in np.load(out / "gt.npy").tolist() if r[0] < start]
        print(f"[resume] {start} images complete, continuing from index {start}", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[setup] {len(imgs)} images, codecs {list(codecs)}, conditions {args.cond}", flush=True)

    yolo = None
    if not args.no_gt:
        from ultralytics import YOLO
        yolo = YOLO(str(_abs(args.yolo)))
    dino = DINOv2Extractor()
    clip = CLIPExtractor() if args.clip else None
    nets = {n: build_swinjscc(str(p), image_dims=(3, args.size, args.size), C=args.C, channel="awgn")
            for n, p in codecs.items()}
    patchers = {n: JammingPatcher(net) for n, net in nets.items()}
    # the excision front-end of the paper's reactive-jam table (MAD scale, blank beyond k = 3 sigma)
    ex_patchers = {n: JammingPatcher(net, rx_defense="excise", rx_k=3.0) for n, net in nets.items()}
    ray_nets = {}
    if any(c.startswith("rayleigh:") for c in args.cond):
        ray_nets = {n: build_swinjscc(str(p), image_dims=(3, args.size, args.size), C=args.C,
                                      channel="rayleigh") for n, p in codecs.items()}

    t0 = time.time()
    for i, path in enumerate(imgs):
        if i < start:
            continue
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        crop, box = centre_crop(bgr, args.size)
        if not args.no_gt:
            for b in load_visdrone_anno(root / "annotations" / (path.stem + ".txt"), box, args.size):
                gt_rows.append((i, *b))
        ref_d = dino.features(crop)[0]
        ref_c = clip.features(crop)[0] if clip else None
        if yolo is not None:
            for p in run_yolo(yolo, crop, conf=args.conf):
                pred_rows.append(("clean", "raw", i, *[float(v) for v in p[:5]], int(p[5])))
        for cond in args.cond:
            for name, net in nets.items():
                if not runs(name, cond):
                    continue
                seed = cell_seed(path.stem, cond)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                use = ray_nets[name] if cond.startswith("rayleigh:") else net
                pat = ex_patchers[name] if cond.split(":")[0].endswith("_excise") else patchers[name]
                rec = reconstruct(use, pat, crop, cond)
                if yolo is not None:
                    for p in run_yolo(yolo, rec, conf=args.conf):
                        pred_rows.append((cond, name, i, *[float(v) for v in p[:5]], int(p[5])))
                row = {"image": path.stem, "idx": i, "cond": cond, "codec": name,
                       "dino_cls": cossim(ref_d, dino.features(rec)[0]), "psnr": psnr(crop, rec)}
                if clip:
                    row["clip_cls"] = cossim(ref_c, clip.features(rec)[0])
                metric_rows.append(row)
        if (i + 1) % 20 == 0 or i + 1 == len(imgs):
            el = time.time() - t0
            done = i + 1 - start
            print(f"[{i + 1}/{len(imgs)}] {el / 60:.1f} min elapsed, "
                  f"~{el / done * (len(imgs) - i - 1) / 60:.1f} min left", flush=True)
            # atomic: a crash mid-write (seen on an unstable host) must not destroy the last good copy
            _save_atomic(out / "gt.npy", lambda f: np.save(f, np.array(gt_rows, dtype=np.float64)), binary=True)
            _save_atomic(out / "preds.json", lambda f: json.dump(pred_rows, f))
            _save_atomic(out / "metrics.json", lambda f: json.dump(metric_rows, f))
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[done]", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
