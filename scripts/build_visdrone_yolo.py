"""Convert VisDrone-DET annotations to YOLO format + write a dataset yaml, so we can
train an aerial vehicle/person detector (replacing the COCO YOLO-World false positives
on UAV imagery). Run inside semvideo:latest with /data mounted.
"""
import glob
import random
from pathlib import Path

import cv2

SRC = Path("/data/VisDrone2019-DET-train")
OUT = Path("/data/visdrone_yolo")
(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "labels").mkdir(parents=True, exist_ok=True)

# VisDrone categories 1..10 -> YOLO 0..9
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
         "tricycle", "awning-tricycle", "bus", "motor"]

imgs = sorted(glob.glob(str(SRC / "images" / "*.jpg")))
print(f"converting {len(imgs)} VisDrone images...", flush=True)
kept = 0
for k, ip in enumerate(imgs):
    stem = Path(ip).stem
    ap = SRC / "annotations" / f"{stem}.txt"
    if not ap.exists():
        continue
    img = cv2.imread(ip)
    if img is None:
        continue
    H, W = img.shape[:2]
    lines = []
    for line in open(ap):
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        try:
            x, y, w, h = map(float, p[:4]); cat = int(p[5])
        except ValueError:
            continue
        if cat < 1 or cat > 10 or w <= 0 or h <= 0:
            continue
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        lines.append(f"{cat - 1} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
    if not lines:
        continue
    # symlink image + write label (parallel images/ labels/ dirs)
    dst = OUT / "images" / f"{stem}.jpg"
    if not dst.exists():
        try:
            dst.symlink_to(ip)
        except OSError:
            import shutil; shutil.copy(ip, dst)
    (OUT / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    kept += 1
    if k % 1000 == 0:
        print(f"  ...{k}/{len(imgs)}", flush=True)

# 90/10 train/val split
stems = [Path(p).stem for p in glob.glob(str(OUT / "images" / "*.jpg"))]
random.seed(0); random.shuffle(stems)
nval = max(50, len(stems) // 10)
val, train = stems[:nval], stems[nval:]
(OUT / "train.txt").write_text("\n".join(f"./images/{s}.jpg" for s in train))
(OUT / "val.txt").write_text("\n".join(f"./images/{s}.jpg" for s in val))

yaml = (f"path: {OUT}\ntrain: train.txt\nval: val.txt\nnc: {len(NAMES)}\n"
        f"names: {NAMES}\n")
(OUT / "visdrone.yaml").write_text(yaml)
print(f"converted {kept} images, train={len(train)} val={len(val)} -> {OUT}/visdrone.yaml")
print("VISDRONE_CONVERT_DONE")
