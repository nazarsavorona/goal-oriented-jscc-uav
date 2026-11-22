"""Convert VisDrone-DET annotations to YOLO format + write a dataset yaml, so we can
train an aerial vehicle/person detector (replacing the COCO YOLO-World false positives
on UAV imagery). Run inside semvideo:latest with /data mounted.

The output directory must be writable; the README's docker example mounts /data
read-only, so pass e.g. ``--out /workspace/visdrone_yolo`` there.
"""
import argparse
import glob
import random
import shutil
from pathlib import Path

import cv2

# VisDrone categories 1..10 -> YOLO 0..9
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
         "tricycle", "awning-tricycle", "bus", "motor"]


def main() -> None:
    """Parse the arguments and convert the VisDrone annotations."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("/data/VisDrone2019-DET-train"),
                        help="VisDrone-DET split with images/ and annotations/")
    parser.add_argument("--out", type=Path, default=Path("/data/visdrone_yolo"),
                        help="output directory (must be writable)")
    args = parser.parse_args()
    src, out = args.src, args.out

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    imgs = sorted(glob.glob(str(src / "images" / "*.jpg")))
    print(f"converting {len(imgs)} VisDrone images...", flush=True)
    kept = 0
    for k, ip in enumerate(imgs):
        stem = Path(ip).stem
        ap = src / "annotations" / f"{stem}.txt"
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
        dst = out / "images" / f"{stem}.jpg"
        if not dst.exists():
            try:
                dst.symlink_to(ip)
            except OSError:
                shutil.copy(ip, dst)
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines))
        kept += 1
        if k % 1000 == 0:
            print(f"  ...{k}/{len(imgs)}", flush=True)

    # 90/10 train/val split
    stems = [Path(p).stem for p in glob.glob(str(out / "images" / "*.jpg"))]
    random.seed(0); random.shuffle(stems)
    nval = max(50, len(stems) // 10)
    val, train = stems[:nval], stems[nval:]
    (out / "train.txt").write_text("\n".join(f"./images/{s}.jpg" for s in train))
    (out / "val.txt").write_text("\n".join(f"./images/{s}.jpg" for s in val))

    yaml = (f"path: {out}\ntrain: train.txt\nval: val.txt\nnc: {len(NAMES)}\n"
            f"names: {NAMES}\n")
    (out / "visdrone.yaml").write_text(yaml)
    print(f"converted {kept} images, train={len(train)} val={len(val)} -> {out}/visdrone.yaml")
    print("VISDRONE_CONVERT_DONE")


if __name__ == "__main__":
    main()
