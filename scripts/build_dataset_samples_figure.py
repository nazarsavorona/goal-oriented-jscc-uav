"""Build the paper's dataset-sample montage from the DJI validation split.

The figure is generated with ``pdf.fonttype = 42`` (no Type 3 font subset), deterministic
frame selection, and the document's own serif so the panel labels match the body type.

Frame selection is deterministic and documented: for each daytime mission (001, 004, 005)
one wide-camera and one zoom-camera validation frame, taken at the median position of that
(mission, camera) group's sorted file list, so the choice cannot be tuned to flatter
frames. All six come from the validation split of the DJI Mavic 3T corpus; the
operator-supplied footage (not public) is deliberately not touched by this figure.

The DJI corpus is not public (see the README), so the default ``--frames-dir`` exists only
where that corpus has been obtained from the authors.

Usage:
    python scripts/build_dataset_samples_figure.py
    python scripts/build_dataset_samples_figure.py --frames-dir ... --out ... --font-serif
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, no Type 3 in the submission PDF
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Palatino Linotype"
matplotlib.rcParams["mathtext.it"] = "Palatino Linotype:italic"
matplotlib.rcParams["mathtext.bf"] = "Palatino Linotype:bold"
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES = REPO_ROOT / "data" / "dji" / "frames_512_v1" / "val"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "figures" / "dataset_samples.pdf"

STEM_RE = re.compile(r"DJI_\d+_(?P<mission>\d{3})_(?P<cam>[WZ])_frame\d+")
MISSIONS = ("001", "004", "005")
CAMS = (("W", "wide"), ("Z", "zoom"))
# height / width of the figure as printed in the paper (511.2 x 355.457 pt), kept when the type size changes
PRINTED_ASPECT = 355.456787109375 / 511.2


def _bbox_at_aspect(fig, aspect: float, pad: float = 0.1):
    """Tight bounding box (inches), grown symmetrically to height / width = ``aspect``.

    The manuscript scales the figure to the text width, so its printed height is the text width
    times this ratio; pinning the ratio lets type sizes change without moving the page layout.
    """
    from matplotlib.transforms import Bbox

    bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad)
    if bb.height / bb.width < aspect:
        d = (bb.width * aspect - bb.height) / 2
        return Bbox.from_extents(bb.x0, bb.y0 - d, bb.x1, bb.y1 + d)
    d = (bb.height / aspect - bb.width) / 2
    return Bbox.from_extents(bb.x0 - d, bb.y0, bb.x1 + d, bb.y1)


def pick_frames(frames_dir: Path) -> list[tuple[str, str, Path]]:
    """One median-position frame per (mission, camera), in a fixed reading order."""
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for p in sorted(frames_dir.glob("*.jpg")):
        m = STEM_RE.match(p.stem)
        if m:
            groups[(m.group("mission"), m.group("cam"))].append(p)

    picks: list[tuple[str, str, Path]] = []
    for mission in MISSIONS:
        for cam, cam_name in CAMS:
            paths = groups.get((mission, cam), [])
            if not paths:
                raise SystemExit(f"no frames for mission {mission}, camera {cam}")
            picks.append((mission, cam_name, paths[len(paths) // 2]))
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--font-serif",
        action="store_true",
        help="Set the panel labels in Palatino to match the MDPI body type instead of "
        "matplotlib's DejaVu Sans.",
    )
    args = ap.parse_args()

    if args.font_serif:
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = [
            "Palatino Linotype", "URW Palladio L", "TeX Gyre Pagella", "Palatino", "serif",
        ]
        matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"

    picks = pick_frames(args.frames_dir)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for ax, (mission, cam_name, path) in zip(axes.T.ravel(), picks, strict=True):
        ax.imshow(mpimg.imread(path))
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_edgecolor("0.45")
            side.set_linewidth(0.6)
        # 11 pt drawn = 8.5 pt printed: the figure is set at 0.77 of its drawn width
        ax.set_title(f"mission {mission}, {cam_name} camera", fontsize=11, pad=4)

    # No numbers in the title: the vehicle-size statistic belongs to the caption, which
    # states it once (median min-side in the source stream) instead of inviting a second,
    # differently-defined figure of the same quantity.
    # No suptitle: the LaTeX caption states the corpus and the codec input once, and a
    # figure heading that repeats it is one more thing to keep in sync.
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    box = _bbox_at_aspect(fig, PRINTED_ASPECT)
    fig.savefig(args.out, bbox_inches=box, dpi=300)
    fig.savefig(args.out.with_suffix(".png"), bbox_inches=box, dpi=200)
    print(f"[dataset-samples] wrote {args.out} and {args.out.with_suffix('.png')}")
    for mission, cam_name, path in picks:
        print(f"    mission {mission} {cam_name:>4}: {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
