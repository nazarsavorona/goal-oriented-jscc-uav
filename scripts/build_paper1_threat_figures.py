"""Build the threat-model figures of the paper: erasure curves and the jamming heatmap.

Three figures are defined here: ``erasure_curves`` and ``jamming_heatmap``, which the
paper prints, and ``reactive_jam``, an earlier figure that is no longer printed and is
kept for reference. All are written with ``pdf.fonttype = 42`` so that no Type 3 font
enters the PDF.

Where a series is read from an evaluation record, the record is opened at import time
and the plotted values are computed from it; nothing is interpolated or re-derived by
hand. The records are the per-run summaries written by ``eval_erasure_cross_corpus.py``
and ``eval_visdrone_gt_detection.py`` and the paired test-dev summary. They are not part
of this repository: they belong to the archived review record (see the README), and
this script expects them under ``records/`` in the repository root, or under the
directory named by the ``SEMVIDEO_RECORDS_DIR`` environment variable::

    records/erasure_figure_source/dji_twostage_summary.csv
    records/erasure_figure_source/dji_erasureaug_p40_summary.csv
    records/gt_detection/{run_full,run_erasure0}/metrics.json
    records/heldout_testdev/testdev_summary.json

The remaining series are constants carried from the measurements named in the comment
above each of them.

Usage:
    python scripts/build_paper1_threat_figures.py
    python scripts/build_paper1_threat_figures.py --only erasure_curves --out-dir /tmp/figs
"""

from __future__ import annotations

import argparse
import csv
import io
import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, no Type 3 in the paper PDF
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "figures"
# Evaluation records the figures are read from (see the module docstring).
RECORDS_DIR = Path(os.environ.get("SEMVIDEO_RECORDS_DIR", REPO_ROOT / "records"))

# One style for every figure of the paper.
RC = {
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Palatino", "URW Palladio L", "DejaVu Serif"],
"mathtext.fontset": "custom",
"mathtext.rm": "Palatino Linotype",
"mathtext.it": "Palatino Linotype:italic",
"mathtext.bf": "Palatino Linotype:bold",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Corpus panel titles. The earlier VisDrone panels used a split derived from the VisDrone
# train distribution and are titled accordingly; the panels that use the official
# test-dev images say so. (The evaluation harness writes "VisDrone_testdev" in its bench
# column for both, so the title, not that column, names the split.)
DJI_TITLE = "DJI val (in-domain)"
VD_TITLE = "VisDrone held-out split (train-derived)"
VD_TITLE_XC = "VisDrone held-out split (train-derived, cross-corpus)"
VD_TITLE_TD = "VisDrone test-dev (held out, n = 1610)"

PASS_GATE = 0.85  # cls-sim reference level used throughout the paper

# --------------------------------------------------------------------------------------
# Figure: reactive_jam.pdf  (an earlier figure, no longer printed in the paper)
# --------------------------------------------------------------------------------------
# Energy-targeted reactive (feature-domain DRFM-proxy) jamming, AWGN SNR = 10 dB, n = 100,
# JNR in {0, 5, 10, 15} dB, harness eval_jamming_cross_corpus.py.
REACTIVE_JNR = np.array([0, 5, 10, 15])

# Erasure-augmented codec with no reactive exposure (checkpoint tag P68), n = 100 per cell:
# DJI 0.976 / 0.886 / 0.637 / 0.561, VisDrone (train-derived split) 0.977 / 0.932 / 0.807 / 0.747.
DJI_REACTIVE_BASE = np.array([0.976, 0.886, 0.637, 0.561])
VD_REACTIVE_BASE = np.array([0.977, 0.932, 0.807, 0.747])

# The same codec after an 8-epoch stage with reactive-jam augmentation (checkpoint tag P80):
# DJI 0.984 / 0.971 / 0.938 / 0.888, VisDrone (train-derived split) 0.974 / 0.950 / 0.882 / 0.822.
DJI_REACTIVE_AUG = np.array([0.984, 0.971, 0.938, 0.888])
VD_REACTIVE_AUG = np.array([0.974, 0.950, 0.882, 0.822])

# --------------------------------------------------------------------------------------
# Figure: erasure_curves.pdf  (cited as fig:erasure-curves)
# --------------------------------------------------------------------------------------
# i.i.d. Bernoulli symbol erasure on the post-channel latent, nominal AWGN SNR = 10 dB.
# The DJI panel is READ from the summaries of the run that produced it rather than
# restated here (full validation set, n = 375 per cell).
_ERASURE_SRC = RECORDS_DIR / "erasure_figure_source"


def _erasure_curve(fname):
    """(p grid, cls-sim means) for one arm, straight from its summary.csv."""
    path = _ERASURE_SRC / fname
    with io.open(path, newline="", encoding="utf-8") as fp:
        rows = sorted(csv.DictReader(fp), key=lambda r: float(r["p"]))
    return (np.array([float(r["p"]) for r in rows]),
            np.array([round(float(r["cls_mean"]), 3) for r in rows]))


_p_two, DJI_ERASURE_TWOSTAGE = _erasure_curve("dji_twostage_summary.csv")
_p_aug, DJI_ERASURE_AUG = _erasure_curve("dji_erasureaug_p40_summary.csv")
assert np.array_equal(_p_two, _p_aug), "the two DJI arms were measured on different grids"
ERASURE_P = _p_two

# VisDrone panel: the official VisDrone2019-DET test-dev images, n = 1610, scored by the
# ground-truth detection run (eval_visdrone_gt_detection.py, pure zero-fill erasure at nominal
# SNR 10 dB, one seed per image shared by every codec). An earlier version of this panel used a
# split cut from VisDrone2019-DET-train that the two-stage codec had seen in training. The
# published SwinJSCC weights are added because the two-stage codec falls below them.
_GT_SRC = RECORDS_DIR / "gt_detection"


def _testdev_erasure(codec):
    """(p grid, per-image mean cls-sim) of one codec over the erasure cells of the test-dev run."""
    import json
    by = {}
    for run in ("run_full", "run_erasure0"):
        f = _GT_SRC / run / "metrics.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            if r["codec"] == codec and r["cond"].startswith("erasure:"):
                by.setdefault(float(r["cond"].split(":")[1]), []).append(r["dino_cls"])
    ps = sorted(by)
    ns = {len(v) for v in by.values()}
    assert ns == {1610}, f"test-dev erasure cells of {codec} are not all n = 1610: {ns}"
    return np.array(ps), np.array([round(sum(by[p]) / len(by[p]), 3) for p in ps])


VD_ERASURE_P, VD_ERASURE_TWOSTAGE = _testdev_erasure("p49")
_vd_p_aug, VD_ERASURE_AUG = _testdev_erasure("p68")
_vd_p_orig, VD_ERASURE_ORIG = _testdev_erasure("orig")
assert np.array_equal(VD_ERASURE_P, _vd_p_aug) and np.array_equal(VD_ERASURE_P, _vd_p_orig),     "test-dev arms measured on different erasure grids"

# (The 0.94 guide line of the original erasure figure is no longer drawn; see build_erasure_curves.)

# --------------------------------------------------------------------------------------
# Figure: jamming_heatmap.pdf  (cited as fig:jamming-heatmap)
# --------------------------------------------------------------------------------------
# 24-cell cross-corpus parametric jamming sweep at AWGN SNR = 20 dB: cls-sim improvement
# (Delta) of the two-stage cross-corpus codec over a zero-shot DIV2K-trained SwinJSCC.
# 3 jammer types x 4 JNR levels x 2 corpora.
JAMMERS = ["Barrage", "Pulsed", "Single-tone"]
JNR_LABELS = ["JNR=0", "JNR=5", "JNR=10", "JNR=15"]

# DJI half: paired per-frame evaluation, both arms in one invocation on one frame list,
# n = 200 per cell. All cells are separated and survive Holm-Bonferroni correction.
DJI_JAM_DELTA = np.array(
    [
        [0.0207, 0.0200, 0.0229, 0.0344],  # barrage,     JNR 0 / 5 / 10 / 15
        [0.0211, 0.0203, 0.0249, 0.0358],  # pulsed
        [0.0207, 0.0207, 0.0218, 0.0350],  # single-tone
    ]
)
# VisDrone half: scored on the official test-dev images, n = 1610 per cell, read from the
# paired test-dev summary (family24: paired per-frame bootstrap, cells required to be
# separated from zero).
def _testdev_family():
    import json
    f = RECORDS_DIR / "heldout_testdev" / "testdev_summary.json"
    fam = json.loads(f.read_text(encoding="utf-8"))["family24"]
    cells = {(c["jammer"], c["jnr"]): c for c in fam if c["corpus"] == "VisDrone_testdev"}
    assert len(cells) == 12 and all(c["n"] == 1610 and c["excludes_zero"] for c in cells.values())
    return np.array([[round(cells[(j, jnr)]["delta"], 4) for jnr in (0.0, 5.0, 10.0, 15.0)]
                     for j in ("barrage", "pulsed", "single_tone")])


VD_JAM_DELTA = _testdev_family()
JAM_VMIN, JAM_VMAX = 0.0, 0.04  # colour scale stated in the caption


def build_reactive_jam(figsize: tuple[float, float]) -> plt.Figure:
    """Draw the reactive-jam collapse-and-rescue figure.

    Args:
        figsize: Figure size in inches. Two panels side by side, each roughly square.

    Returns:
        The finished figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, title, base, aug in [
        (axes[0], DJI_TITLE, DJI_REACTIVE_BASE, DJI_REACTIVE_AUG),
        (axes[1], VD_TITLE, VD_REACTIVE_BASE, VD_REACTIVE_AUG),
    ]:
        ax.plot(
            REACTIVE_JNR,
            base,
            "o-",
            color="#cc3333",
            linewidth=1.4,
            markersize=6,
            label="Erasure-aug baseline (no reactive exposure)",
        )
        ax.plot(
            REACTIVE_JNR,
            aug,
            "^-",
            color="#226699",
            linewidth=2.2,
            markersize=7,
            label="Reactive-jam-augmented model",
        )
        ax.axhline(PASS_GATE, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.text(
            15,
            PASS_GATE + 0.01,
            r"PASS $\geq 0.85$",
            va="bottom",
            ha="right",
            fontsize=8,
            color="gray",
        )
        ax.set_xlabel("JNR [dB]")
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_ylim([0.5, 1.02])
        ax.set_xticks(REACTIVE_JNR)
    axes[0].set_ylabel("DINOv2 cls-similarity")
    axes[0].legend(loc="lower left", framealpha=0.95, fontsize=7.5)
    fig.tight_layout()
    return fig


def build_erasure_curves(figsize: tuple[float, float]) -> plt.Figure:
    """Draw the erasure-robustness curves.

    Args:
        figsize: Figure size in inches. Two panels side by side, each roughly square.

    Returns:
        The finished figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    # Panels are labelled (a)/(b) in the image and the caption describes each.
    for ax, title, grid, two_stage, aug, orig in [
        (axes[0], "(a) " + DJI_TITLE, ERASURE_P, DJI_ERASURE_TWOSTAGE, DJI_ERASURE_AUG, None),
        (axes[1], "(b) " + VD_TITLE_TD, VD_ERASURE_P, VD_ERASURE_TWOSTAGE, VD_ERASURE_AUG, VD_ERASURE_ORIG),
    ]:
        if orig is not None:
            # one colour and marker per codec in Figures 4 and 7 (Okabe-Ito, colour-blind safe)
            ax.plot(grid * 100, orig, "D--", color="#555555", linewidth=1.2, markersize=3.6,
                    label="SwinJSCC, published weights (zero-shot)")
        ax.plot(
            grid * 100,
            two_stage,
            "o-",
            color="#D55E00",
            linewidth=1.4,
            markersize=5,
            label="Two-stage codec (no erasure-aug)",
        )
        ax.plot(
            grid * 100,
            aug,
            "s-",
            color="#0072B2",
            linewidth=2.0,
            markersize=5,
            label="Erasure-augmented model",
        )
        # No horizontal reference line: the 0.94 line the original figure drew is not defined
        # anywhere in the text, so a reader could not tell what it marks.
        # Plain "%", not the LaTeX "\%" the original generator carried: matplotlib's
        # default text path is not usetex, so the escape rendered literally as "[\%]".
        ax.set_xlabel("Erasure rate $p$ (%)")
        if ax is axes[0]:
            # Mark the operating point the text builds its erasure claim on, so the figure
            # carries the finding rather than leaving it to the caption.
            _i30 = int(np.where(np.isclose(ERASURE_P, 0.30))[0][0])
            ax.annotate(
                f"at 30% loss: {aug[_i30]:.3f} against {two_stage[_i30]:.3f}",
                xy=(30, aug[_i30]),
                xytext=(4, 0.55),
                fontsize=7.5,
                color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.45", linewidth=0.8),
            )
        ax.set_title(title, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_ylim([0.25, 1.02])
    axes[0].set_ylabel("DINOv2 cls-token similarity")
    axes[0].legend(loc="lower left", framealpha=0.95, fontsize=7.5)
    axes[1].legend(loc="lower left", framealpha=0.95, fontsize=7.5)
    fig.tight_layout()
    return fig


def _ink_for(im: matplotlib.image.AxesImage, value: float) -> str:
    """Pick black or white annotation ink for whichever reads better on a cell.

    Args:
        im: The drawn image, used for its colormap and normalization.
        value: The cell's value.

    Returns:
        ``"black"`` or ``"white"``, whichever has the higher WCAG contrast ratio
        against the cell's fill.
    """
    r, g, b, _ = im.cmap(im.norm(value))
    # Relative luminance (WCAG 2.x): linearize sRGB, then weight.
    chan = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    lum = 0.2126 * chan[0] + 0.7152 * chan[1] + 0.0722 * chan[2]
    return "black" if (lum + 0.05) / 0.05 >= 1.05 / (lum + 0.05) else "white"


def build_jamming_heatmap(figsize: tuple[float, float]) -> plt.Figure:
    """Draw the 24-cell cross-corpus jamming Delta heatmap.

    Args:
        figsize: Figure size in inches. Two 3x4 panels side by side sharing one colorbar
            (both use the same JAM_VMIN..JAM_VMAX range) and the row labels of panel (a).

    Returns:
        The finished figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, title, data in [
        (axes[0], "(a) " + DJI_TITLE, DJI_JAM_DELTA),
        (axes[1], "(b) " + VD_TITLE_TD, VD_JAM_DELTA),
    ]:
        # Sequential, not diverging: every cell is a positive improvement, so a
        # red-to-green map would paint the smaller VisDrone gains red.
        im = ax.imshow(data, cmap="viridis", vmin=JAM_VMIN, vmax=JAM_VMAX,
                       aspect="auto")
        ax.set_xticks(range(len(JNR_LABELS)))
        # numbers on the ticks and the unit on the axis (MDPI: consistent unit format)
        ax.set_xticklabels([lab.split("=")[1] for lab in JNR_LABELS], fontsize=8.2)
        ax.set_xlabel("JNR (dB)", fontsize=8.6)
        ax.set_yticks(range(len(JAMMERS)))
        if ax is axes[0]:
            ax.set_yticklabels(JAMMERS, fontsize=8.2)
        else:
            ax.tick_params(axis="y", left=False)
        ax.set_title(title, fontsize=9)
        for i in range(len(JAMMERS)):
            for j in range(len(JNR_LABELS)):
                # Ink colour picked from the *cell's own* luminance rather than from a
                # fixed 0.02 value cut-off: RdYlGn is pale in its middle, so the old
                # value rule printed the +0.019 DJI cell in white on light orange.
                ax.text(
                    j,
                    i,
                    f"+{data[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color=_ink_for(im, data[i, j]),
                )
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.035, pad=0.02, label=r"$\Delta$ cls-sim")
    cbar.ax.tick_params(labelsize=8.2)
    cbar.ax.yaxis.label.set_size(8.6)
    return fig


# name -> (builder, default figsize). The two figures the paper prints at \textwidth
# (394 pt = 5.46 in in the MDPI class) are drawn at about that width, so the type prints at
# its nominal size (7-9 pt) instead of being scaled to about 5 pt as at the earlier 8.8-9.4 in.
# reactive_jam is no longer printed.
FIGURES = {
    "reactive_jam": (build_reactive_jam, (8.8, 3.5)),
    "erasure_curves": (build_erasure_curves, (5.6, 2.7)),
    "jamming_heatmap": (build_jamming_heatmap, (5.6, 2.1)),
}
# height / width of the figures as printed in the paper, kept when type sizes change so the printed height
# (text width x aspect) and with it the page layout of the manuscript stay where they are
PRINTED_ASPECT = {
    "erasure_curves": 187.71937561035156 / 383.03375244140625,
    "jamming_heatmap": 170.40213012695312 / 400.983154296875,
}


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


def echo_values() -> None:
    """Echo every plotted series to stdout as the figures' accessible twin."""
    print("[threat-figs] reactive jam, AWGN SNR = 10 dB, n = 100, cls-sim vs JNR [dB]")
    print(f"  JNR                       {'  '.join(f'{v:>6d}' for v in REACTIVE_JNR)}")
    for label, series in [
        ("DJI erasure-aug baseline", DJI_REACTIVE_BASE),
        ("DJI reactive-augmented  ", DJI_REACTIVE_AUG),
        ("VD  erasure-aug baseline", VD_REACTIVE_BASE),
        ("VD  reactive-augmented  ", VD_REACTIVE_AUG),
    ]:
        print(f"  {label}  {'  '.join(f'{v:>6.3f}' for v in series)}")

    print("\n[threat-figs] erasure, nominal SNR = 10 dB, cls-sim vs erasure rate p")
    print(f"  p                         {'  '.join(f'{v:>6.2f}' for v in ERASURE_P)}")
    for label, series in [
        ("DJI two-stage    (n=375)", DJI_ERASURE_TWOSTAGE),
        ("DJI erasure-aug  (n=375)", DJI_ERASURE_AUG),
        ("VD  two-stage    (n=200)", VD_ERASURE_TWOSTAGE),
        ("VD  erasure-aug  (n=200)", VD_ERASURE_AUG),
    ]:
        print(f"  {label}  {'  '.join(f'{v:>6.3f}' for v in series)}")

    print("\n[threat-figs] 24-cell jamming Delta cls-sim, AWGN SNR = 20 dB")
    for corpus, data in [("DJI val", DJI_JAM_DELTA), ("VisDrone", VD_JAM_DELTA)]:
        print(f"  {corpus:<10}          {'  '.join(f'{c:>6}' for c in JNR_LABELS)}")
        for jammer, row in zip(JAMMERS, data, strict=True):
            print(f"    {jammer:<16}  {'  '.join(f'{v:>+6.3f}' for v in row)}")
    both = np.concatenate([DJI_JAM_DELTA.ravel(), VD_JAM_DELTA.ravel()])
    print(f"  range over all {both.size} cells: [{both.min():+.3f}, {both.max():+.3f}]")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and write the requested figures.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    ap.add_argument(
        "--only",
        nargs="+",
        choices=sorted(FIGURES),
        default=None,
        help="build only these figures (default: all three)",
    )
    ap.add_argument("--dpi", type=float, default=300.0, help="raster DPI for the .png twin")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    echo_values()
    print()
    with plt.rc_context(RC):
        for name in args.only or sorted(FIGURES):
            builder, figsize = FIGURES[name]
            fig = builder(figsize)
            pdf = args.out_dir / f"{name}.pdf"
            png = pdf.with_suffix(".png")
            box = _bbox_at_aspect(fig, PRINTED_ASPECT[name]) if name in PRINTED_ASPECT else "tight"
            fig.savefig(pdf, bbox_inches=box)
            fig.savefig(png, dpi=args.dpi, bbox_inches=box)
            plt.close(fig)
            print(f"[threat-figs] wrote {pdf} and {png}  (figsize {figsize[0]}x{figsize[1]} in)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
