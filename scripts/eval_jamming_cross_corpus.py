"""Cross-corpus jamming evaluation: the harness behind the paper's jamming results.

Runs the jamming sweep on the DJI validation split and on VisDrone-DET test-dev for
the two-stage cross-corpus checkpoint (tag P49) against the published SwinJSCC
weights (ORIG, DIV2K-trained).

Method:
  * Both networks are SwinJSCC w/ SA, built with channel_type="awgn".
  * We monkey-patch ``net.feature_pass_channel`` to bypass SwinJSCC's built-in
    Gaussian channel and route the (B, N, C) latent through the *external*
    ``semvideo.channel.jamming.JammingChannel`` instead. The latent is treated
    as a flat real-valued symbol vector of shape (1, N*C); JammingChannel adds
    AWGN at the requested SNR plus an interference term of the requested
    jammer_type and JNR. We preserve SwinJSCC's per-tensor unit-power
    normalisation so the noise/jammer power numbers behave like the native
    channel would.

Sweeps:
  jammer_type ∈ {barrage, pulsed, single_tone}  ×  JNR_dB ∈ {0, 5, 10, 15}
  at fixed background SNR = 20 dB.

Benchmarks:
  * DJI val (codec-resolution crops at 512x512)
  * VisDrone-DET test-dev (codec-resolution crops at 512x512)

Outputs:
  experiments/jamming_cross_corpus_<ts>/per_frame.csv
  experiments/jamming_cross_corpus_<ts>/summary.csv
  experiments/jamming_cross_corpus_<ts>/summary.json

A raw-codec (no JSCC) reference is included so that the table reads
"P49 vs ORIG vs no-codec floor" per (bench, jammer, JNR) cell.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMVIDEO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = SEMVIDEO_ROOT / "experiments"
# Upstream SwinJSCC code is not distributed with this repository (see README, "Third-party
# code"). Set SWINJSCC_DIR, or place it at <parent of this repository>/third_party/SwinJSCC
# (/third_party/SwinJSCC when the repository is mounted at /workspace in the Docker image).
SWINJSCC_DIR = Path(os.environ.get("SWINJSCC_DIR", REPO_ROOT / "third_party" / "SwinJSCC"))

# Ensure imports from the project + SwinJSCC checkpoint resolve like the other
# eval scripts (they ``os.chdir`` to SWINJSCC_DIR so relative checkpoint paths
# inside SwinJSCC's loader work).
sys.path.insert(0, str(SEMVIDEO_ROOT / "scripts"))
sys.path.insert(0, str(SEMVIDEO_ROOT / "src"))
sys.path.insert(0, str(SWINJSCC_DIR))
os.chdir(str(SWINJSCC_DIR))

from eval_swinjscc_dinov2 import build_swinjscc  # noqa: E402
from eval_benchmark_sweep import (  # noqa: E402
    DINOv2Extractor,
    cossim,
    list_images,
    psnr,
)
from semvideo.channel.jamming import JammingChannel  # noqa: E402
from semvideo.channel.jamming_reactive import ReactiveJammer  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Jamming substitute for SwinJSCC.feature_pass_channel
# ---------------------------------------------------------------------------


class JammingPatcher:
    """Monkey-patches ``net.feature_pass_channel`` to push the SwinJSCC
    latent through the *external* JammingChannel instead of the built-in AWGN
    Gaussian noise layer.

    Power normalisation mirrors SwinJSCC's own ``Channel.forward``:
      * normalise to unit power (per-tensor), apply noise, scale back.

    Usage::

        patcher = JammingPatcher(net)
        with patcher(jammer_type="barrage", jnr_db=10.0):
            recon, *_ = net(x, given_SNR=20.0, given_rate=None)
    """

    def __init__(self, net, rx_defense: str = "none", rx_k: float = 3.0):
        self.net = net
        self._original_fpc = net.feature_pass_channel
        self._jammer: JammingChannel | None = None
        self.rx_defense = rx_defense
        self.rx_k = rx_k
        # Reactive-automaton settings. The defaults are the ones every reactive number in
        # the paper uses; an adaptive-adversary sweep overrides them per run.
        self.reactive_detect_threshold = 0.5
        self.reactive_latency = 8
        self.reactive_hold = 64
        self.reactive_power_reference = "followed"

    def _apply_rx_defense(self, x: torch.Tensor) -> torch.Tensor:
        """Classical training-free anti-jam receiver front-end (RJ9 baseline).

        Robust per-vector scale via MAD (median absolute deviation, 1.4826 =
        consistency constant for a Gaussian); symbols beyond ``rx_k * sigma``
        are clipped (amplitude limiter) or excised (blanked to zero) - the two
        textbook impulsive-interference mitigations applied on the received
        symbol vector before decoding.
        """
        if self.rx_defense == "none":
            return x
        med = x.median()
        sigma = 1.4826 * (x - med).abs().median()
        thr = self.rx_k * sigma
        if self.rx_defense == "clip":
            return med + (x - med).clamp(-thr, thr)
        if self.rx_defense == "excise":
            return torch.where((x - med).abs() > thr, torch.zeros_like(x), x)
        raise ValueError(f"unknown rx_defense {self.rx_defense}")

    def _patched_feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        if self._jammer is None:
            # Fall back to original if no jammer is active.
            return self._original_fpc(feature, chan_param, avg_pwr)

        # Per-tensor unit-power normalisation, matching SwinJSCC's Channel.
        if avg_pwr:
            channel_tx = feature / torch.sqrt(avg_pwr * 2)
            pwr_scale = torch.sqrt(avg_pwr * 2)
        else:
            pwr = torch.mean(feature ** 2) * 2
            channel_tx = feature / torch.sqrt(pwr)
            pwr_scale = torch.sqrt(pwr)

        input_shape = channel_tx.shape
        # JammingChannel expects (B, N) – treat the flattened latent as a real
        # symbol vector of length total_elements with batch dim 1.
        flat = channel_tx.reshape(1, -1)
        noisy_flat = self._jammer(flat, snr_db=float(chan_param))
        noisy_flat = self._apply_rx_defense(noisy_flat)
        noisy = noisy_flat.reshape(input_shape)
        return noisy * pwr_scale

    def __call__(self, jammer_type: str, jnr_db: float, duty_cycle: float = 0.5,
                 tone_freq_norm: float = 0.25, reactive_power_reference: str = "followed",
                 erasure_p: float | None = None):
        return _PatcherContext(self, jammer_type, jnr_db, duty_cycle, tone_freq_norm,
                               reactive_power_reference, erasure_p)


class _PatcherContext:
    def __init__(self, patcher: JammingPatcher, jammer_type: str, jnr_db: float,
                 duty_cycle: float, tone_freq_norm: float,
                 reactive_power_reference: str = "followed",
                 erasure_p: float | None = None):
        self.patcher = patcher
        if jammer_type == "erasure":
            # Not a jammer: symbol erasure carried through the same hook, because the
            # channel exposes the same forward(x, snr_db=...) contract. This is what gives
            # the erasure result a downstream anchor instead of a metric-only claim.
            from semvideo.channel.packet_loss import BernoulliErasureChannel
            if erasure_p is None:
                raise ValueError("jammer_type='erasure' needs erasure_p")

            class _PureErasure(BernoulliErasureChannel):
                """Zero-fill only: survivors take no residual AWGN.

                The hook hands every channel the link SNR, which is right for the jammers
                but would add noise to surviving symbols here. The paper's erasure table is
                pure zero-fill, so the two must match or the numbers are not comparable.
                """

                def forward(self, x, p=None, snr_db=None):  # noqa: D102
                    return super().forward(x, p=p, snr_db=None)

            self.jammer = _PureErasure(p=float(erasure_p), snr_db=None)
        elif jammer_type == "reactive":
            # 5th mode: DRFM-style automaton (arXiv 2502.02385). Cls_sim drops
            # only when the jammer's hold window aligns with high-energy symbols.
            # Detect threshold 0.5 = "above-average power", react_latency 8 +
            # hold_duration 64 mimic realistic hardware. JammingChannel and
            # ReactiveJammer expose the same forward(x, snr_db=...) contract.
            self.jammer = ReactiveJammer(
                jnr_db=jnr_db,
                detect_threshold=patcher.reactive_detect_threshold,
                react_latency=patcher.reactive_latency,
                hold_duration=patcher.reactive_hold,
                power_reference=reactive_power_reference,
            ).to(DEVICE)
        else:
            self.jammer = JammingChannel(
                jammer_type=jammer_type,
                jnr_db=jnr_db,
                duty_cycle=duty_cycle,
                snr_db=20.0,  # overridden by the snr_db kwarg in forward()
                tone_freq_norm=tone_freq_norm,
            ).to(DEVICE)

    def __enter__(self):
        self.patcher._jammer = self.jammer
        # Kept after __exit__ so the caller can read power/duty diagnostics.
        self.patcher._last_jammer = self.jammer
        self.patcher.net.feature_pass_channel = self.patcher._patched_feature_pass_channel
        return self.patcher

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patcher.net.feature_pass_channel = self.patcher._original_fpc
        self.patcher._jammer = None
        return False


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


# Cache of successful symlink re-rootings, keyed by the directory holding the
# links: (visible_base, index into the link target's parts). Populated on the
# first dangling link so the remaining frames cost a single stat() each.
_SYMLINK_REMAP_CACHE: dict[str, tuple[Path, int]] = {}


def _remap_invisible_symlink(path: Path) -> Path | None:
    """Re-root an absolute symlink target that is invisible in this mount namespace.

    Flat benchmark dirs (e.g. ``uavdt_val_flat``) are built from symlinks whose
    targets are baked as *host* absolute paths (``/home/<user>/data/...``). Inside
    a Docker container mounted as ``-v $HOME/data:/data`` those targets do not
    exist, so the link dangles and ``cv2.imread`` silently returns ``None``.
    The same bytes are usually still reachable under one of the link directory's
    own visible ancestors, so try re-rooting successively shorter target tails.

    Args:
        path: Path of the (possibly dangling) symlink.

    Returns:
        A readable path to the same file, or ``None`` if nothing was found.
    """
    if not path.is_symlink():
        return None
    try:
        target = Path(os.readlink(path))
    except OSError:
        return None
    if target.is_file():
        return target
    parts = target.parts
    key = str(path.parent)
    cached = _SYMLINK_REMAP_CACHE.get(key)
    if cached is not None:
        base, idx = cached
        cand = base.joinpath(*parts[idx:])
        if cand.is_file():
            return cand
    for base in [path.parent, *path.parent.parents]:
        for idx in range(1, len(parts)):
            cand = base.joinpath(*parts[idx:])
            if cand.is_file():
                _SYMLINK_REMAP_CACHE[key] = (base, idx)
                return cand
    return None


def load_image_bgr(path: Path) -> np.ndarray | None:
    """Read an image as BGR, surviving dangling symlinks and odd path encodings.

    ``cv2.imread`` reports every failure the same way (``None``), which is how a
    whole corpus can be dropped without a word. Fall back to (1) re-rooting an
    invisible symlink target and (2) a byte-level ``imdecode`` before giving up.

    Args:
        path: Image file path.

    Returns:
        Decoded ``(H, W, 3)`` BGR array, or ``None`` if the file is unreadable.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is not None:
        return bgr
    alt = _remap_invisible_symlink(path)
    if alt is not None:
        bgr = cv2.imread(str(alt), cv2.IMREAD_COLOR)
        if bgr is not None:
            return bgr
    src = alt if alt is not None else path
    try:
        buf = np.fromfile(str(src), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def describe_unreadable(path: Path) -> str:
    """One-line diagnosis of why ``path`` could not be decoded.

    Args:
        path: Path that ``load_image_bgr`` returned ``None`` for.

    Returns:
        Human-readable reason, naming the unresolved symlink target when that is
        the cause (the usual Docker mount mistake).
    """
    if path.is_symlink():
        try:
            target = os.readlink(path)
        except OSError:
            target = "<unreadable link>"
        if not Path(target).is_file():
            return (
                f"{path} is a symlink to {target}, which does not exist in this "
                f"mount namespace (mount the symlink *targets* into the container, "
                f"or materialise real files in the flat dir)"
            )
    if not path.exists():
        return f"{path} does not exist"
    return f"{path} exists but no decoder accepted it (truncated/corrupt file?)"


def list_dji_val_frames(root: Path, max_n: int) -> list[Path]:
    """Deterministic subsample of DJI val frames (linspace by sorted name)."""
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise FileNotFoundError(f"no frames in {root}")
    if max_n > 0 and len(paths) > max_n:
        idx = np.linspace(0, len(paths) - 1, max_n, dtype=int)
        paths = [paths[int(i)] for i in idx]
    return paths


def list_visdrone_frames(root: Path, max_n: int, seed: int = 0) -> list[Path]:
    """Reproducible random subsample of VisDrone test-dev images."""
    return list_images(root, max_n if max_n > 0 else None)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


@torch.no_grad()
def reconstruct(net, patcher, crop_bgr, snr_db, jammer_type, jnr_db):
    """Center-cropped 512x512 input -> SwinJSCC encode -> JammingChannel -> decode."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    duty = getattr(patcher, "default_duty_cycle", 0.5)
    power_ref = getattr(patcher, "reactive_power_reference", "followed")
    with patcher(jammer_type=jammer_type, jnr_db=jnr_db, duty_cycle=duty,
                 reactive_power_reference=power_ref):
        rec, _, _, _, _ = net(x, given_SNR=snr_db, given_rate=None)
    rec = rec.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    rec_uint = (rec * 255.0).astype(np.uint8)
    return cv2.cvtColor(rec_uint, cv2.COLOR_RGB2BGR)


def center_square_resize(bgr, codec_size):
    h, w = bgr.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = bgr[y0:y0 + side, x0:x0 + side]
    if side == codec_size:
        return crop
    # INTER_AREA is the right kernel only when decimating; frames whose short
    # side is below the codec size (small legacy corpora) must be upsampled.
    interp = cv2.INTER_AREA if side > codec_size else cv2.INTER_CUBIC
    return cv2.resize(crop, (codec_size, codec_size), interpolation=interp)


# ---------------------------------------------------------------------------
# Per-(model, bench) eval
# ---------------------------------------------------------------------------


def eval_model_on_bench(
    net, patcher, dino, paths, label, bench, codec_size, snr_db,
    jammer_types, jnr_grid, per_frame_writer,
):
    """Eval a single model on a single benchmark across the (jammer, JNR) grid.

    Returns dict keyed by (jammer, jnr) -> {cls_mean, cls_std, patch_mean,
    patch_std, psnr_mean, psnr_std, n}.
    """
    print(f"\n[bench] {label} on {bench}: {len(paths)} images, SNR={snr_db}")

    # Cache crops + reference DINOv2 features (done once across all cells).
    crops = {}
    refs = {}
    unreadable: list[Path] = []
    for p in paths:
        bgr = load_image_bgr(p)
        if bgr is None:
            unreadable.append(p)
            continue
        crop = center_square_resize(bgr, codec_size)
        crops[p.stem] = crop
        refs[p.stem] = dino.features(crop)
    print(f"[bench]   cached {len(refs)} reference DINOv2 features")
    if unreadable:
        print(f"[bench]   WARNING: {len(unreadable)}/{len(paths)} frames unreadable; "
              f"first: {describe_unreadable(unreadable[0])}")
    if not crops:
        raise RuntimeError(
            f"{bench}: all {len(paths)} listed frames failed to decode, so every "
            f"metric would be n=0/NaN. First failure: "
            f"{describe_unreadable(unreadable[0]) if unreadable else 'no frames listed'}"
        )

    cell_results: dict[tuple[str, float], dict] = {}
    for jammer in jammer_types:
        for jnr in jnr_grid:
            cls_vals, patch_vals, psnr_vals = [], [], []
            power_ratios: list[float] = []
            for stem, crop in crops.items():
                rec_bgr = reconstruct(net, patcher, crop, snr_db, jammer, jnr)
                cls_o, patch_o = refs[stem]
                cls_r, patch_r = dino.features(rec_bgr)
                c = cossim(cls_o, cls_r)
                pp = cossim(patch_o, patch_r)
                ps = psnr(crop, rec_bgr)
                cls_vals.append(c)
                patch_vals.append(pp)
                psnr_vals.append(ps)
                # Injected-energy accounting: only the reactive jammer records it.
                pr = getattr(getattr(patcher, "_last_jammer", None), "last_power_ratio", None)
                if pr is not None:
                    power_ratios.append(float(pr))
                per_frame_writer.writerow([
                    label, bench, jammer, f"{jnr:.1f}", snr_db, stem,
                    round(c, 5), round(pp, 5), round(ps, 3),
                ])
            cell_results[(jammer, jnr)] = dict(
                jammer=jammer, jnr=jnr,
                cls_mean=float(np.mean(cls_vals)),
                cls_std=float(np.std(cls_vals)),
                patch_mean=float(np.mean(patch_vals)),
                patch_std=float(np.std(patch_vals)),
                psnr_mean=float(np.mean(psnr_vals)),
                psnr_std=float(np.std(psnr_vals)),
                n=len(cls_vals),
                power_ratio=float(np.mean(power_ratios)) if power_ratios else None,
            )
            r = cell_results[(jammer, jnr)]
            pr_txt = f"  Pinj/Pbarrage={r['power_ratio']:.2f}x" if r["power_ratio"] else ""
            print(
                f"  {jammer:>11s}  JNR={jnr:>4.1f}  "
                f"cls={r['cls_mean']:.4f}+-{r['cls_std']:.4f}  "
                f"patch={r['patch_mean']:.4f}+-{r['patch_std']:.4f}  "
                f"PSNR={r['psnr_mean']:5.2f}{pr_txt}",
                flush=True,
            )
    return cell_results


def eval_raw_reference(dino, paths, bench, codec_size, jammer_types, jnr_grid,
                       per_frame_writer, snr_db):
    """Raw (no-codec, no-channel) reference: cls_sim = 1 by construction, but we
    record PSNR=inf and patch_sim=1 to populate the table. This gives a 'ceiling'
    row per (bench, jammer, JNR) for completeness."""
    print(f"\n[ref] no-codec reference on {bench}: {len(paths)} images")
    cell_results: dict[tuple[str, float], dict] = {}
    for jammer in jammer_types:
        for jnr in jnr_grid:
            cls_vals, patch_vals, psnr_vals = [], [], []
            first_bad: Path | None = None
            for p in paths:
                bgr = load_image_bgr(p)
                if bgr is None:
                    first_bad = first_bad or p
                    continue
                crop = center_square_resize(bgr, codec_size)
                # Reference = self, by construction cls_sim=patch_sim=1, psnr=inf.
                cls_vals.append(1.0)
                patch_vals.append(1.0)
                psnr_vals.append(float("inf"))
                per_frame_writer.writerow([
                    "RAW", bench, jammer, f"{jnr:.1f}", snr_db, p.stem,
                    1.0, 1.0, "inf",
                ])
            if not cls_vals:
                raise RuntimeError(
                    f"{bench}: all {len(paths)} listed frames failed to decode. "
                    f"First failure: "
                    f"{describe_unreadable(first_bad) if first_bad else 'no frames listed'}"
                )
            cell_results[(jammer, jnr)] = dict(
                jammer=jammer, jnr=jnr,
                cls_mean=1.0, cls_std=0.0,
                patch_mean=1.0, patch_std=0.0,
                psnr_mean=float("inf"), psnr_std=0.0,
                n=len(cls_vals),
            )
    return cell_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--p49-ckpt", type=str,
        default=str(SEMVIDEO_ROOT / "experiments" / "p49_p46_awgn_finetune" / "best.pt"),
    )
    ap.add_argument(
        "--orig-ckpt", type=str,
        default=str(
            SWINJSCC_DIR / "checkpoint" / "SwinJSCC" /
            "SwinJSCC w_ SA" / "SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model"
        ),
    )
    ap.add_argument("--C", type=int, default=96)
    ap.add_argument("--codec-size", type=int, default=512)
    ap.add_argument("--snr", type=int, default=20)
    ap.add_argument(
        "--reactive-power-reference", type=str, default="followed",
        choices=["followed", "matched_barrage"],
        help="Power convention for the reactive jammer. 'followed' scales interference to "
             "the followed symbol's energy (on-target JSR = JNR, more total energy than "
             "barrage); 'matched_barrage' keeps the gating but equalises the injected "
             "budget against an equal-JNR barrage.",
    )
    ap.add_argument("--reactive-detect-threshold", type=float, default=0.5,
                    help="Reactive automaton gating threshold, in units of mean symbol "
                         "energy. Lower = the jammer engages on more symbols.")
    ap.add_argument("--reactive-latency", type=int, default=8,
                    help="Consecutive above-threshold symbols before the jammer engages.")
    ap.add_argument("--reactive-hold", type=int, default=64,
                    help="Symbols the jammer stays engaged once triggered.")
    ap.add_argument("--duty-cycle", type=float, default=0.5,
                    help="Pulsed-jammer duty cycle d; active-block power is 1/d times the "
                         "time-average JNR level. Affects the pulsed mode only.")
    ap.add_argument(
        "--jammer-types", type=str, nargs="+",
        default=["barrage", "pulsed", "single_tone"],
    )
    ap.add_argument(
        "--jnr-grid", type=float, nargs="+",
        default=[0.0, 5.0, 10.0, 15.0],
    )
    ap.add_argument(
        "--dji-val-dir", type=str,
        default=str(REPO_ROOT / "data" / "dji" / "frames_512_v1" / "val"),
    )
    ap.add_argument(
        "--visdrone-dir", type=str,
        default=str(REPO_ROOT / "data" / "visdrone" / "VisDrone2019-DET-test-dev" / "images"),
    )
    ap.add_argument("--n-dji", type=int, default=80)
    ap.add_argument("--n-visdrone", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p49-label", type=str, default="P49",
                    help="Row label for the --p49-ckpt model (e.g. P68/P80 when "
                         "pointing the flag at a different checkpoint).")
    ap.add_argument("--rx-defense", type=str, default="none",
                    choices=["none", "clip", "excise"],
                    help="Classical receiver-side anti-jam front-end applied to the "
                         "received symbols before decoding (RJ9 baseline).")
    ap.add_argument("--rx-k", type=float, default=3.0,
                    help="Threshold multiple of the robust (MAD) sigma for --rx-defense.")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--skip-visdrone", action="store_true")
    ap.add_argument("--skip-orig", action="store_true")
    ap.add_argument("--skip-p49", action="store_true")
    ap.add_argument("--skip-raw", action="store_true",
                    help="Skip the no-codec reference rows (saves a few seconds).")
    args = ap.parse_args()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = str(EXPERIMENTS_DIR / f"jamming_cross_corpus_{ts}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[setup] out:        {out_dir}")
    print(f"[setup] P49:        {args.p49_ckpt}")
    print(f"[setup] ORIG:       {args.orig_ckpt}")
    print(f"[setup] codec_size: {args.codec_size}, C={args.C}, SNR={args.snr}")
    print(f"[setup] jammers:    {args.jammer_types}")
    print(f"[setup] JNR grid:   {args.jnr_grid}")
    print(f"[setup] n_dji={args.n_dji}, n_visdrone={args.n_visdrone}")

    # Resolve benchmark paths once (so subsamples match across models).
    dji_paths = list_dji_val_frames(Path(args.dji_val_dir), args.n_dji)
    print(f"[data] DJI val: {len(dji_paths)} frames")
    if not args.skip_visdrone:
        vd_paths = list_visdrone_frames(Path(args.visdrone_dir), args.n_visdrone, seed=args.seed)
        print(f"[data] VisDrone: {len(vd_paths)} images")
    else:
        vd_paths = []

    print("[load] DINOv2 ViT-S/14...")
    dino = DINOv2Extractor()

    per_frame_path = out_dir / "per_frame.csv"
    summary_csv_path = out_dir / "summary.csv"
    fp_pf = open(per_frame_path, "w", newline="")
    pw = csv.writer(fp_pf)
    pw.writerow([
        "label", "bench", "jammer", "jnr", "snr", "image",
        "cls_sim", "patch_sim", "psnr",
    ])

    all_results: dict[str, dict[str, dict]] = {}

    bench_paths = [("DJI_val", dji_paths)]
    if not args.skip_visdrone:
        bench_paths.append(("VisDrone_testdev", vd_paths))

    # Raw reference (cheap, always 1.0 – skipable for token budget).
    if not args.skip_raw:
        all_results["RAW"] = {}
        for bench, paths in bench_paths:
            all_results["RAW"][bench] = eval_raw_reference(
                dino, paths, bench, args.codec_size,
                args.jammer_types, args.jnr_grid, pw, args.snr,
            )

    # Eval each model on each benchmark; load checkpoints lazily to fit in 6 GB.
    models_to_run = []
    defense_suffix = ("" if args.rx_defense == "none"
                      else f"+{args.rx_defense.upper()}{args.rx_k:g}")
    if not args.skip_p49:
        models_to_run.append((args.p49_label + defense_suffix, args.p49_ckpt))
    if not args.skip_orig:
        models_to_run.append(("ORIG" + defense_suffix, args.orig_ckpt))

    for label, ckpt in models_to_run:
        print(f"\n[load] {label} SwinJSCC + checkpoint: {ckpt}")
        net = build_swinjscc(
            Path(ckpt),
            image_dims=(3, args.codec_size, args.codec_size),
            C=args.C,
        )
        patcher = JammingPatcher(net, rx_defense=args.rx_defense, rx_k=args.rx_k)
        patcher.default_duty_cycle = args.duty_cycle
        patcher.reactive_power_reference = args.reactive_power_reference
        patcher.reactive_detect_threshold = args.reactive_detect_threshold
        patcher.reactive_latency = args.reactive_latency
        patcher.reactive_hold = args.reactive_hold
        all_results[label] = {}
        for bench, paths in bench_paths:
            all_results[label][bench] = eval_model_on_bench(
                net, patcher, dino, paths, label, bench,
                args.codec_size, args.snr,
                args.jammer_types, args.jnr_grid, pw,
            )
        # Free GPU memory before loading the next model.
        del net, patcher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fp_pf.close()

    # ----- summary CSV -----
    with open(summary_csv_path, "w", newline="") as fp:
        sw = csv.writer(fp)
        sw.writerow([
            "label", "bench", "jammer", "jnr", "snr", "n",
            "cls_mean", "cls_std",
            "patch_mean", "patch_std",
            "psnr_mean", "psnr_std",
        ])
        for label, by_bench in all_results.items():
            for bench, cells in by_bench.items():
                for (jammer, jnr), r in cells.items():
                    sw.writerow([
                        label, bench, jammer, f"{jnr:.1f}", args.snr, r["n"],
                        f"{r['cls_mean']:.5f}", f"{r['cls_std']:.5f}",
                        f"{r['patch_mean']:.5f}", f"{r['patch_std']:.5f}",
                        f"{r['psnr_mean']:.3f}", f"{r['psnr_std']:.3f}",
                    ])

    # ----- summary JSON -----
    json_summary = {
        "p49_ckpt": str(args.p49_ckpt),
        "orig_ckpt": str(args.orig_ckpt),
        "codec_size": args.codec_size, "C": args.C, "snr": args.snr,
        "jammer_types": list(args.jammer_types),
        "jnr_grid": list(args.jnr_grid),
        "n_dji": args.n_dji, "n_visdrone": args.n_visdrone,
        "results": {
            label: {
                bench: {f"{j}@JNR{jnr:.1f}": r for (j, jnr), r in cells.items()}
                for bench, cells in by_bench.items()
            }
            for label, by_bench in all_results.items()
        },
        "note": (
            "External JammingChannel substituted for SwinJSCC's built-in channel "
            "via monkey-patched feature_pass_channel. Per-tensor unit-power "
            "normalisation preserved. Both P49 and ORIG see identical jamming "
            "noise realisations per cell."
        ),
    }
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(json_summary, fp, indent=2, default=str)

    # ----- terminal pretty table -----
    print("\n" + "=" * 110)
    print(f"FINAL TABLE – jamming cross-corpus, SNR={args.snr} dB background, codec-native (512x512)")
    print("=" * 110)
    for bench in [b for b, _ in bench_paths]:
        print(f"\n[ {bench} ]")
        header = (
            f"{'jammer':>11s} {'JNR':>4s} | "
            f"{'P49 cls':>14s} {'ORIG cls':>14s} {'d(P49-ORIG)':>11s} | "
            f"{'P49 PSNR':>10s} {'ORIG PSNR':>10s}"
        )
        print(header)
        print("-" * len(header))
        for jammer in args.jammer_types:
            for jnr in args.jnr_grid:
                p49 = all_results.get("P49", {}).get(bench, {}).get((jammer, jnr))
                orig = all_results.get("ORIG", {}).get(bench, {}).get((jammer, jnr))
                p49_cls = f"{p49['cls_mean']:.4f}+-{p49['cls_std']:.3f}" if p49 else "    -    "
                orig_cls = f"{orig['cls_mean']:.4f}+-{orig['cls_std']:.3f}" if orig else "    -    "
                if p49 and orig:
                    delta = p49["cls_mean"] - orig["cls_mean"]
                    delta_s = f"{delta:+.4f}"
                else:
                    delta_s = "   -   "
                p49_psnr = f"{p49['psnr_mean']:.2f}" if p49 else "  -  "
                orig_psnr = f"{orig['psnr_mean']:.2f}" if orig else "  -  "
                print(
                    f"{jammer:>11s} {jnr:>4.1f} | "
                    f"{p49_cls:>14s} {orig_cls:>14s} {delta_s:>11s} | "
                    f"{p49_psnr:>10s} {orig_psnr:>10s}"
                )
    print("=" * 110)
    print(f"\n[saved] {per_frame_path}")
    print(f"[saved] {summary_csv_path}")
    print(f"[saved] {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
