"""Public-benchmark DINOv2 cls_sim sweep for our SwinJSCC checkpoints.

Grounds the cls_sim results on UAV imagery against publicly known
benchmarks. This script runs an arbitrary checkpoint on:

  - Kodak (24 images, the classical neural-codec benchmark)
  - Any other ``<benchmark_root>/*.{png,jpg}`` directory the user points us at
    (VisDrone, AI-TOD-v2, ImageNet val sample, etc.)

Pipeline mirrors ``scripts/eval_swinjscc_dinov2.py`` but works on still
images instead of a video. Reports both PSNR (so we can cross-check
against SwinJSCC's published Kodak numbers) and DINOv2 cls_sim
(ViT-S/14, the paper's metric).

Usage:
    python scripts/eval_benchmark_sweep.py \
        --checkpoint experiments/<run_id>/best.pt \
        --benchmark-root data/kodak \
        --label MYRUN_on_Kodak \
        --snrs 1 4 7 10 13 20
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

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMVIDEO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = SEMVIDEO_ROOT / "experiments"
# Upstream SwinJSCC code is not distributed with this repository (see README, "Third-party
# code"). Set SWINJSCC_DIR, or place it at <parent of this repository>/third_party/SwinJSCC
# (/third_party/SwinJSCC when the repository is mounted at /workspace in the Docker image).
SWINJSCC_DIR = Path(os.environ.get("SWINJSCC_DIR", REPO_ROOT / "third_party" / "SwinJSCC"))
sys.path.insert(0, str(SEMVIDEO_ROOT / "scripts"))  # to import sibling eval_swinjscc_dinov2
sys.path.insert(0, str(SEMVIDEO_ROOT / "src"))
sys.path.insert(0, str(SWINJSCC_DIR))
os.chdir(str(SWINJSCC_DIR))

from net.network import SwinJSCC  # noqa: E402, F401

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The metric path (DINOv2 class-token cosine, PSNR) lives in semvideo.metrics.dinov2_clssim so
# that scripts which need only the metric can use it without this module's SwinJSCC import.
# It is re-exported here, so ``from eval_benchmark_sweep import DINOv2Extractor`` still works.
from semvideo.metrics.dinov2_clssim import (  # noqa: E402, F401
    DINO_RES,
    DINOv2Extractor,
    cossim,
    psnr,
)


# Reuse the canonical loader / forward path from eval_swinjscc_dinov2.py
# instead of re-implementing – fewer chances to drift from the proven config.
from eval_swinjscc_dinov2 import build_swinjscc as _build_swinjscc_awgn  # noqa: E402

import torch.nn as nn  # noqa: E402


def build_swinjscc(checkpoint, image_dims=(3, 512, 512), C=96, channel="awgn"):
    """Wrap the canonical loader so we can swap channel_type without forking.

    ``channel`` ∈ {"awgn", "rayleigh"} – passed into SwinJSCC's channel module
    selection at init time. (The W&B-style "given_SNR" forward kwarg controls
    the SNR; the channel TYPE has to be picked at construction time.)
    """
    if channel == "awgn":
        return _build_swinjscc_awgn(checkpoint, image_dims=image_dims, C=C)
    # Rebuild with the right channel_type – same recipe as the sister
    # script, but channel_type="rayleigh" instead of "awgn".
    from types import SimpleNamespace
    import torch
    from net.network import SwinJSCC
    model_name = "SwinJSCC_w/_SA"
    sj_args = SimpleNamespace(
        model=model_name, channel_type=channel, C=str(C),
        multiple_snr="10", distortion_metric="MSE", trainset="DIV2K", logger=None,
    )
    cfg = SimpleNamespace(
        seed=42, pass_channel=True, CUDA=True, device=DEVICE, norm=False,
        print_step=100, plot_step=10000, normalize=False, learning_rate=1e-4,
        image_dims=image_dims, batch_size=1, downsample=4, logger=None,
    )
    cfg.encoder_kwargs = dict(
        model=model_name, img_size=(image_dims[1], image_dims[2]), patch_size=2,
        in_chans=3, embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2],
        num_heads=[4, 6, 8, 10], C=C, window_size=8, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True,
    )
    cfg.decoder_kwargs = dict(
        model=model_name, img_size=(image_dims[1], image_dims[2]),
        embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2],
        num_heads=[10, 8, 6, 4], C=C, window_size=8, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True,
    )
    net = SwinJSCC(sj_args, cfg).to(DEVICE).eval()
    state = torch.load(str(checkpoint), map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    target = net.state_dict()
    filtered, dropped = {}, []
    for k, v in state.items():
        if k in target and target[k].shape == v.shape:
            filtered[k] = v
        elif k in target:
            dropped.append(k)
    if dropped:
        print(f"[load] dropped {len(dropped)} shape-mismatched tensors")
    missing, unexpected = net.load_state_dict(filtered, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    return net


@torch.no_grad()
def reconstruct(net, img_bgr, snr_db, codec_size):
    """Center-crop to square, resize to codec_size, encode/decode at given SNR."""
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = img_bgr[y0:y0 + side, x0:x0 + side]
    crop_resized = cv2.resize(crop, (codec_size, codec_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    rec, _, _, _, _ = net(x, given_SNR=snr_db, given_rate=None)
    rec = rec.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    rec_uint = (rec * 255.0).astype(np.uint8)
    rec_bgr = cv2.cvtColor(rec_uint, cv2.COLOR_RGB2BGR)
    return crop_resized, rec_bgr


class CLIPExtractor:
    """OpenAI CLIP ViT-B/32 features. Different family from DINOv2 – addresses
    the "what if DINOv2 cherry-picks" objection. Output is the
    pre-projection ``pooler_output`` (CLS-equivalent) and the patch tokens.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPVisionModel, CLIPImageProcessor
        self.model = CLIPVisionModel.from_pretrained(model_name).to(DEVICE).eval()
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.backbone = model_name

    @torch.no_grad()
    def features(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(DEVICE)
        out = self.model(**inputs)
        cls = out.pooler_output.squeeze(0)
        # patch tokens (skip the CLS at index 0)
        patches = out.last_hidden_state[0, 1:, :]
        return cls, patches


def list_images(root: Path, max_n: int | None) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
    if max_n is not None and len(paths) > max_n:
        # Random reproducible subsample for large benchmarks.
        rng = np.random.default_rng(0)
        idx = rng.choice(len(paths), size=max_n, replace=False)
        idx.sort()
        paths = [paths[i] for i in idx]
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--benchmark-root", type=str, required=True)
    ap.add_argument("--label", type=str, default="SwinJSCC_eval")
    ap.add_argument("--max-n", type=int, default=None,
                    help="Subsample large benchmarks (random seed=0).")
    ap.add_argument("--snrs", type=int, nargs="+", default=[1, 4, 7, 10, 13, 20])
    ap.add_argument("--codec-size", type=int, default=512)
    ap.add_argument("--C", type=int, default=96)
    ap.add_argument("--metric-backbone", type=str, default="dinov2_vits14",
                    choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
                            "clip_vitb32"],
                    help="Feature backbone for cls_sim. DINOv2 family or "
                         "clip_vitb32 (a different family, as a "
                         "metric-robustness check).")
    ap.add_argument("--channel", type=str, default="awgn",
                    choices=["awgn", "rayleigh"],
                    help="Wireless channel type. Rayleigh tests fading-channel "
                         "robustness on public benchmarks.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = str(EXPERIMENTS_DIR / f"benchmark_eval_{args.label}_{ts}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] out: {out_dir}")
    print(f"[setup] checkpoint: {args.checkpoint}")
    print(f"[setup] benchmark: {args.benchmark_root}")
    print(f"[setup] codec_size: {args.codec_size}, C: {args.C}")

    print(f"[load] metric backbone: {args.metric_backbone}...")
    if args.metric_backbone == "clip_vitb32":
        dino = CLIPExtractor()
    else:
        dino = DINOv2Extractor(backbone=args.metric_backbone)
    print(f"[load] SwinJSCC (channel={args.channel})...")
    net = build_swinjscc(
        Path(args.checkpoint),
        image_dims=(3, args.codec_size, args.codec_size),
        C=args.C,
        channel=args.channel,
    )

    img_paths = list_images(Path(args.benchmark_root), args.max_n)
    print(f"[data] {len(img_paths)} images from {args.benchmark_root}")

    refs = {}
    crops = {}
    for p in img_paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        # We pre-resize to the codec resolution and compute DINOv2 features
        # on the *codec-resolution crop*, NOT the display res. This isolates
        # the codec's contribution from any super-resolution effect at the
        # operator pipeline. The DJI eval used display upsampling; here we
        # use the codec-native crop so the benchmark numbers are directly
        # comparable to how SwinJSCC reports Kodak PSNR (no upsampling).
        h, w = bgr.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        crop = bgr[y0:y0 + side, x0:x0 + side]
        crop_resized = cv2.resize(crop, (args.codec_size, args.codec_size),
                                  interpolation=cv2.INTER_AREA)
        crops[p.stem] = crop_resized
        refs[p.stem] = dino.features(crop_resized)

    per_path = out_dir / "per_frame.csv"
    sum_path = out_dir / "metrics.csv"

    with open(per_path, "w", newline="") as fp:
        pw = csv.writer(fp)
        pw.writerow(["label", "channel", "snr", "image", "psnr", "cls_sim", "patch_sim"])

        with open(sum_path, "w", newline="") as fp2:
            sw = csv.writer(fp2)
            sw.writerow(["label", "channel", "snr", "n",
                         "psnr_mean", "psnr_std",
                         "cls_mean", "cls_std", "patch_mean", "patch_std"])

            for snr in args.snrs:
                psnr_vals, cls_vals, patch_vals = [], [], []
                for stem, crop_bgr in crops.items():
                    orig, rec_bgr = reconstruct(net, crop_bgr, snr, args.codec_size)
                    ps = psnr(orig, rec_bgr)
                    cls_r, patch_r = dino.features(rec_bgr)
                    cls_o, patch_o = refs[stem]
                    c = cossim(cls_o, cls_r)
                    p = cossim(patch_o, patch_r)
                    psnr_vals.append(ps); cls_vals.append(c); patch_vals.append(p)
                    pw.writerow([args.label, args.channel, snr, stem,
                                 round(ps, 3), round(c, 5), round(p, 5)])
                pm, ps_ = float(np.mean(psnr_vals)), float(np.std(psnr_vals))
                cm, cs = float(np.mean(cls_vals)), float(np.std(cls_vals))
                pcm, pcs = float(np.mean(patch_vals)), float(np.std(patch_vals))
                sw.writerow([args.label, args.channel, snr, len(psnr_vals),
                             f"{pm:.3f}", f"{ps_:.3f}",
                             f"{cm:.5f}", f"{cs:.5f}",
                             f"{pcm:.5f}", f"{pcs:.5f}"])
                print(f"  {args.channel:8s} snr={snr:2d}  PSNR={pm:.2f}±{ps_:.2f}  "
                      f"cls={cm:.4f}±{cs:.4f}  patch={pcm:.4f}±{pcs:.4f}")

    print(f"\n[saved] {sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
