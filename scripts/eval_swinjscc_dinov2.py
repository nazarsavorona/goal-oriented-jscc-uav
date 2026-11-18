"""DINOv2 cls/patch similarity for SwinJSCC reconstructions on DJI eval frames.

Pipeline:
  - Sample 30 frames from the source video (same indices as the compact-codec
    DINOv2 evaluation in dinov2_metric.py – frames linspace(10850, 11800)).
  - For each frame, downsample 1920x1080 -> 512 (matches our P12 operator pipeline).
  - For each (channel, snr) ∈ {awgn} × {0,5,10,15,20}:
      - SwinJSCC w/ SA forward (their AWGN channel implementation)
      - Upsample 512 -> 1080 (bicubic, same operator pipeline)
      - DINOv2 ViT-S/14 cls_sim and patch_sim against the raw 1080p original
  - Aggregate: mean ± std over 30 frames per cell.

Outputs:
  experiments/swinjscc_dji_dinov2_<timestamp>/metrics.csv
  experiments/swinjscc_dji_dinov2_<timestamp>/per_frame.csv
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
import torchvision.transforms as T

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMVIDEO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = SEMVIDEO_ROOT / "experiments"
# Upstream SwinJSCC code is not distributed with this repository (see README, "Third-party
# code"). Set SWINJSCC_DIR, or place it at <parent of this repository>/third_party/SwinJSCC
# (/third_party/SwinJSCC when the repository is mounted at /workspace in the Docker image).
SWINJSCC_DIR = Path(os.environ.get("SWINJSCC_DIR", REPO_ROOT / "third_party" / "SwinJSCC"))
sys.path.insert(0, str(SWINJSCC_DIR))
os.chdir(str(SWINJSCC_DIR))

from net.network import SwinJSCC  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The 1080p source video is not public (see README); point SEMVIDEO_SOURCE_VIDEO at your own.
SOURCE_VIDEO = Path(
    os.environ.get("SEMVIDEO_SOURCE_VIDEO", REPO_ROOT / "data" / "source_video.mp4")
)
DINO_RES = 518  # 14 * 37


def build_swinjscc(checkpoint: Path, image_dims=(3, 512, 512), C=96):
    model_name = "SwinJSCC_w/_SA"
    sj_args = SimpleNamespace(
        model=model_name, channel_type="awgn", C=str(C),
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
    # Filter out any tensors whose shape doesn't match the freshly-built model.
    # SwinJSCC's attn_mask buffers are resolution-dependent – they should be
    # taken from the new model, not the checkpoint, when crops differ.
    target = net.state_dict()
    filtered, dropped = {}, []
    for k, v in state.items():
        if k in target and target[k].shape == v.shape:
            filtered[k] = v
        elif k in target:
            dropped.append(f"{k}({tuple(v.shape)}->{tuple(target[k].shape)})")
        else:
            dropped.append(f"{k}[unexpected]")
    if dropped:
        print(f"[load] dropped {len(dropped)} shape-mismatched/unexpected tensors")
        # attn_mask is a resolution-derived buffer, so dropping it is benign; the
        # fresh model recomputes it. Anything else dropped is a LEARNED tensor
        # silently replaced by random init, which yields plausible-looking
        # garbage rather than an error. Make that case impossible to miss.
        suspect = [d for d in dropped if "attn_mask" not in d]
        if suspect:
            print(f"[load] *** WARNING: {len(suspect)} dropped tensors are NOT "
                  f"attn_mask buffers -- results are NOT trustworthy: {suspect[:8]}")
    missing, unexpected = net.load_state_dict(filtered, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    return net


@torch.no_grad()
def swinjscc_reconstruct(net, frame_1080_bgr, snr_db, codec_size=512):
    """Operator pipeline: 1080 -> codec_size -> SwinJSCC -> codec_size -> 1080."""
    h, w = frame_1080_bgr.shape[:2]
    small = cv2.resize(frame_1080_bgr, (codec_size, codec_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    recon, _, _, _, _ = net(x, given_SNR=snr_db, given_rate=None)
    recon = recon.clamp(0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()
    rec_bgr = cv2.cvtColor((recon * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    rec_1080 = cv2.resize(rec_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    return rec_1080


class DINOv2Extractor:
    def __init__(self):
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", verbose=False,
        ).eval().to(DEVICE)
        self.tx = T.Compose([
            T.ToTensor(),
            T.Resize((DINO_RES, DINO_RES), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def features(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = self.tx(rgb).unsqueeze(0).to(DEVICE)
        out = self.model.forward_features(x)
        return out["x_norm_clstoken"].squeeze(0), out["x_norm_patchtokens"].squeeze(0)


def cossim(a, b):
    if a.dim() == 1:
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    return float(F.cosine_similarity(a, b, dim=-1).mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="path to SwinJSCC DJI checkpoint (best.pt or .model)")
    ap.add_argument("--label", type=str, default="SwinJSCC_DJI",
                    help="row label for the metrics CSV")
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--snrs", type=int, nargs="+", default=[0, 5, 10, 15, 20])
    ap.add_argument("--codec-size", type=int, default=512)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = str(EXPERIMENTS_DIR / f"swinjscc_dji_dinov2_{ts}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] out: {out_dir}")
    print(f"[setup] checkpoint: {args.checkpoint}")

    print("[load] DINOv2 ViT-S/14...")
    dino = DINOv2Extractor()
    print("[load] SwinJSCC...")
    net = build_swinjscc(Path(args.checkpoint))

    cap = cv2.VideoCapture(str(SOURCE_VIDEO))
    if not cap.isOpened():
        print(f"[error] cannot open {SOURCE_VIDEO}")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(10850, min(11800, total - 1), args.n_frames, dtype=int).tolist()
    raw_frames = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if ok:
            raw_frames.append((int(fi), f))
    cap.release()
    print(f"[data] sampled {len(raw_frames)} frames")

    print("[ref ] DINOv2 features on raw 1080 frames...")
    ref = {}
    for fi, frame in raw_frames:
        ref[fi] = dino.features(frame)

    per_frame_path = out_dir / "per_frame.csv"
    summary_path = out_dir / "metrics.csv"

    with open(per_frame_path, "w", newline="") as fp:
        pw = csv.writer(fp)
        pw.writerow(["label", "channel", "snr", "frame", "cls_sim", "patch_sim"])

        with open(summary_path, "w", newline="") as fp2:
            sw = csv.writer(fp2)
            sw.writerow(["label", "channel", "snr",
                         "cls_mean", "cls_std", "patch_mean", "patch_std"])

            channel = "awgn"
            for snr in args.snrs:
                cls_vals, patch_vals = [], []
                for fi, frame in raw_frames:
                    rec = swinjscc_reconstruct(net, frame, snr, codec_size=args.codec_size)
                    cls_r, patch_r = dino.features(rec)
                    cls_o, patch_o = ref[fi]
                    c = cossim(cls_o, cls_r)
                    p = cossim(patch_o, patch_r)
                    cls_vals.append(c)
                    patch_vals.append(p)
                    pw.writerow([args.label, channel, snr, fi,
                                 round(c, 5), round(p, 5)])
                cm, cs = float(np.mean(cls_vals)), float(np.std(cls_vals))
                pm, ps = float(np.mean(patch_vals)), float(np.std(patch_vals))
                sw.writerow([args.label, channel, snr,
                             f"{cm:.5f}", f"{cs:.5f}",
                             f"{pm:.5f}", f"{ps:.5f}"])
                print(f"  {channel:8s} snr={snr:>2d} cls={cm:.4f}±{cs:.4f} "
                      f"patch={pm:.4f}±{ps:.4f}", flush=True)

    print(f"\n[saved] {summary_path}")
    print(f"[saved] {per_frame_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
