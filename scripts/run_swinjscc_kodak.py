"""Reproduce SwinJSCC paper numbers on Kodak.

Loads their `SwinJSCC w/ SA` AWGN HR model (C=96, fixed rate, multiple SNR train)
and evaluates on the Kodak 24-image set across SNR ∈ {1,4,7,10,13}.

Outputs:
  - experiments/swinjscc_kodak_2026_05_03/results.csv
  - terminal table with PSNR + MS-SSIM per SNR

Acceptance: ~31 dB at AWGN SNR=10, CBR=1/16 (matches Yang et al. 2024).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from glob import glob
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

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
from loss.distortion import MS_SSIM  # noqa: E402


def build_config(model_name, image_dims=(3, 768, 512), C=96, model_size="base"):
    """Build a config matching SwinJSCC's main.py for the DIV2K (HR) regime."""
    cfg = SimpleNamespace()
    cfg.seed = 42
    cfg.pass_channel = True
    cfg.CUDA = True
    cfg.device = torch.device("cuda:0")
    cfg.norm = False
    cfg.print_step = 100
    cfg.plot_step = 10000
    cfg.normalize = False
    cfg.learning_rate = 1e-4
    cfg.image_dims = image_dims
    cfg.batch_size = 1
    cfg.downsample = 4
    cfg.logger = None

    # Per main.py: w/o_SAandRA + w/_SA use fixed C; w/_RA + w/_SAandRA use Rate ModNet (C=None)
    if model_name in ("SwinJSCC_w/o_SAandRA", "SwinJSCC_w/_SA"):
        channel_number = C
    else:
        channel_number = None

    encoder = dict(
        model=model_name,
        img_size=(image_dims[1], image_dims[2]),
        patch_size=2,
        in_chans=3,
        embed_dims=[128, 192, 256, 320],
        depths=[2, 2, 6, 2],
        num_heads=[4, 6, 8, 10],
        C=channel_number,
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )
    decoder = dict(
        model=model_name,
        img_size=(image_dims[1], image_dims[2]),
        embed_dims=[320, 256, 192, 128],
        depths=[2, 6, 2, 2],
        num_heads=[10, 8, 6, 4],
        C=channel_number,
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )
    cfg.encoder_kwargs = encoder
    cfg.decoder_kwargs = decoder
    return cfg


def kodak_iterator(kodak_dir: Path):
    paths = sorted(glob(str(kodak_dir / "*.png")))
    to_tensor = T.ToTensor()
    for p in paths:
        img = Image.open(p).convert("RGB")
        # Crop to multiple of 128
        w, h = img.size
        nw = w - w % 128
        nh = h - h % 128
        img = img.crop((0, 0, nw, nh))
        x = to_tensor(img)  # (3, H, W) in [0,1]
        yield Path(p).name, x.unsqueeze(0)  # (1, 3, H, W)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            SWINJSCC_DIR / "checkpoint" / "SwinJSCC" / "SwinJSCC w_ SA"
            / "SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model"
        ),
    )
    ap.add_argument("--kodak", type=str, default=str(REPO_ROOT / "data" / "kodak"))
    ap.add_argument("--snr-list", type=str, default="1,4,7,10,13")
    ap.add_argument("--C", type=int, default=96)
    ap.add_argument("--out", type=str, default=str(
        EXPERIMENTS_DIR / "swinjscc_kodak_2026_05_03" / "results.csv"))
    ap.add_argument("--model", type=str, default="SwinJSCC_w/_SA",
                    choices=["SwinJSCC_w/o_SAandRA", "SwinJSCC_w/_SA",
                             "SwinJSCC_w/_RA", "SwinJSCC_w/_SAandRA"])
    ap.add_argument("--channel-type", type=str, default="awgn",
                    choices=["awgn", "rayleigh"])
    args_ns = ap.parse_args()

    out_csv = Path(args_ns.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    snr_list = [int(s) for s in args_ns.snr_list.split(",")]

    # Args required by SwinJSCC.__init__
    sj_args = SimpleNamespace(
        model=args_ns.model,
        channel_type=args_ns.channel_type,
        C=str(args_ns.C),
        multiple_snr=",".join(str(s) for s in snr_list),
        distortion_metric="MSE",
        trainset="DIV2K",
        logger=None,
    )

    # Build the model at the original training resolution (256x256) so attn_mask
    # buffer shapes match the checkpoint, then call update_resolution() to adapt.
    cfg = build_config(args_ns.model, image_dims=(3, 256, 256), C=args_ns.C)

    print(f"[setup] model={sj_args.model}, channel={sj_args.channel_type}, "
          f"C={sj_args.C}, snrs={snr_list}")
    print(f"[setup] checkpoint: {args_ns.checkpoint}")
    if not Path(args_ns.checkpoint).is_file():
        print(f"[error] checkpoint not found: {args_ns.checkpoint}")
        return 1

    net = SwinJSCC(sj_args, cfg).cuda().eval()

    state = torch.load(args_ns.checkpoint, map_location="cuda:0", weights_only=False)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] missing keys (first 5): {list(missing)[:5]}")
    if unexpected:
        print(f"[load] unexpected keys (first 5): {list(unexpected)[:5]}")
    print(f"[load] loaded {len(state)} tensors from checkpoint")

    ssim_metric = MS_SSIM(data_range=1.0, levels=4, channel=3).cuda()

    # Header
    rows = []
    print("\n  SNR  | PSNR (dB) | MS-SSIM (1-x)| CBR")
    print("-" * 50)

    with torch.no_grad():
        for snr in snr_list:
            psnrs, msssims, cbrs = [], [], []
            for name, x in kodak_iterator(Path(args_ns.kodak)):
                x = x.cuda()
                recon, cbr, snr_used, mse, _ = net(x, given_SNR=snr, given_rate=args_ns.C)
                recon = recon.clamp(0.0, 1.0)
                # PSNR computed on [0, 255]
                if mse.item() > 0:
                    psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
                    psnrs.append(psnr)
                msssim = 1 - ssim_metric(x, recon).mean().item()
                msssims.append(msssim)
                cbrs.append(cbr)
                rows.append({
                    "snr": snr,
                    "image": name,
                    "psnr": round(psnr, 3),
                    "msssim": round(msssim, 5),
                    "cbr": round(cbr, 5),
                })
            avg_psnr = float(np.mean(psnrs))
            avg_mss = float(np.mean(msssims))
            avg_cbr = float(np.mean(cbrs))
            print(f"  {snr:>3} | {avg_psnr:>9.3f} | {avg_mss:>13.5f}| {avg_cbr:>.5f}")

    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["snr", "image", "psnr", "msssim", "cbr"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[saved] {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
