"""Generate PyTorch reference outputs for the Jetson benchmark package.

Produces:
    - sample_images/dji_*.png (5 DJI val frames, 512x512 PNG)
    - sample_images/dji_*_pt_ref_snr10.npz (PyTorch reconstruction + latent at SNR=10)

Bundled with the exported ONNX graphs so that TensorRT and PyTorch can be compared
end-to-end on the same inputs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMVIDEO_ROOT = Path(__file__).resolve().parents[1]
# Upstream SwinJSCC code is not distributed with this repository (see README, "Third-party
# code"). Set SWINJSCC_DIR, or place it at <parent of this repository>/third_party/SwinJSCC
# (/third_party/SwinJSCC when the repository is mounted at /workspace in the Docker image).
SWINJSCC_DIR = Path(os.environ.get("SWINJSCC_DIR", REPO_ROOT / "third_party" / "SwinJSCC"))
sys.path.insert(0, str(SEMVIDEO_ROOT / "src"))
sys.path.insert(0, str(SWINJSCC_DIR))
os.chdir(str(SWINJSCC_DIR))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_swinjscc(checkpoint: Path, image_dims=(3, 512, 512), C=96) -> nn.Module:
    """Same shape-filtered loader used in scripts/export_tensorrt.py."""
    from net.network import SwinJSCC

    sj_args = SimpleNamespace(
        model="SwinJSCC_w/_SA", channel_type="awgn", C=str(C),
        multiple_snr="10", distortion_metric="MSE", trainset="DIV2K", logger=None,
    )
    cfg = SimpleNamespace(
        seed=42, pass_channel=True, CUDA=torch.cuda.is_available(),
        device=DEVICE, norm=False, print_step=100, plot_step=10000,
        normalize=False, learning_rate=1e-4, image_dims=image_dims,
        batch_size=1, downsample=4, logger=None,
    )
    cfg.encoder_kwargs = dict(
        model="SwinJSCC_w/_SA", img_size=(image_dims[1], image_dims[2]),
        patch_size=2, in_chans=3,
        embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2],
        num_heads=[4, 6, 8, 10], C=C, window_size=8, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True,
    )
    cfg.decoder_kwargs = dict(
        model="SwinJSCC_w/_SA", img_size=(image_dims[1], image_dims[2]),
        embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2],
        num_heads=[10, 8, 6, 4], C=C, window_size=8, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True,
    )
    net = SwinJSCC(sj_args, cfg).to(DEVICE).eval()
    state = torch.load(str(checkpoint), map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    target = net.state_dict()
    loaded = {k: v for k, v in state.items()
              if k in target and v.shape == target[k].shape}
    for k, v in state.items():
        stripped = k.replace("channel.base_channel.", "channel.")
        if stripped in target and v.shape == target[stripped].shape:
            loaded[stripped] = v
    target.update(loaded)
    net.load_state_dict(target, strict=False)
    print(f"[load] {len(loaded)}/{len(target)} keys")
    return net


def process_corpus(net: nn.Module, val_dir: Path, n_samples: int, out_dir: Path,
                   prefix: str, snr_db: float, start_index: int = 0) -> int:
    """Process n_samples evenly-spaced images from val_dir into out_dir.

    Returns the next index to use for downstream prefixes.
    """
    frames = sorted(p for p in val_dir.rglob("*")
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not frames:
        print(f"[warn] no frames in {val_dir} – skipping {prefix}")
        return start_index
    idx = np.linspace(0, len(frames) - 1, n_samples, dtype=int)
    selected = [frames[int(i)] for i in idx]
    print(f"[{prefix}] selected {len(selected)} of {len(frames)} frames from {val_dir}")

    for j, p in enumerate(selected):
        i = start_index + j
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  skip unreadable: {p}")
            continue
        h, w = bgr.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        crop = cv2.resize(bgr[y0:y0 + side, x0:x0 + side], (512, 512),
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        out_png = out_dir / f"{prefix}_{i:02d}_{p.stem[:40]}.png"
        cv2.imwrite(str(out_png), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        x = (torch.from_numpy(rgb).float().permute(2, 0, 1)
             .unsqueeze(0).to(DEVICE)) / 255.0

        with torch.no_grad():
            # Single full-pipeline forward: encoder → channel → decoder.
            # The reference reconstruction reflects this single trajectory
            # (i.e. a specific draw of channel noise). The Jetson-side stage 6
            # re-draws AWGN with its own RNG and compares the resulting recon
            # to ours; a perfect-match PSNR is not expected (different noise
            # samples), but the *distribution* of error magnitudes should be
            # comparable across runs. We deliberately do NOT bundle the clean
            # pre-channel latent because it would be a different noise draw
            # than the one that produced this recon.
            rec, cbr, snr_used, mse, _ = net(x, given_SNR=snr_db,
                                              given_rate=None)
            rec = rec.clamp(0, 1)

        rec_np = (rec.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        out_npz = out_dir / f"{prefix}_{i:02d}_{p.stem[:40]}_pt_ref_snr{int(snr_db)}.npz"
        np.savez_compressed(out_npz,
                            input_rgb_u8=rgb,
                            recon_rgb_u8=rec_np,
                            snr_db=float(snr_db),
                            corpus=prefix)
        print(f"  [{i:02d}] {p.name[:45]:45s} -> {out_png.name}")

    return start_index + len(selected)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dji-val-dir", type=Path, default=None,
                    help="Path to DJI val frames (in-domain UAV)")
    ap.add_argument("--visdrone-dir", type=Path, default=None,
                    help="Path to VisDrone images (cross-corpus UAV)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-dji", type=int, default=20)
    ap.add_argument("--n-visdrone", type=int, default=10)
    ap.add_argument("--snr-db", type=float, default=10.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] checkpoint: {args.checkpoint}")
    print(f"[setup] out: {args.out_dir}")
    print(f"[setup] n_dji={args.n_dji}  n_visdrone={args.n_visdrone}")

    net = build_swinjscc(args.checkpoint)
    torch.manual_seed(42)

    idx = 0
    if args.dji_val_dir is not None and args.n_dji > 0:
        idx = process_corpus(net, args.dji_val_dir, args.n_dji,
                              args.out_dir, "dji", args.snr_db, idx)
    if args.visdrone_dir is not None and args.n_visdrone > 0:
        idx = process_corpus(net, args.visdrone_dir, args.n_visdrone,
                              args.out_dir, "visdrone", args.snr_db, idx)

    print(f"[ok] {idx} reference samples total in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
