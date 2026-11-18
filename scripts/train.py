"""Training script of the compact convolutional JSCC codec, with W&B logging.

Supports:
- Model: DeepJSCC (the compact convolutional codec of the paper)
- Channels: AWGN, Rayleigh fading, mixed (both), barrage / pulsed jamming, and optional
  training-time symbol erasure
- Losses: MSE; perceptual_yolo (MSE + MS-SSIM + frozen-detector feature L1)
- Tracking: Weights & Biases + CSV + image snapshots

Usage (from the repository root):
    python scripts/train.py --model deepjscc --channel awgn --loss mse --epochs 50
    python scripts/train.py --model deepjscc --channel mixed --loss perceptual_yolo --epochs 100
"""

import argparse
import csv
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np

from semvideo.channel.awgn import AWGNChannel
from semvideo.channel.fading import RayleighFadingChannel
from semvideo.channel.jamming import JammingChannel
from semvideo.metrics.psnr_ssim import psnr, ssim
from semvideo.metrics.lpips_fid import LPIPSMetric
from semvideo.models.deep_jscc import DeepJSCC

# ── W&B setup ────────────────────────────────────────────────────────
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ── Dataset ──────────────────────────────────────────────────────────
class FrameDataset(Dataset):
    """Fast dataset loading pre-extracted JPG frames from one or more folders.

    ``root`` may be a single path string or a list of paths. When multiple
    paths are given, frames from every folder are pooled into a single index.
    Used to combine the DJI v3 and short v2 datasets for training while
    keeping each source's directory structure independent on disk.
    """

    def __init__(
        self,
        root: str | list[str] | list[Path],
        augment: bool = True,
        img_size: int = 256,
    ) -> None:
        if isinstance(root, (str, Path)):
            roots = [Path(root)]
        else:
            roots = [Path(r) for r in root]
        self.roots = roots
        files: list[Path] = []
        for r in roots:
            if not r.exists():
                continue
            files.extend(
                p for p in r.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        self.files = sorted(files)
        self.augment = augment
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = cv2.imread(str(self.files[idx]))
        if img is None:
            # Audit 2026-06-03 finding: corrupt JPGs (zero-byte from interrupted
            # extraction) crash worker with obscure cvtColor error. Substitute a
            # neighbouring frame to keep the run alive; log via print to avoid
            # logger setup in workers.
            for j in range(1, 8):
                alt = self.files[(idx + j) % len(self.files)]
                img = cv2.imread(str(alt))
                if img is not None:
                    print(f"[FrameDataset] corrupt {self.files[idx].name}, "
                          f"using {alt.name}")
                    break
            if img is None:
                # All neighbours bad too – return a neutral grey frame so the
                # batch still composes.
                img = np.full((self.img_size, self.img_size, 3), 128, np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Audit 2026-06-03 critical fix: honour self.img_size. Previously the
        # --img-size flag never reached the dataloader, so JPGs were returned at
        # their native resolution (e.g. 512 for frames_512_v1) while the model
        # was built for args.img_size (e.g. 256) – a silent train/eval domain
        # mismatch that invalidated cls_sim comparisons with P49/P68.
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            # Data-audit 2026-07-24: make the resample LOUD (once) -- a
            # frames_512_v1 folder consumed with the default --img-size 128
            # silently downsampled 4x and confounded resolution studies.
            if not getattr(FrameDataset, "_resize_warned", False):
                FrameDataset._resize_warned = True
                print(
                    f"[FrameDataset] WARNING: native {img.shape[1]}x{img.shape[0]} "
                    f"frames resampled to --img-size {self.img_size} "
                    f"(intentional? pass --img-size {img.shape[0]} to keep native)"
                )
            img = cv2.resize(
                img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
            )

        if self.augment:
            if random.random() > 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-10, 10)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(
                    np.uint8
                )

        return torch.from_numpy(img).float().permute(2, 0, 1) / 255.0


# ── Helpers ──────────────────────────────────────────────────────────
def build_model(name: str, device: str, channel_ratio: float = 1 / 6) -> nn.Module:
    """Create the codec by name (this release ships the compact convolutional codec only)."""
    if name == "deepjscc":
        # DeepJSCC is fully convolutional; img_size doesn't affect arch.
        return DeepJSCC(channel_ratio=channel_ratio, base_channels=64).to(device)
    raise ValueError(f"Unknown model: {name}")


def build_channel(
    channel_type: str,
    snr: float,
    device: str,
    jnr_db: float = 5.0,
    duty_cycle: float = 0.5,
) -> nn.Module:
    """Create channel by type and SNR.

    Channel types:
        awgn / rayleigh             – single-channel training
        jam_barrage / jam_pulsed    – jamming with the given JNR (background SNR `snr`)
        mixed                       – uniform random over {AWGN, Rayleigh}
        mixed_jam                   – uniform random over {AWGN, Rayleigh, barrage, pulsed}
    """
    if channel_type == "awgn":
        return AWGNChannel(snr_db=snr).to(device)
    elif channel_type == "rayleigh":
        return RayleighFadingChannel(snr_db=snr).to(device)
    elif channel_type == "jam_barrage":
        return JammingChannel(
            jammer_type="barrage", jnr_db=jnr_db, snr_db=snr
        ).to(device)
    elif channel_type == "jam_pulsed":
        return JammingChannel(
            jammer_type="pulsed", jnr_db=jnr_db, duty_cycle=duty_cycle, snr_db=snr
        ).to(device)
    elif channel_type == "mixed":
        if random.random() < 0.5:
            return AWGNChannel(snr_db=snr).to(device)
        else:
            return RayleighFadingChannel(snr_db=snr).to(device)
    elif channel_type == "mixed_jam":
        # Sample uniformly across AWGN / Rayleigh / barrage / pulsed each step.
        # Random JNR in [0, 10] dB to expose the model to jammer power variability.
        roll = random.random()
        if roll < 0.25:
            return AWGNChannel(snr_db=snr).to(device)
        if roll < 0.50:
            return RayleighFadingChannel(snr_db=snr).to(device)
        sampled_jnr = random.uniform(0.0, 10.0)
        if roll < 0.75:
            return JammingChannel(
                jammer_type="barrage", jnr_db=sampled_jnr, snr_db=snr
            ).to(device)
        return JammingChannel(
            jammer_type="pulsed",
            jnr_db=sampled_jnr,
            duty_cycle=random.uniform(0.3, 0.7),
            snr_db=snr,
        ).to(device)
    else:
        raise ValueError(f"Unknown channel: {channel_type}")


def get_lr_with_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))


def check_gradients(model: nn.Module) -> dict[str, float]:
    """Check gradient norms for key components."""
    grad_norms: dict[str, list[float]] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            component = name.split(".")[0]
            grad_norms.setdefault(component, []).append(
                param.grad.data.norm(2).item()
            )
    return {k: float(np.mean(v)) for k, v in grad_norms.items()}


# ── Validation ───────────────────────────────────────────────────────
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    lpips_model: LPIPSMetric,
    channel_types: list[str],
) -> dict:
    """Run validation across SNR values and channel types.

    Uses MATCHED SNR conditioning: the model is told the same SNR the channel
    actually delivers. (A previous version hardwired the conditioning to 10 dB
    for every sweep point "to avoid broken boundary embeddings at SNR=0/20" --
    that hid any endpoint-embedding degradation and made the SNR-quality curve a
    fixed-conditioning curve rather than the operating curve. If endpoint quality
    drops under matched conditioning, that is a real finding to report, not hide.
    Note: the headline paper metrics come from the eval scripts, which already
    pass given_SNR = actual SNR; this only fixes training-time monitoring.)
    """
    model.eval()
    snr_values = [0, 5, 10, 15, 20]
    results: dict = {}

    with torch.no_grad():
        for ch_type in channel_types:
            results[ch_type] = {
                snr: {"psnr": [], "ssim": [], "lpips": []} for snr in snr_values
            }
            for batch in val_loader:
                images = batch.to(device)
                for snr in snr_values:
                    if ch_type == "awgn":
                        ch = AWGNChannel(snr_db=float(snr)).to(device)
                    elif ch_type == "rayleigh":
                        ch = RayleighFadingChannel(snr_db=float(snr)).to(device)
                    elif ch_type == "jam_barrage":
                        # Default JNR=10 dB (typical operational scenario).
                        ch = JammingChannel(
                            snr_db=float(snr), jnr_db=10.0,
                            jammer_type="barrage",
                        ).to(device)
                    elif ch_type == "jam_pulsed":
                        ch = JammingChannel(
                            snr_db=float(snr), jnr_db=10.0,
                            jammer_type="pulsed", duty_cycle=0.5,
                        ).to(device)
                    else:
                        raise ValueError(f"unknown val channel type: {ch_type}")
                    recon = model(images, channel=ch, snr_db=float(snr))
                    recon = recon.clamp(0, 1)
                    results[ch_type][snr]["psnr"].append(psnr(recon, images))
                    results[ch_type][snr]["ssim"].append(ssim(recon, images))
                    # Downsample for LPIPS: VGG16 at 512x512 batch=16 dominates
                    # validation time.
                    if recon.shape[-1] > 128:
                        rf = nn.functional.interpolate(recon, size=(128, 128),
                                                       mode="bilinear", align_corners=False)
                        if_ = nn.functional.interpolate(images, size=(128, 128),
                                                        mode="bilinear", align_corners=False)
                    else:
                        rf, if_ = recon, images
                    results[ch_type][snr]["lpips"].append(
                        lpips_model(rf, if_).item()
                    )

    for ch_type in channel_types:
        for snr in snr_values:
            for metric in ["psnr", "ssim", "lpips"]:
                results[ch_type][snr][metric] = float(
                    np.mean(results[ch_type][snr][metric])
                )
    return results


# ── Image logging ────────────────────────────────────────────────────
def _gaussian_blur(img: torch.Tensor, kernel_size: int = 15, sigma: float = 4.0) -> torch.Tensor:
    """Apply Gaussian blur to an image tensor (B, C, H, W)."""
    import torch.nn.functional as F

    k = kernel_size
    gauss = torch.arange(k, dtype=img.dtype, device=img.device) - k // 2
    gauss = torch.exp(-gauss ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).expand(3, 1, k, k)
    return F.conv2d(img, kernel_2d, padding=k // 2, groups=3)


def _select_diverse_samples(val_loader: DataLoader, n_samples: int = 6) -> list[torch.Tensor]:
    """Select visually diverse samples spread across the validation set."""
    all_images = []
    for batch in val_loader:
        all_images.append(batch)
    all_images = torch.cat(all_images, dim=0)

    total = all_images.shape[0]
    # Pick evenly spaced indices across the dataset for diversity
    indices = [int(i * total / n_samples) for i in range(n_samples)]
    return [all_images[i : i + 1] for i in indices]


def log_images(
    model: nn.Module,
    val_loader: DataLoader,
    epoch: int,
    exp_dir: Path,
    device: str,
    snr_values: list[int],
    use_wandb: bool = False,
) -> None:
    """Save reconstruction comparison grid with diverse samples and blur reference.

    Layout:
      Columns: Original | Blurred (ref) | SNR=0dB | SNR=5dB | SNR=10dB | SNR=15dB | SNR=20dB
      Rows: Different validation images (diverse scenes)

    Each cell shows PSNR/SSIM metrics. Blur reference helps stakeholders
    see the quality floor that JSCC must beat.
    """
    model.eval()
    epoch_dir = exp_dir / "images" / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    samples = _select_diverse_samples(val_loader, n_samples=6)
    n_cols = len(snr_values) + 2  # original + blur + SNR columns
    n_rows = len(samples)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))

    # Column headers
    col_labels = ["Original", "Blurred\n(reference)"] + [f"JSCC\nSNR={s}dB" for s in snr_values]

    with torch.no_grad():
        for row, x in enumerate(samples):
            x = x.to(device)
            orig_np = x.squeeze(0).cpu().permute(1, 2, 0).numpy()

            # Col 0: Original
            axes[row, 0].imshow(orig_np)
            if row == 0:
                axes[row, 0].set_title(col_labels[0], fontsize=10, fontweight="bold")
            axes[row, 0].set_ylabel(f"Sample {row + 1}", fontsize=9, rotation=0, labelpad=40)
            axes[row, 0].axis("off")

            # Col 1: Gaussian blur reference (what "bad quality" looks like)
            blurred = _gaussian_blur(x, kernel_size=15, sigma=4.0).clamp(0, 1)
            blur_np = blurred.squeeze(0).cpu().permute(1, 2, 0).numpy()
            bp, bs = psnr(blurred, x), ssim(blurred, x)
            axes[row, 1].imshow(blur_np)
            if row == 0:
                axes[row, 1].set_title(col_labels[1], fontsize=10, fontweight="bold")
            axes[row, 1].set_xlabel(f"{bp:.1f}dB / {bs:.3f}", fontsize=8)
            axes[row, 1].axis("off")

            # Cols 2+: JSCC reconstructions at each SNR
            for col, snr in enumerate(snr_values):
                ch = AWGNChannel(snr_db=float(snr)).to(device)
                recon = model(x, channel=ch, snr_db=10.0).clamp(0, 1)
                rn = recon.squeeze(0).cpu().permute(1, 2, 0).numpy()
                p, s = psnr(recon, x), ssim(recon, x)

                ax = axes[row, col + 2]
                ax.imshow(rn)
                if row == 0:
                    ax.set_title(col_labels[col + 2], fontsize=10, fontweight="bold")
                ax.set_xlabel(f"{p:.1f}dB / {s:.3f}", fontsize=8)
                ax.axis("off")

    fig.suptitle(
        f"Epoch {epoch} -- Reconstruction Quality Across SNR\n"
        f"(metrics: PSNR / SSIM -- higher is better)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    # Save overview grid
    grid_path = epoch_dir / "grid.png"
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Save individual sample comparisons as separate files
    for i, x in enumerate(samples):
        x = x.to(device)
        fig2, ax2 = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))
        orig_np = x.squeeze(0).cpu().permute(1, 2, 0).numpy()

        ax2[0].imshow(orig_np)
        ax2[0].set_title("Original", fontsize=9)
        ax2[0].axis("off")

        blurred = _gaussian_blur(x, kernel_size=15, sigma=4.0).clamp(0, 1)
        ax2[1].imshow(blurred.squeeze(0).cpu().permute(1, 2, 0).numpy())
        bp = psnr(blurred, x)
        ax2[1].set_title(f"Blur {bp:.1f}dB", fontsize=9)
        ax2[1].axis("off")

        for col, snr in enumerate(snr_values):
            ch = AWGNChannel(snr_db=float(snr)).to(device)
            recon = model(x, channel=ch, snr_db=float(snr)).clamp(0, 1)
            p = psnr(recon, x)
            ax2[col + 2].imshow(recon.squeeze(0).cpu().detach().permute(1, 2, 0).numpy())
            ax2[col + 2].set_title(f"SNR={snr} {p:.1f}dB", fontsize=9)
            ax2[col + 2].axis("off")

        plt.tight_layout()
        plt.savefig(epoch_dir / f"sample_{i + 1}.png", dpi=120, bbox_inches="tight")
        plt.close(fig2)

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"reconstructions": wandb.Image(str(grid_path))}, step=epoch)


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train JSCC models")
    parser.add_argument("--model", choices=["deepjscc"], default="deepjscc",
                        help="Codec to train: the compact convolutional codec of the paper.")
    parser.add_argument(
        "--channel",
        choices=["awgn", "rayleigh", "jam_barrage", "jam_pulsed", "mixed", "mixed_jam"],
        default="awgn",
    )
    parser.add_argument("--loss",
                        choices=["mse", "perceptual_yolo"],
                        default="mse",
                        help="mse | perceptual_yolo (MSE+MS-SSIM + YOLO-feature L1).")
    parser.add_argument("--yolo-feat-weight", type=float, default=0.5,
                        help="Weight on the DetectorFeatureLoss term when --loss perceptual_yolo. "
                             "Adapt-ICMH-equivalent default; halve if PSNR collapses below 22 dB.")
    parser.add_argument("--yolo-feat-warmup", type=int, default=3,
                        help="Pixel-only warmup epochs before adding YOLO-feat loss. "
                             "SR4IR uses 3 – without this, training is unstable.")
    parser.add_argument("--yolo-weights", default="yolov8m.pt",
                        help="YOLOv8 weights for feature distillation (defaults to COCO-yolov8m).")
    parser.add_argument("--channel-ratio", type=float, default=1/6,
                        help="DeepJSCC channel symbol ratio (latent-channels / 256). "
                             "Default 1/6 = 18.3x compression. 1/3 = 6x compression "
                             "(more bandwidth, easier task preservation).")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--jnr-db", type=float, default=5.0,
                        help="Jammer-to-noise ratio (dB) for jam_* channels")
    parser.add_argument("--duty-cycle", type=float, default=0.5,
                        help="Pulsed jammer duty cycle (0-1)")
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--keep-warmup-on-init", action="store_true",
                        help="Keep LR warmup when --init-from is used. "
                        "By default warmup is skipped on fine-tunes.")
    parser.add_argument("--snr-fixed", type=float, default=None,
                        help="If set, train at this fixed SNR (no random sampling).")
    parser.add_argument("--data-dir",
                        default="data/frames",
                        help="Dataset root(s). Comma-separated for multi-source. "
                             "Each root must contain train/ and val/ subdirs.")
    parser.add_argument("--img-size", type=int, default=128,
                        help="Spatial input size (HxW); must match the size of pre-extracted JPGs in data-dir.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training (torch.amp). ~2x VRAM reduction + speedup on tensor cores.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers. Default 4 (overlaps CPU JPG-decode with GPU compute, "
                             "~24%% epoch speedup at 256x256). Set to 0 for sync if debugging.")
    parser.add_argument("--exp-name", default="")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed for reproducibility (Python, NumPy, "
                             "torch CPU + CUDA). Set to negative to disable seeding.")
    parser.add_argument("--cudnn-deterministic", action="store_true",
                        help="Set torch.backends.cudnn.deterministic=True. Slower but "
                             "bit-exact under the same seed.")
    parser.add_argument("--erasure-p-max", type=float, default=0.0,
                        help="Training-time symbol erasure: p ~ U[0, this] per batch. "
                             "0 disables erasure-aug. Use 0.4-0.5 for stage-3 deployments.")
    parser.add_argument("--resume", default="", help="Checkpoint path to resume from")
    parser.add_argument("--init-from", default="",
                        help="Warm-start model weights from a checkpoint without loading "
                             "optimizer state (fine-tuning).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    # Force unbuffered output (W&B can interfere with stdout buffering)
    sys.stdout.reconfigure(line_buffering=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Reproducibility ─────────────────────────────────────────────
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        if args.cudnn_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Seeded RNGs with {args.seed} (cudnn.deterministic={args.cudnn_deterministic})")

    # ── Experiment setup ─────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.exp_name or f"{args.channel}_{args.model}_{args.loss}"
    exp_dir = Path("experiments") / f"{name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "images").mkdir(exist_ok=True)

    # Resolve git commit (used by both config snapshot and W&B). Falls back to
    # SEMVIDEO_GIT_SHA env var when git isn't available – useful inside Docker
    # containers where .git may not be mounted but the caller knows the SHA.
    import subprocess as _sp
    try:
        git_sha = _sp.check_output(
            ["git", "rev-parse", "HEAD"], stderr=_sp.DEVNULL, cwd=Path(__file__).parent
        ).decode().strip()
    except Exception:
        git_sha = os.environ.get("SEMVIDEO_GIT_SHA", "unknown")

    # Per-experiment config snapshot for reproducibility (independent of W&B).
    import json as _json
    config_snapshot = vars(args).copy()
    config_snapshot["git_sha"] = git_sha
    config_snapshot["timestamp"] = timestamp
    config_snapshot["torch_version"] = torch.__version__
    config_snapshot["cuda_available"] = torch.cuda.is_available()
    config_snapshot["device"] = device
    with open(exp_dir / "config.json", "w") as f:
        _json.dump(config_snapshot, f, indent=2, default=str)

    # ── W&B ──────────────────────────────────────────────────────────
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        try:
            wb_config = vars(args).copy()
            wb_config["git_sha"] = git_sha
            wb_config["exp_dir"] = str(exp_dir)
            os.environ.setdefault("WANDB_INIT_TIMEOUT", "30")
            # Project defaults to "semvideo" (one canonical project, no
            # scattered duplicates). WANDB_PROJECT env
            # var still respected for one-off overrides, but defaults to
            # the canonical name.
            wb_project = os.environ.get("WANDB_PROJECT", "semvideo")
            # Run name: prefer args.exp_name; fall back to a self-describing slug.
            run_name = args.exp_name or f"{args.model}_{args.channel}_{args.loss}_{timestamp}"
            # Tag set is consistent across runs so the dashboard filters
            # are stable: model, channel, loss, and the initialisation mode.
            tags = sorted({
                args.model, args.channel, args.loss,
                "from-scratch" if not args.init_from else "init-from",
            })
            wandb.init(
                project=wb_project,
                name=run_name,
                config=wb_config,
                tags=tags,
                dir=str(exp_dir),
                settings=wandb.Settings(init_timeout=30),
                # Group runs by exp-name prefix so
                # related sweeps cluster on the dashboard.
                group=run_name.split("_")[0] if "_" in run_name else run_name,
            )
            print(f"W&B run: {wandb.run.url}")
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without W&B")
            use_wandb = False

    # ── Data ─────────────────────────────────────────────────────────
    data_roots = [r.strip() for r in args.data_dir.split(",") if r.strip()]
    train_dirs = [f"{r}/train" for r in data_roots]
    val_dirs = [f"{r}/val" for r in data_roots]
    train_ds = FrameDataset(train_dirs, augment=True, img_size=args.img_size)
    val_ds = FrameDataset(val_dirs, augment=False, img_size=args.img_size)
    print(f"Data roots ({len(data_roots)}): {data_roots}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    nw = args.num_workers

    # Audit 2026-06-03 fix: with num_workers>0, Python stdlib `random` (used by
    # FrameDataset augmentations) is NOT auto-seeded per worker; only torch is.
    # So `--seed 42 --cudnn-deterministic` previously did NOT yield bit-identical
    # metrics at nw>0 (the earlier reproducibility check was at nw=0). Re-seed
    # all three RNGs inside each worker.
    def _seed_worker(worker_id):  # noqa: ARG001 – id unused; torch seeds per-worker
        import random as _rnd
        s = torch.initial_seed() % (2**32)
        _rnd.seed(s)
        np.random.seed(s)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=(nw > 0),
        worker_init_fn=_seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0),
        worker_init_fn=_seed_worker,
    )

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    # When loading a fine-tune checkpoint via --init-from, the LR-warmup
    # schedule should NOT restart from zero (audit 2026-06-03 finding:
    # default --warmup-epochs=15 in fine-tune scripts trained at near-zero
    # LR for the whole budget, silently invalidating fine-tune results).
    # Skip warmup when init-from is supplied unless --keep-warmup-on-init
    # is explicitly requested.
    if args.init_from and not getattr(args, "keep_warmup_on_init", False):
        if warmup_steps > 0:
            print(
                f"[init-from] disabling LR warmup ({warmup_steps} -> 0 steps) "
                "to avoid fine-tune cold-restart"
            )
        warmup_steps = 0

    # ── Model ────────────────────────────────────────────────────────
    model = build_model(args.model, device, channel_ratio=args.channel_ratio)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | {params:,} params")

    if hasattr(model, "get_compression_stats"):
        comp = model.get_compression_stats()
        print(f"Compression: {comp['compression_ratio']:.1f}x ({comp['n_symbols']} symbols)")

    if args.resume and args.init_from:
        raise ValueError("Use either --resume or --init-from, not both.")

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        # Back-compat: older checkpoints used "model_state"; newer use
        # "model_state_dict". Also accept raw state_dict at top level.
        if isinstance(ckpt, dict):
            state = (
                ckpt.get("model_state_dict")
                or ckpt.get("model_state")
                or ckpt.get("model")
                or ckpt
            )
        else:
            state = ckpt
        load_result = model.load_state_dict(state, strict=False)
        print(
            f"Initialized weights from {args.init_from} "
            f"(missing={len(load_result.missing_keys)}, "
            f"unexpected={len(load_result.unexpected_keys)})"
        )
        if load_result.missing_keys:
            print(f"  Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"  Unexpected keys: {load_result.unexpected_keys}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.999),
    )

    # ── Loss ─────────────────────────────────────────────────────────
    lpips_model = LPIPSMetric(device=device).to(device)

    if args.loss == "perceptual_yolo":
        # SR4IR / Adapt-ICMH style objective: MSE + MS-SSIM + frozen-detector feature L1.
        # Loss is built lazily per-batch since it has both pixel and feature components.
        from semvideo.training.detector_feature_loss import DetectorFeatureLoss
        from semvideo.training.losses import ms_ssim_loss as _ms_ssim_fn
        yolo_feat_loss = DetectorFeatureLoss(
            weights_path=args.yolo_weights, device=device,
        )

        class _PerceptualYOLOWrap(nn.Module):
            def __init__(self, alpha, gamma, fw, warmup_steps):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.fw = fw
                self.warmup_steps = warmup_steps
                self.steps = 0

            def forward(self, pred, target):
                self.steps += 1
                mse = nn.functional.mse_loss(pred, target)
                msssim = _ms_ssim_fn(pred, target)
                if self.steps > self.warmup_steps:
                    feat = yolo_feat_loss(pred, target)
                    total = self.alpha * mse + self.gamma * msssim + self.fw * feat
                    return {"total": total, "mse": mse.detach(),
                            "ms_ssim": msssim.detach(), "yolo_feat": feat.detach()}
                else:
                    total = self.alpha * mse + self.gamma * msssim
                    return {"total": total, "mse": mse.detach(),
                            "ms_ssim": msssim.detach()}

        # This also overrides the LR warm-up set above; kept on purpose (paper, Section 3.7).
        warmup_steps = args.yolo_feat_warmup * steps_per_epoch
        criterion = _PerceptualYOLOWrap(alpha=1.0, gamma=0.15,
                                        fw=args.yolo_feat_weight,
                                        warmup_steps=warmup_steps)
        print(f"Loss: perceptual_yolo (MSE + MS-SSIM + {args.yolo_feat_weight}*YOLO-feat-L1; "
              f"warmup {args.yolo_feat_warmup} epochs of pixel-only)")
    else:
        criterion = None
        print("Loss: MSE")

    # ── Mixed precision ─────────────────────────────────────────────
    use_amp = args.amp and device.startswith("cuda")
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
        print("AMP (mixed precision): ENABLED")
    else:
        scaler = None

    val_channel_types = ["awgn"]
    if args.channel in ("rayleigh", "mixed", "mixed_jam"):
        val_channel_types.append("rayleigh")
    # Audit 2026-06-03 fix: jamming-trained runs (P80/P81, P82) silently saved
    # best.pt based on AWGN PSNR alone – the regime augmentation is HURTING.
    # Include jamming channels at validation so jam-trained runs select for
    # jamming robustness too.
    if args.channel in ("jam_barrage", "mixed_jam"):
        val_channel_types.append("jam_barrage")
    if args.channel in ("jam_pulsed", "mixed_jam"):
        val_channel_types.append("jam_pulsed")

    # ── Resume ───────────────────────────────────────────────────────
    start_epoch = 1
    best_psnr = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch - 1}, best PSNR: {best_psnr:.2f}")

    # ── CSV logging ──────────────────────────────────────────────────
    # We use a try/finally for the close because the training loop below is
    # very long; converting to a `with` block would indent ~170 lines and
    # bloat the diff. try/finally is functionally equivalent for resource
    # release on any path (normal exit, KeyboardInterrupt, OOM, exception).
    csv_file = open(exp_dir / "metrics.csv", "w", newline="")
    csv_w = csv.writer(csv_file)
    header = ["epoch", "train_loss", "train_psnr", "lr"]
    for ch_type in val_channel_types:
        for snr in [0, 5, 10, 15, 20]:
            header.extend([
                f"val_{ch_type}_psnr_{snr}",
                f"val_{ch_type}_ssim_{snr}",
                f"val_{ch_type}_lpips_{snr}",
            ])
    csv_w.writerow(header)
    csv_file.flush()

    # ── Training loop ────────────────────────────────────────────────
    global_step = (start_epoch - 1) * steps_per_epoch
    print(
        f"\n--- Training {args.epochs} epochs | {args.model} | "
        f"{args.channel} | {args.loss} ---\n"
    )

    # Register an atexit hook so csv_file is closed even on KeyboardInterrupt,
    # OOM, or exception – without needing a try/finally that would re-indent
    # the entire training loop body.
    import atexit
    atexit.register(lambda f=csv_file: (not f.closed) and f.close())

    best_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss, total_psnr, n = 0.0, 0.0, 0
        epoch_grads: dict[str, list[float]] = {}
        t_start = time.time()

        for batch in train_loader:
            images = batch.to(device)

            if args.snr_fixed is not None:
                snr = float(args.snr_fixed)
            else:
                snr = float(torch.empty(1).uniform_(args.snr_min, args.snr_max).item())
            ch = build_channel(
                args.channel, snr, device,
                jnr_db=args.jnr_db, duty_cycle=args.duty_cycle,
            )
            # Optional erasure-augmentation: compose ch ∘ BernoulliErasure.
            # ``--erasure-p-max`` > 0 enables training-time symbol erasure
            # with ``p ~ U[0, erasure_p_max]`` per batch. Forwards to the
            # underlying base channel first, then applies erasure to the
            # noisy symbols.
            if args.erasure_p_max > 0.0:
                from semvideo.channel.packet_loss import BernoulliErasureChannel as _BEC
                _base = ch
                _erase = _BEC(p=None, p_min=0.0, p_max=args.erasure_p_max).to(device)
                class _EraseWrap(nn.Module):
                    def __init__(self, base, erase):
                        super().__init__()
                        self.base = base
                        self.erase = erase
                    def forward(self, x, snr_db=None):
                        return self.erase(self.base(x, snr_db))
                ch = _EraseWrap(_base, _erase).to(device)

            lr = get_lr_with_warmup(global_step, total_steps, warmup_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                recon = model(images, channel=ch, snr_db=snr)
                # Audit 2026-06-03 fix: validation clamps recon to [0,1] but
                # training did NOT, so the detector-feature loss saw
                # out-of-range pixels under jamming or large-MSE batches and
                # could produce NaNs that taint the running loss average.
                # Use a soft (differentiable) clamp at boundary.
                recon = recon - F.relu(recon - 1.0) - F.relu(-recon)

                if criterion is not None:
                    loss_dict = criterion(recon, images)
                    loss = loss_dict["total"]
                else:
                    loss = nn.functional.mse_loss(recon, images)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if global_step % 10 == 0:
                grads = check_gradients(model)
                for k, v in grads.items():
                    epoch_grads.setdefault(k, []).append(v)

            total_loss += loss.item()
            total_psnr += psnr(recon.detach(), images.detach())
            n += 1
            global_step += 1

        avg_loss = total_loss / n
        avg_psnr = total_psnr / n
        elapsed = time.time() - t_start
        avg_grads = {k: float(np.mean(v)) for k, v in epoch_grads.items()}

        val_res = validate(model, val_loader, device, lpips_model, val_channel_types)
        val_psnr_10 = val_res["awgn"][10]["psnr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | {elapsed:5.1f}s | "
            f"Loss={avg_loss:.5f} PSNR={avg_psnr:.1f}dB | "
            f"Val@10dB={val_psnr_10:.1f}dB SSIM={val_res['awgn'][10]['ssim']:.3f} "
            f"LPIPS={val_res['awgn'][10]['lpips']:.4f} | LR={lr:.1e}"
        )

        if epoch <= 5 or epoch % 10 == 0:
            grad_str = " | ".join(
                f"{k}={v:.6f}" for k, v in sorted(avg_grads.items())
            )
            print(f"  Grads: {grad_str}")

        # ── W&B logging ──────────────────────────────────────────────
        if use_wandb:
            wb_log: dict = {
                "train/loss": avg_loss,
                "train/psnr": avg_psnr,
                "train/lr": lr,
            }
            for ch_type in val_channel_types:
                for snr_val in [0, 5, 10, 15, 20]:
                    prefix = f"val/{ch_type}/snr_{snr_val}"
                    wb_log[f"{prefix}/psnr"] = val_res[ch_type][snr_val]["psnr"]
                    wb_log[f"{prefix}/ssim"] = val_res[ch_type][snr_val]["ssim"]
                    wb_log[f"{prefix}/lpips"] = val_res[ch_type][snr_val]["lpips"]

            if epoch <= 5 or epoch % 10 == 0:
                for k, v in avg_grads.items():
                    wb_log[f"grad/{k}"] = v

            wandb.log(wb_log, step=epoch)

        # ── Save best ────────────────────────────────────────────────
        if val_psnr_10 > best_psnr:
            best_psnr = val_psnr_10
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_psnr": best_psnr,
                "config": vars(args),
            }
            torch.save(ckpt_data, exp_dir / "best.pt")
            print(f"  >>> New best: {best_psnr:.2f} dB")
            best_epoch = epoch
            # The W&B artifact is logged once after the loop, not here. best.pt carries the
            # optimizer state as well as the weights (~66 MB at 5.5 M params under Adam), and
            # uploading it on every improvement queued 1.5 GB at process exit on a 50-epoch run
            # -- which a chained evaluation then waits behind with the GPU idle.

        # ── Images ───────────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            log_images(
                model, val_loader, epoch, exp_dir, device,
                [0, 5, 10, 15, 20], use_wandb,
            )

        # ── Checkpoint ───────────────────────────────────────────────
        # Always save last.pt for post-hoc cls_sim selection (audit 2026-06-03:
        # best.pt is selected by val PSNR which is NOT our canonical SOTA metric;
        # cls_sim winners can lose PSNR. Keep last + periodic epochs so we can
        # re-rank by cls_sim retrospectively with the cls-sim evaluators.)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "config": vars(args),
        }, exp_dir / "last.pt")
        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_psnr": best_psnr,
                "config": vars(args),
            }, exp_dir / f"epoch_{epoch}.pt")

        # ── CSV ──────────────────────────────────────────────────────
        row = [epoch, f"{avg_loss:.6f}", f"{avg_psnr:.2f}", f"{lr:.2e}"]
        for ch_type in val_channel_types:
            for snr_val in [0, 5, 10, 15, 20]:
                row.extend([
                    f"{val_res[ch_type][snr_val]['psnr']:.2f}",
                    f"{val_res[ch_type][snr_val]['ssim']:.4f}",
                    f"{val_res[ch_type][snr_val]['lpips']:.4f}",
                ])
        csv_w.writerow(row)
        csv_file.flush()

    csv_file.close()
    print(f"\nDone! Best val PSNR: {best_psnr:.2f} dB")
    print(f"Experiment: {exp_dir}")

    if use_wandb:
        # One upload of the final best checkpoint. See the comment at the new-best branch.
        if (exp_dir / "best.pt").exists():
            artifact = wandb.Artifact(
                f"best-model-{args.model}", type="model",
                metadata={"psnr": best_psnr, "epoch": best_epoch, **vars(args)},
            )
            artifact.add_file(str(exp_dir / "best.pt"))
            wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
