"""Train SwinJSCC w/ SA on the DJI 512 dataset: the trainer of the SwinJSCC-capacity chain.

Pipeline:
  - HR_image-style 256x256 random crops from data/dji/frames_512_v1/train (their canonical setup)
  - SwinJSCC w/_SA, C=96 fixed, multiple SNR train ∈ {1,4,7,10,13}, AWGN
  - Adam lr=1e-4, batch=8 (6GB GPU constraint vs their 16)
  - Validate every epoch on data/dji/frames_512_v1/val using full 512 frames
  - Save best-val-PSNR checkpoint

Outputs:
  - experiments/swinjscc_dji_512_<timestamp>/best.pt
  - experiments/swinjscc_dji_512_<timestamp>/log.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMVIDEO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = SEMVIDEO_ROOT / "experiments"
# Upstream SwinJSCC code is not distributed with this repository (see README, "Third-party
# code"). Set SWINJSCC_DIR, or place it at <parent of this repository>/third_party/SwinJSCC
# (/third_party/SwinJSCC when the repository is mounted at /workspace in the Docker image).
SWINJSCC_DIR = Path(os.environ.get("SWINJSCC_DIR", REPO_ROOT / "third_party" / "SwinJSCC"))
sys.path.insert(0, str(SEMVIDEO_ROOT / "src"))
sys.path.insert(0, str(SWINJSCC_DIR))
os.chdir(str(SWINJSCC_DIR))

from loss.distortion import MS_SSIM  # noqa: E402
from net.network import SwinJSCC  # noqa: E402


def repo_path(path_value: str | None) -> Path | None:
    """Resolve relative CLI paths against the repository root, not SwinJSCC cwd."""
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


class DJIDataset(Dataset):
    """Dataset of single frames: random 256x256 crops for training, 512x512 for validation."""

    def __init__(
        self,
        root: Path | list[Path],
        crop: int | None = 256,
        train: bool = True,
        max_samples: int | None = None,
    ):
        # Accept either a single root or a list of roots (for mixed-corpus
        # training). When multiple are given we pool frames from every dir,
        # which addresses the cross-corpus weakness of single-corpus training.
        if isinstance(root, (str, Path)):
            roots = [Path(root)]
        else:
            roots = [Path(r) for r in root]
        all_paths = []
        for r in roots:
            all_paths.extend(glob(str(r / "*.jpg")))
            all_paths.extend(glob(str(r / "*.png")))
            # Also walk one level down (covers e.g. VisDrone/images/*.jpg).
            all_paths.extend(glob(str(r / "**" / "*.jpg")))
            all_paths.extend(glob(str(r / "**" / "*.png")))
        # Dedupe + canonical sort for reproducibility.
        self.paths = sorted(set(all_paths))
        if not self.paths:
            raise FileNotFoundError(f"No frames in {roots}")
        if max_samples is not None and max_samples > 0:
            self.paths = self.paths[:max_samples]
        self.crop = crop

        if train:
            assert crop is not None
            self.tx = transforms.Compose([
                transforms.RandomCrop((crop, crop)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            # squarify non-512 val (VisDrone) so SwinJSCC's window partition holds;
            # this is the transform every SwinJSCC checkpoint was validated with
            self.tx = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop((512, 512)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.paths)

    def _load_pil(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, i):
        return self.tx(self._load_pil(self.paths[i]))


def build_config(model_name, image_dims, channel_count):
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
    channel_number = (
        channel_count if model_name in ("SwinJSCC_w/o_SAandRA", "SwinJSCC_w/_SA") else None
    )

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


def load_compatible_state(
    net: nn.Module,
    checkpoint: Path,
    device: torch.device,
) -> tuple[int, int, int]:
    """Load checkpoint tensors whose names and shapes match the current model."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    current = net.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    skipped = len(state) - len(compatible)
    missing, unexpected = net.load_state_dict(compatible, strict=False)
    return len(missing), len(unexpected), skipped


@torch.no_grad()
def validate(net, val_loader, snr_list, device):
    net.eval()
    out = {}
    for snr in snr_list:
        psnrs = []
        for x in val_loader:
            x = x.to(device)
            recon, cbr, _, mse, _ = net(x, given_SNR=snr, given_rate=None)
            recon = recon.clamp(0.0, 1.0)
            mse_val = mse.item()
            if mse_val > 0:
                psnrs.append(10 * math.log10(255.0 * 255.0 / mse_val))
        out[snr] = float(np.mean(psnrs)) if psnrs else 0.0
    net.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "dji" / "frames_512_v1" / "train"),
    )
    ap.add_argument(
        "--val-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "dji" / "frames_512_v1" / "val"),
    )
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--C", type=int, default=96)
    ap.add_argument("--loss", choices=["mse", "mse_yolo"], default="mse",
                    help="Training loss. mse_yolo adds MS-SSIM + frozen YOLO feature L1.")
    ap.add_argument("--yolo-weights", type=str, default=str(SEMVIDEO_ROOT / "yolov8m.pt"),
                    help="YOLO weights for detector-feature distillation.")
    ap.add_argument("--yolo-feat-weight", type=float, default=0.5,
                    help="Weight for YOLO feature L1 when --loss mse_yolo.")
    ap.add_argument("--ms-ssim-weight", type=float, default=0.15,
                    help="Weight for 1-MS-SSIM when --loss mse_yolo.")
    ap.add_argument("--yolo-feat-warmup", type=int, default=3,
                    help="Pixel-only warmup epochs before adding feature loss.")
    ap.add_argument("--dino-feat-weight", type=float, default=0.0,
                    help="Weight for DINOv2 patch-token L1. If >0, runs in "
                         "parallel with YOLO feat loss → cross-supervisor "
                         "ensemble. Activates with the same warmup schedule.")
    ap.add_argument("--dino-layers", type=str, default="",
                    help="Comma-separated 0-indexed DINOv2 block indices for "
                         "multi-layer distillation (e.g. '5,8,11'). Empty = "
                         "use only the final block (legacy single-layer).")
    ap.add_argument("--dino-layer-weights", type=str, default="",
                    help="Comma-separated weights matching --dino-layers length. "
                         "Empty = default (0.25,0.25,0.50) for 3 layers, else uniform.")
    ap.add_argument("--dino-model", type=str, default="dinov2_vits14",
                    choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
                             "dinov3_vits16", "dinov3_vitb16"],
                    help="Frozen feature-supervisor backbone. dinov2 via torch.hub "
                         "(vits14/vitb14/vitl14 = small/base/large; all open). "
                         "dinov3 via HuggingFace (gated; needs auth).")
    ap.add_argument("--scenegraph-feat-weight", type=float, default=0.0,
                    help="Weight for the vision-language patch-token L1 (third "
                         "supervisor; the SigLIP term of the paper with "
                         "--scenegraph-model siglip). If >0, runs in parallel "
                         "with the other feature losses and shares the same "
                         "warmup schedule. The flag name is historical.")
    ap.add_argument("--scenegraph-model", type=str, default="owlv2",
                    help="Backbone of that supervisor: 'siglip' (SigLIP "
                         "ViT-B/16, used in the paper) or 'owlv2' (historical "
                         "default). May also be a fully-qualified HF model id.")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="If >0, maintain an exponential moving average of the "
                         "model weights (Morales-Brotons 2024). Recommended "
                         "0.999-0.9999. EMA weights are evaluated for "
                         "best-checkpoint selection.")
    ap.add_argument("--snr-train", type=str, default="1,4,7,10,13")
    ap.add_argument("--snr-val", type=str, default="0,5,10,15,20")
    ap.add_argument("--train-channel", type=str, default="awgn",
                    choices=["awgn", "rayleigh"],
                    help="Wireless channel for training. Rayleigh tests "
                         "fading-channel robustness.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="Optional cap for smoke tests; use full train set by default.")
    ap.add_argument("--max-val-samples", type=int, default=None,
                    help="Optional cap for smoke tests; use full val set by default.")
    ap.add_argument("--init-from", type=str, default=str(
        SWINJSCC_DIR / "checkpoint" / "SwinJSCC" / "SwinJSCC w_ SA"
        / "SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model"))
    ap.add_argument("--train-erasure-p-max", type=float, default=0.0,
                    help="If >0, during training apply i.i.d. Bernoulli erasure "
                         "on the post-channel latent with p sampled uniformly "
                         "from [0, this value] per minibatch. Augments the "
                         "training distribution with packet-loss conditions so "
                         "the codec learns to tolerate erasure. Eval is "
                         "unaffected. Recommended range: 0.10-0.25.")
    ap.add_argument("--train-reactive-jam-prob", type=float, default=0.0,
                    help="If >0, during training a fraction of minibatches "
                         "get post-channel reactive jamming (arXiv:2502.02385). "
                         "Augments training against the reactive-jam "
                         "hard-negative. Eval is unaffected.")
    ap.add_argument("--train-reactive-jam-jnr-min", type=float, default=5.0,
                    help="Min JNR (dB) per jammed minibatch.")
    ap.add_argument("--train-reactive-jam-jnr-max", type=float, default=15.0,
                    help="Max JNR (dB) per jammed minibatch.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for torch, numpy, and dataloader workers. "
                         "Use the same seed to reproduce metrics within 0.01.")
    ap.add_argument("--amp-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="Mixed-precision dtype for the train/eval forward. "
                         "bf16 = 1.8-2.5x faster on RTX 4090/Ampere+ with no "
                         "GradScaler needed. fp32 = legacy / strict reproduce.")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(net, mode='reduce-overhead'). Adds ~30s "
                         "first-step JIT cost, then 1.3-1.8x faster steady-state.")
    args_ns = ap.parse_args()

    # ---- Reproducibility: seed every RNG before model build ----
    import random
    random.seed(args_ns.seed)
    np.random.seed(args_ns.seed)
    torch.manual_seed(args_ns.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_ns.seed)
    print(f"[seed] torch/numpy/random seeded to {args_ns.seed}")

    if args_ns.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args_ns.out = str(EXPERIMENTS_DIR / f"swinjscc_dji_512_{ts}")
    out_dir = repo_path(args_ns.out)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] out: {out_dir}")

    snr_train = [int(s) for s in args_ns.snr_train.split(",")]
    snr_val = [int(s) for s in args_ns.snr_val.split(",")]

    sj_args = SimpleNamespace(
        model="SwinJSCC_w/_SA",
        channel_type=args_ns.train_channel,
        C=str(args_ns.C),
        multiple_snr=",".join(str(s) for s in snr_train),
        distortion_metric="MSE",
        trainset="DIV2K",
        logger=None,
    )

    # Build model at 256 (training resolution) so attn_mask matches their pretrained
    cfg = build_config(sj_args.model, image_dims=(3, 256, 256), channel_count=args_ns.C)
    device = torch.device("cuda:0")
    net = SwinJSCC(sj_args, cfg).to(device)

    # Init from their pretrained (warm start gives us a much faster convergence).
    init_path = repo_path(args_ns.init_from)
    if init_path is not None and init_path.is_file():
        missing, unexpected, skipped = load_compatible_state(net, init_path, device)
        print(f"[init] warm-start from {init_path.name}: "
              f"missing={missing}, unexpected={unexpected}, skipped_shape={skipped}")
    else:
        print("[init] no init checkpoint -> train from scratch")

    if args_ns.train_erasure_p_max > 0:
        # Proper nn.Module replacement for the prior monkey-patch on
        # feature_pass_channel. Survives state_dict round-trip and TensorRT
        # export (Jetson deployment).
        from semvideo.channel.erasure_wrapped import ChannelWithErasure
        # If we already wrapped (e.g. resume from a wrapped checkpoint), don't double-wrap.
        if not isinstance(net.channel, ChannelWithErasure):
            net.channel = ChannelWithErasure(
                net.channel, p_max=args_ns.train_erasure_p_max, always_apply=False
            ).to(device)
        else:
            net.channel.p_max = args_ns.train_erasure_p_max
        print(f"[erasure-aug] training-time Bernoulli erasure on, "
              f"p ~ U[0, {args_ns.train_erasure_p_max}] per minibatch "
              f"(ChannelWithErasure module, eval-passthrough)")

    if args_ns.train_reactive_jam_prob > 0:
        # Train-time DRFM-style reactive-jam augmentation. Mirrors the
        # erasure-aug pattern: P80/P81 family.
        from semvideo.channel.reactive_jam_wrapped import ChannelWithReactiveJam
        if not isinstance(net.channel, ChannelWithReactiveJam):
            net.channel = ChannelWithReactiveJam(
                net.channel,
                jam_prob=args_ns.train_reactive_jam_prob,
                jnr_min_db=args_ns.train_reactive_jam_jnr_min,
                jnr_max_db=args_ns.train_reactive_jam_jnr_max,
                always_apply=False,
            ).to(device)
        else:
            net.channel.jam_prob = args_ns.train_reactive_jam_prob
        print(f"[reactive-jam-aug] training-time reactive jamming on, "
              f"prob={args_ns.train_reactive_jam_prob}, "
              f"JNR ~ U[{args_ns.train_reactive_jam_jnr_min}, "
              f"{args_ns.train_reactive_jam_jnr_max}] dB "
              f"(ChannelWithReactiveJam, eval-passthrough)")

    # AMP + compile setup. bf16 is preferred on RTX 4090/Ampere+ – no GradScaler
    # needed (no overflow risk), 1.8-2.5x throughput. Compile adds first-step JIT
    # cost ~30s then steady-state speedup.
    amp_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args_ns.amp_dtype]
    use_amp = (amp_dtype is not torch.float32)
    if args_ns.compile:
        # torch.compile failure happens at first-step JIT, not at wrap time.
        # Pre-set dynamo suppress_errors so the failure (e.g. missing g++ in
        # a slim docker image) silently falls back to eager rather than crashing.
        try:
            # NB: ``import torch._dynamo`` here would rebind ``torch`` as a
            # local name throughout main() and break the earlier
            # ``torch.manual_seed(...)``. Use ``importlib`` to side-step.
            import importlib
            _dynamo = importlib.import_module("torch._dynamo")
            _dynamo.config.suppress_errors = True
            net = torch.compile(net, mode="reduce-overhead", dynamic=False)
            print(f"[compile] torch.compile enabled (reduce-overhead); "
                  f"first-step JIT failures will silently fall back to eager")
        except Exception as e:
            print(f"[compile] WARNING: torch.compile wrap failed, eager mode: {e}")
    print(f"[amp] dtype={args_ns.amp_dtype}, autocast={use_amp}")


    # Comma-separated --train-dir lets us pool multiple corpora (mixed-corpus
    # training). Each comma-piece is a directory; we glob both top-level and
    # nested *.jpg / *.png so VisDrone-style {root}/images/*.jpg works directly.
    train_roots = [repo_path(p.strip()) for p in str(args_ns.train_dir).split(",") if p.strip()]
    train_roots = [r for r in train_roots if r is not None]
    train_ds = DJIDataset(
        train_roots if len(train_roots) > 1 else train_roots[0],
        crop=256,
        train=True,
        max_samples=args_ns.max_train_samples,
    )
    val_ds_full = DJIDataset(
        repo_path(args_ns.val_dir),
        crop=None,
        train=False,
        max_samples=args_ns.max_val_samples,
    )
    # Use a sub-sample for fast in-loop validation; full sweep at end
    val_subset_idx = list(range(0, len(val_ds_full), max(1, len(val_ds_full) // 80)))[:80]
    val_ds = torch.utils.data.Subset(val_ds_full, val_subset_idx)
    print(f"[data] train: {len(train_ds)}, val (subset/full): {len(val_ds)}/{len(val_ds_full)}")

    def _worker_init(worker_id: int) -> None:
        """Seed each dataloader worker deterministically (reproducibility)."""
        seed = args_ns.seed + worker_id
        np.random.seed(seed)
        import random as _random
        _random.seed(seed)
        torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(args_ns.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args_ns.batch, shuffle=True,
        num_workers=args_ns.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args_ns.num_workers > 0,
        worker_init_fn=_worker_init, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True,
        worker_init_fn=_worker_init,
    )

    optimizer = optim.Adam(net.parameters(), lr=args_ns.lr)
    mse_fn = nn.MSELoss()
    yolo_feat_loss = None
    dino_feat_loss = None
    scenegraph_feat_loss = None
    ms_ssim_fn = None
    warmup_steps = args_ns.yolo_feat_warmup * len(train_loader)

    # EMA of model weights (Morales-Brotons 2024). Maintains a shadow copy
    # of the weights with `w_ema = decay * w_ema + (1 - decay) * w`; evaluated
    # for best-checkpoint selection.
    #
    # Note: under ``torch.compile`` the
    # wrapper prefixes state-dict keys with ``_orig_mod.``. We always read
    # from the *unwrapped* module so EMA keys / checkpoint keys stay stable
    # regardless of ``--compile``.
    def _unwrapped(m: nn.Module) -> nn.Module:
        return m._orig_mod if hasattr(m, "_orig_mod") else m

    ema_state: dict[str, torch.Tensor] | None = None
    if args_ns.ema_decay > 0:
        ema_state = {k: v.detach().clone()
                     for k, v in _unwrapped(net).state_dict().items()
                     if v.dtype.is_floating_point}
        print(f"[ema] decay={args_ns.ema_decay} shadow weights enabled", flush=True)
    if args_ns.loss == "mse_yolo":
        from semvideo.training.detector_feature_loss import DetectorFeatureLoss

        yolo_weights = repo_path(args_ns.yolo_weights)
        assert yolo_weights is not None
        yolo_feat_loss = DetectorFeatureLoss(
            weights_path=str(yolo_weights),
            device=str(device),
        )
        ms_ssim_fn = MS_SSIM(data_range=1.0, levels=4, channel=3).to(device)
        if args_ns.dino_feat_weight > 0:
            from semvideo.training.dinov2_feature_loss import DINOv2FeatureLoss
            dino_layers: tuple[int, ...] | None = None
            dino_lw: tuple[float, ...] | None = None
            if args_ns.dino_layers.strip():
                dino_layers = tuple(int(s) for s in args_ns.dino_layers.split(",") if s.strip())
                if args_ns.dino_layer_weights.strip():
                    dino_lw = tuple(float(s) for s in args_ns.dino_layer_weights.split(",") if s.strip())
            dino_feat_loss = DINOv2FeatureLoss(device=str(device), layers=dino_layers, layer_weights=dino_lw, model_name=args_ns.dino_model)
            print(
                "[loss] mse_yolo + dino ENSEMBLE: MSE + "
                f"{args_ns.ms_ssim_weight}*(1-MS-SSIM) + "
                f"{args_ns.yolo_feat_weight}*YOLO-feat-L1 + "
                f"{args_ns.dino_feat_weight}*DINO-patch-L1; "
                f"warmup={args_ns.yolo_feat_warmup} epochs",
                flush=True,
            )
        else:
            print(
                "[loss] mse_yolo: MSE + "
                f"{args_ns.ms_ssim_weight}*(1-MS-SSIM) + "
                f"{args_ns.yolo_feat_weight}*YOLO-feat-L1; "
                f"warmup={args_ns.yolo_feat_warmup} epochs",
                flush=True,
            )
        if args_ns.scenegraph_feat_weight > 0:
            from semvideo.training.scenegraph_feature_loss import (
                SceneGraphFeatureLoss,
            )
            scenegraph_feat_loss = SceneGraphFeatureLoss(
                device=str(device),
                model_name=args_ns.scenegraph_model,
            )
            print(
                f"[loss] + vision-language supervisor ({args_ns.scenegraph_model}): "
                f"{args_ns.scenegraph_feat_weight}*VL-patch-L1; "
                f"shares warmup={args_ns.yolo_feat_warmup} epochs",
                flush=True,
            )
    else:
        print("[loss] mse", flush=True)

    log_csv = out_dir / "log.csv"
    with open(log_csv, "w", newline="") as fp:
        csv.writer(fp).writerow(["epoch", "train_loss", "train_mse",
                                 "train_ms_ssim", "train_yolo_feat",
                                 "train_psnr", "wall_s",
                                 *(f"val_psnr_snr{s}" for s in snr_val)])

    best_val = -1.0
    global_step = 0
    t_start = time.time()
    for epoch in range(args_ns.epochs):
        net.train()
        epoch_loss, epoch_mse, epoch_msssim, epoch_yolo, epoch_psnr, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        ep_start = time.time()
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            # bf16 autocast – no GradScaler needed (no overflow risk). fp32 path
            # is the legacy default (--amp-dtype fp32).
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                recon, cbr, snr_used, mse, _ = net(x)
                pred = recon.clamp(0.0, 1.0)
                mse_loss = mse_fn(pred, x)
                loss = mse_loss
                ms_ssim_loss = pred.new_zeros(())
                feat_loss = pred.new_zeros(())
                dino_loss = pred.new_zeros(())
                scenegraph_loss = pred.new_zeros(())
                if args_ns.loss == "mse_yolo" and global_step >= warmup_steps:
                    assert yolo_feat_loss is not None
                    assert ms_ssim_fn is not None
                    ms_ssim_loss = ms_ssim_fn(pred, x).mean()
                    feat_loss = yolo_feat_loss(pred, x)
                    loss = (
                        mse_loss
                        + args_ns.ms_ssim_weight * ms_ssim_loss
                        + args_ns.yolo_feat_weight * feat_loss
                    )
                    if dino_feat_loss is not None:
                        dino_loss = dino_feat_loss(pred, x)
                        loss = loss + args_ns.dino_feat_weight * dino_loss
                    if scenegraph_feat_loss is not None:
                        scenegraph_loss = scenegraph_feat_loss(pred, x)
                        loss = loss + args_ns.scenegraph_feat_weight * scenegraph_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # EMA update on float tensors only. Read from unwrapped module so
            # keys match regardless of torch.compile (see fix above).
            if ema_state is not None:
                d = args_ns.ema_decay
                with torch.no_grad():
                    for k, v in _unwrapped(net).state_dict().items():
                        if k in ema_state:
                            ema_state[k].mul_(d).add_(v.detach(), alpha=1 - d)
            with torch.no_grad():
                if mse.item() > 0:
                    psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
                    epoch_psnr += psnr
                    n += 1
            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_msssim += float(ms_ssim_loss.detach().item())
            epoch_yolo += float(feat_loss.detach().item())
            global_step += 1
        ep_dur = time.time() - ep_start
        avg_loss = epoch_loss / max(1, len(train_loader))
        avg_mse_loss = epoch_mse / max(1, len(train_loader))
        avg_msssim_loss = epoch_msssim / max(1, len(train_loader))
        avg_yolo_loss = epoch_yolo / max(1, len(train_loader))
        avg_psnr = epoch_psnr / max(1, n)
        print(f"[epoch {epoch:3d}] loss={avg_loss:.5f} psnr={avg_psnr:.2f} "
              f"mse={avg_mse_loss:.5f} yolo={avg_yolo_loss:.5f} "
              f"({ep_dur:.0f}s, {ep_dur / max(1, len(train_loader)):.2f}s/it)", flush=True)

        # Validate
        if (epoch + 1) % args_ns.val_every == 0 or epoch == args_ns.epochs - 1:
            # NOTE: tried EMA-aware validation via load_state_dict swap; broken
            # because SwinJSCC's attn_mask buffers resize with input resolution
            # (training crop=256 vs val crop=512), making the round-trip fail.
            # Compromise: validate with live weights but SAVE the EMA shadow as
            # the canonical checkpoint when "best" is hit. EMA tracks live with
            # decay 0.9999 so the selection signal is correlated either way.
            val_metrics = validate(net, val_loader, snr_val, device)
            val_str = " ".join(f"snr{s}={v:.2f}" for s, v in val_metrics.items())
            val_at_10 = val_metrics.get(10, val_metrics[snr_val[len(snr_val) // 2]])
            print(f"[val   {epoch:3d}] {val_str}", flush=True)
            if val_at_10 > best_val:
                best_val = val_at_10
                ckpt_path = out_dir / "best.pt"
                # Save the EMA weights as the canonical checkpoint when EMA active.
                # Always read from the unwrapped module to strip ``_orig_mod.``
                # prefix added by torch.compile.
                live_state = _unwrapped(net).state_dict()
                save_state = ema_state if ema_state is not None else live_state
                if ema_state is not None:
                    save_state = dict(save_state)
                    for k, v in live_state.items():
                        if k not in save_state:
                            save_state[k] = v
                torch.save({
                    "epoch": epoch,
                    "model_state": save_state,
                    "val_psnr_snr10": best_val,
                    "args": vars(args_ns),
                    "ema_decay": args_ns.ema_decay,
                }, ckpt_path)
                print(f"[save ] best -> {ckpt_path} (val PSNR@10 {best_val:.3f})", flush=True)

            with open(log_csv, "a", newline="") as fp:
                csv.writer(fp).writerow([
                    epoch, f"{avg_loss:.6f}", f"{avg_mse_loss:.6f}",
                    f"{avg_msssim_loss:.6f}", f"{avg_yolo_loss:.6f}",
                    f"{avg_psnr:.4f}", f"{time.time() - t_start:.0f}",
                    *(f"{val_metrics[s]:.4f}" for s in snr_val),
                ])

    # Final full sweep on ALL 375 val frames
    print("\n[final] running full-val sweep on all val frames...")
    full_val_loader = DataLoader(
        val_ds_full,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    final_metrics = validate(net, full_val_loader, snr_val, device)
    final_str = " ".join(f"snr{s}={v:.3f}" for s, v in final_metrics.items())
    print(f"[final] {final_str}")
    with open(out_dir / "final_val.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["snr", "psnr"])
        for s, v in final_metrics.items():
            w.writerow([s, f"{v:.4f}"])

    print(f"\n[done] best val PSNR@10 = {best_val:.3f}")
    print(f"[done] total wall time: {(time.time() - t_start) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
