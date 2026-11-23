"""Objective transfer to a MambaJSCC trunk: the matched pair of Appendix C.1 of the paper.

Question. Does the detector-feature objective lift a state-space (MambaJSCC, VSSM) trunk
that was not designed for it, or only the convolutional codec it was developed on?

Design. Both arms start from the SAME DJI-trained, PSNR-only MambaJSCC checkpoint, run the
SAME number of epochs with the SAME seed and the SAME data order. Only the loss differs:

    PSNR arm    : MambaJSCC's own published MSE objective (the control)
    DETFEAT arm : the paper's objective, Eq. (2) -- 1.0*MSE + 0.15*(1-MS-SSIM)
                  + 0.5*frozen-YOLOv8m-FPN-L1 over stages P3/P4/P5 at (1.0, 0.5, 0.25).
                  0.5 is the paper's lambda and the weight of the compact matched pairs.

Continuing one checkpoint two ways is a stronger control than two runs from scratch: the
arms share their entire optimization history up to the split, so the difference at the end
is attributable to the loss and not to initialization variance. The paper reports three
seeds (``--seed 1024``, the MambaJSCC default, then ``7`` and ``1337``).

Scale convention. MambaJSCC's MSE loss is a per-sample SUM (``mse(..., reduction='sum')/B``)
and its LPIPS option scales a mean-based loss by ``numel/B`` to match. We follow that
convention exactly, so all three of our terms keep their relative weights while sitting on
the scale the upstream learning rate was tuned for.

Gate. Before either arm trains, the loaded checkpoint is evaluated on DJI val at AWGN
SNR = 20 dB and must reproduce the initialization the paper reports (cls-sim 0.647 at
PSNR 31.61 dB, Appendix C.1), within the tolerances below. Otherwise the script refuses to
train: a lift measured from the wrong initialization is worthless. ``--epochs 0`` runs the
gate only, which is how the resolution control of Appendix C.1 evaluates the 256 px and
512 px trunks (pass ``--skip-init-gate`` for a trunk other than the 256 px one, and
``--train-config configs/mambajscc/vssm_tiny_DJI512.yaml`` to evaluate at 512 px).

Configuration. The training configurations of the two DJI trunks are in this repository,
``configs/mambajscc/vssm_tiny_DJI.yaml`` (256 px, the default) and ``vssm_tiny_DJI512.yaml``;
the model configuration is upstream's ``configs/vssm/vssm_tiny_DIV2K.yaml`` (the vssm_tiny
architecture). The input size comes from the training configuration, not from the checkpoint.

Requirements (not included in this repository):
  * the upstream MambaJSCC code, https://github.com/Wireless3C-SJTU/MambaJSCC, at
    ``$MAMBAJSCC_DIR`` (default ``<parent of this repository>/third_party/MambaJSCC``), in an
    environment where its selective-scan CUDA kernels are built, plus ``ultralytics`` and
    ``transformers``; a CUDA GPU;
  * the PSNR-only trunk, trained with MambaJSCC's own ``main.py`` on the configuration above
    (see the README, "MambaJSCC matched pair").

Usage::

    python scripts/mamba_matched_pair.py --loss DETFEAT --epochs 30 --out-tag detfeat_e30
    python scripts/mamba_matched_pair.py --loss PSNR    --epochs 30 --out-tag psnrctl_e30
    python scripts/mamba_matched_pair.py --loss DETFEAT --epochs 30 --seed 7 \\
        --out-tag detfeat_e30_s7
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMVIDEO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = SEMVIDEO_ROOT / "experiments"
# Upstream MambaJSCC code is not distributed with this repository (see README, "Third-party
# code"). Set MAMBAJSCC_DIR, or place it at <parent of this repository>/third_party/MambaJSCC
# (/third_party/MambaJSCC when the repository is mounted at /workspace in the Docker layout).
MAMBAJSCC_DIR = Path(os.environ.get("MAMBAJSCC_DIR", REPO_ROOT / "third_party" / "MambaJSCC"))
DETECTOR_LOSS_SRC = SEMVIDEO_ROOT / "src" / "semvideo" / "training" / "detector_feature_loss.py"

DINOV2 = "facebook/dinov2-small"
# Initialization the paper reports for the DJI PSNR-trained 256 px trunk (Appendix C.1):
# PSNR 31.61 dB, cls-sim 0.647 at AWGN SNR = 20 dB, n = 375. (value, tolerance)
GATE_PSNR = (31.61, 1.0)
GATE_CLSSIM = (0.647, 0.030)


def load_detector_feature_loss(device: str, yolo_weights: str) -> torch.nn.Module:
    """Import the paper's detector-feature loss from this repository by file path.

    By path, not by package import: ``semvideo.training`` pulls in the whole training
    package, and the point here is to run the *same* loss code as the SwinJSCC
    experiments with no other semvideo machinery in the process.
    """
    spec = importlib.util.spec_from_file_location("detector_feature_loss", DETECTOR_LOSS_SRC)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {DETECTOR_LOSS_SRC}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DetectorFeatureLoss(weights_path=yolo_weights, device=device)


class MatchedCriterion(torch.nn.Module):
    """Either arm's objective, on MambaJSCC's per-sample-sum scale."""

    def __init__(
        self,
        mode: str,
        device: str,
        yolo_weights: str,
        ms_ssim_cls: type | None = None,
        gamma: float = 0.15,
        feat_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.gamma = gamma
        self.feat_weight = feat_weight
        if mode == "DETFEAT":
            assert ms_ssim_cls is not None
            self.msssim = ms_ssim_cls(data_range=1.0, levels=4, channel=3).to(device)
            self.detfeat = load_detector_feature_loss(device, yolo_weights)

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> dict:
        """Return the total loss and its detached parts; ``recon``/``target`` are (B, 3, H, W)."""
        b = target.shape[0]
        per_sample = target.numel() / b
        mse_sum = F.mse_loss(recon, target, reduction="sum") / b
        if self.mode == "PSNR":
            return {"total": mse_sum, "mse": mse_sum.detach()}
        msssim = self.msssim(recon, target).mean() * per_sample
        feat = self.detfeat(recon, target) * per_sample
        total = mse_sum + self.gamma * msssim + self.feat_weight * feat
        return {
            "total": total,
            "mse": mse_sum.detach(),
            "ms_ssim": msssim.detach(),
            "yolo_feat": feat.detach(),
        }


class ClsSim:
    """The paper's goal metric: DINOv2 class-token cosine similarity."""

    def __init__(self, device: str) -> None:
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(DINOV2).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def cls(self, x: torch.Tensor) -> torch.Tensor:
        """Class token of a (B, 3, H, W) batch in [0, 1]."""
        x = F.interpolate(x.clamp(0, 1), (518, 518), mode="bilinear", align_corners=False)
        return self.model(pixel_values=(x - self.mean) / self.std).last_hidden_state[:, 0]

    @torch.no_grad()
    def sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity of the class tokens of two single-image batches."""
        return float(F.cosine_similarity(self.cls(a), self.cls(b)).item())


def channel_roundtrip(
    encoder, decoder, channel, config, x: torch.Tensor, snr: float
) -> torch.Tensor:
    """Encode -> channel -> decode, mirroring MambaJSCC's run/train.py exactly."""
    feature = encoder(x, snr)
    received, pwr, h = channel.forward(feature, snr)
    if config.CHANNEL.TYPE == "rayleigh":
        sigma_square = 1.0 / (10 ** (snr / 10))
        received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
    elif config.CHANNEL.TYPE != "awgn":
        raise ValueError("channel type error")
    received = torch.cat((torch.real(received), torch.imag(received)), dim=2) * torch.sqrt(pwr)
    return decoder(received, snr)


@torch.no_grad()
def evaluate(
    encoder,
    decoder,
    channel,
    config,
    metric: ClsSim,
    val_glob: str,
    snrs: list[float],
    img_size: int,
    n: int,
    device: str,
) -> dict:
    """cls-sim and PSNR on ``n`` evenly spaced validation frames at each SNR."""
    import cv2

    paths = sorted(glob.glob(val_glob))
    if not paths:
        raise SystemExit(f"no val frames matched {val_glob}")
    idx = np.linspace(0, len(paths) - 1, min(n, len(paths)), dtype=int)
    paths = [paths[int(i)] for i in idx]
    encoder.eval()
    decoder.eval()
    out = {}
    for snr in snrs:
        sims, psnrs = [], []
        for k, path in enumerate(paths):
            bgr = cv2.imread(path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            x = F.interpolate(x, (img_size, img_size), mode="bilinear", align_corners=False)
            torch.manual_seed(2000 + k)  # same channel realization across arms
            rec = channel_roundtrip(encoder, decoder, channel, config, x, snr).clamp(0, 1)
            sims.append(metric.sim(x, rec))
            mse = F.mse_loss(rec, x).item()
            psnrs.append(10.0 * np.log10(1.0 / max(mse, 1e-12)))
        out[str(snr)] = {
            "cls_sim": float(np.mean(sims)),
            "cls_sim_std": float(np.std(sims)),
            "psnr": float(np.mean(psnrs)),
            "n": len(paths),
            "per_frame_cls_sim": [float(s) for s in sims],
        }
        print(
            f"  [eval] SNR={snr:>4}  cls-sim {out[str(snr)]['cls_sim']:.4f}  "
            f"PSNR {out[str(snr)]['psnr']:.2f} dB  (n={len(paths)})",
            flush=True,
        )
    encoder.train()
    decoder.train()
    return out


def load_module_state(model: torch.nn.Module, path: str, what: str) -> None:
    """MambaJSCC saves whole modules; take the state dict out of the pickle."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.state_dict() if hasattr(obj, "state_dict") else obj
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"{what} state mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected -- refusing to continue"
        )
    print(f"[init] {what} <- {Path(path).name}", flush=True)


def _abs(path_value: str) -> str:
    """Resolve a CLI path against the caller's working directory (the script later chdirs)."""
    return path_value if not path_value else str(Path(path_value).expanduser().resolve())


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Matched MambaJSCC continuation: the paper's objective against MambaJSCC's "
        "own MSE objective from one PSNR-trained checkpoint (Appendix C.1).",
    )
    ap.add_argument("--loss", choices=["PSNR", "DETFEAT"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="Seed passed to MambaJSCC's seed_torch(); 1024 is its default and "
        "the first seed of the paper (then 7 and 1337).",
    )
    ap.add_argument(
        "--model-config",
        default=str(MAMBAJSCC_DIR / "configs" / "vssm" / "vssm_tiny_DIV2K.yaml"),
        help="MambaJSCC model configuration (default: upstream vssm_tiny_DIV2K.yaml; the "
        "original runs read a DJI copy of it with the same architecture, which the "
        "strict state load of the trunk checkpoint confirms).",
    )
    ap.add_argument(
        "--train-config",
        default=str(SEMVIDEO_ROOT / "configs" / "mambajscc" / "vssm_tiny_DJI.yaml"),
        help="MambaJSCC training configuration; sets the input size, the training data and "
        "the directories where MambaJSCC saved the PSNR-trained trunk "
        "(configs/mambajscc/vssm_tiny_DJI512.yaml for 512 px).",
    )
    ap.add_argument(
        "--train-dir",
        default="",
        help="Override DATA.train_data_dir of the training configuration "
        "(an ImageFolder root: one subdirectory holding the DJI train frames).",
    )
    ap.add_argument(
        "--init-encoder",
        default="",
        help="Encoder of the PSNR-trained trunk (default: last *.pt in the "
        "configuration's TRAIN.ENCODER_PATH).",
    )
    ap.add_argument(
        "--init-decoder",
        default="",
        help="Decoder of the PSNR-trained trunk (default: last *.pt in the "
        "configuration's TRAIN.DECODER_PATH).",
    )
    ap.add_argument("--yolo-weights", default=str(SEMVIDEO_ROOT / "yolov8m.pt"))
    ap.add_argument(
        "--feat-weight",
        type=float,
        default=0.5,
        help="lambda on the detector-feature term; 0.5 is the published value",
    )
    ap.add_argument(
        "--val-glob", default=str(REPO_ROOT / "data" / "dji" / "frames_512_v1" / "val" / "*")
    )
    ap.add_argument("--eval-snrs", default="20,10,5")
    ap.add_argument("--eval-n", type=int, default=375)
    ap.add_argument(
        "--skip-init-gate",
        action="store_true",
        help="do not stop when the initialization differs from the paper's 256 px "
        "trunk (needed to evaluate any other trunk with --epochs 0)",
    )
    ap.add_argument("--out-root", default=str(EXPERIMENTS_DIR / "mamba_matched_pair"))
    args = ap.parse_args()

    for name in (
        "model_config",
        "train_config",
        "train_dir",
        "init_encoder",
        "init_decoder",
        "yolo_weights",
        "out_root",
    ):
        setattr(args, name, _abs(getattr(args, name)))
    args.val_glob = str(
        Path(args.val_glob).expanduser().parent.resolve() / Path(args.val_glob).name
    )

    # Pick the GPU before CUDA is first touched (MambaJSCC's own main.py does this with a GPU
    # picker at import time; MAMBA_GPU selects the device here).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("MAMBA_GPU", "0"))
    if not torch.cuda.is_available():
        raise SystemExit("MambaJSCC's selective-scan kernels need a CUDA GPU")
    if not MAMBAJSCC_DIR.is_dir():
        raise SystemExit(
            f"upstream MambaJSCC code not found at {MAMBAJSCC_DIR} "
            "(set MAMBAJSCC_DIR; see the README)"
        )

    # MambaJSCC is imported from its own directory, as its run scripts expect.
    sys.path.insert(0, str(MAMBAJSCC_DIR))
    os.chdir(str(MAMBAJSCC_DIR))
    from configs.config import get_config
    from data.datasets import get_loader
    from models.channel import Channel
    from models.network import Mamba_decoder, Mamba_encoder
    from utils.distortion import MS_SSIM
    from utils.utils import GradualWarmupScheduler, seed_torch

    class _Args:
        model_config_path = args.model_config
        train_config_path = args.train_config
        mode = "train"

    device = "cuda"
    config = get_config(_Args)
    if args.train_dir:
        config.defrost()
        config.DATA.train_data_dir = args.train_dir
        config.freeze()
    out_dir = Path(args.out_root) / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_init = (
        args.init_encoder or sorted(glob.glob(os.path.join(config.TRAIN.ENCODER_PATH, "*.pt")))[-1]
    )
    dec_init = (
        args.init_decoder or sorted(glob.glob(os.path.join(config.TRAIN.DECODER_PATH, "*.pt")))[-1]
    )

    seed_torch(args.seed)
    encoder = Mamba_encoder(config).cuda()
    decoder = Mamba_decoder(config).cuda()
    channel = Channel(config)
    load_module_state(encoder, enc_init, "encoder")
    load_module_state(decoder, dec_init, "decoder")

    metric = ClsSim(device)
    img_size = config.DATA.IMG_SIZE
    eval_snrs = [float(s) for s in args.eval_snrs.split(",")]

    # ---- gate: the initialization must be the checkpoint the paper describes
    print(f"\n[gate] evaluating the initialization ({img_size}px, n={args.eval_n})", flush=True)
    init_metrics = evaluate(
        encoder,
        decoder,
        channel,
        config,
        metric,
        args.val_glob,
        [20.0],
        img_size,
        args.eval_n,
        device,
    )["20.0"]
    for name, got, (want, tol) in (
        ("PSNR", init_metrics["psnr"], GATE_PSNR),
        ("cls-sim", init_metrics["cls_sim"], GATE_CLSSIM),
    ):
        ok = abs(got - want) <= tol
        print(
            f"[gate] {name}: got {got:.4f}, paper {want} +-{tol} -> {'OK' if ok else 'MISMATCH'}",
            flush=True,
        )
        if not ok and not args.skip_init_gate:
            raise SystemExit(
                "GATE FAILED: the loaded checkpoint is not the one the paper "
                "describes -- refusing to train an arm off it"
            )
    print("[gate] PASS", flush=True)

    # ---- training, mirroring MambaJSCC's run/train.py
    train_loader, _ = get_loader(config)
    criterion = MatchedCriterion(
        args.loss, device, args.yolo_weights, ms_ssim_cls=MS_SSIM, feat_weight=args.feat_weight
    )
    opt_e = optim.AdamW(encoder.parameters(), lr=config.TRAIN.BASE_LR, weight_decay=1e-4)
    opt_d = optim.AdamW(decoder.parameters(), lr=config.TRAIN.BASE_LR, weight_decay=1e-4)
    sch_e = GradualWarmupScheduler(
        optimizer=opt_e,
        multiplier=2.0,
        warm_epoch=0.1,
        after_scheduler=optim.lr_scheduler.CosineAnnealingLR(
            opt_e, T_max=args.epochs, eta_min=0, last_epoch=-1
        ),
    )
    sch_d = GradualWarmupScheduler(
        optimizer=opt_d,
        multiplier=2.0,
        warm_epoch=0.1,
        after_scheduler=optim.lr_scheduler.CosineAnnealingLR(
            opt_d, T_max=args.epochs, eta_min=0, last_epoch=-1
        ),
    )

    encoder.train()
    decoder.train()
    snr_list = config.CHANNEL.SNR
    seed_torch(args.seed)  # both arms now share data order and SNR draws
    history = []
    for e in range(args.epochs):
        run_loss, steps = 0.0, 0
        for input_image, _ in train_loader:
            snr = snr_list[torch.randint(0, len(snr_list), (1,)).item()]
            x = input_image.cuda()  # (B, 3, H, W)
            opt_e.zero_grad()
            opt_d.zero_grad()
            rec = channel_roundtrip(encoder, decoder, channel, config, x, snr)
            parts = criterion(rec, x)
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)
            opt_e.step()
            opt_d.step()
            run_loss += float(parts["total"].item())
            steps += 1
        sch_e.step()
        sch_d.step()
        mean_loss = run_loss / max(steps, 1)
        history.append({"epoch": e, "loss": mean_loss})
        print(
            f"[{args.loss}] epoch {e + 1}/{args.epochs}  loss {mean_loss:.2f}  "
            f"lr {opt_e.param_groups[0]['lr']:.2e}",
            flush=True,
        )

    torch.save(encoder, out_dir / "encoder.pt")
    torch.save(decoder, out_dir / "decoder.pt")

    print(f"\n[final] {args.loss} arm, {args.epochs} epochs", flush=True)
    final = evaluate(
        encoder,
        decoder,
        channel,
        config,
        metric,
        args.val_glob,
        eval_snrs,
        img_size,
        args.eval_n,
        device,
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "arm": args.loss,
                "feat_weight": args.feat_weight,
                "epochs": args.epochs,
                "img_size": img_size,
                "seed": args.seed,
                "init_encoder": enc_init,
                "init_decoder": dec_init,
                "init_gate": {k: v for k, v in init_metrics.items() if k != "per_frame_cls_sim"},
                "history": history,
                "final": final,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] wrote {out_dir}/metrics.json", flush=True)
    print("MAMBA_ARM_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
