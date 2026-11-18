"""Does a common lambda give the four frozen supervisors the same effective distillation strength?

The supervisor matrix (the paper's detector-supervisor ablation) trains every arm at the same
detector-feature weight
lambda = 0.5. The detector term is a raw L1 over FPN-stage features (stages 4, 6, 9, weights
1.0 / 0.5 / 0.25), so supervisors with different feature scales, channel counts or activation
statistics can contribute differently sized gradients at the same lambda. This script measures
that directly, on identical inputs for all four supervisors:

  inputs   DJI val frames (sorted, linspace subset), codec input 256 px as in the matched-run
           protocol; reconstructions from each seed's pixel-only arm (supervisor-independent)
           through AWGN at a fixed SNR, per-frame seeded;
  measures for each supervisor s: L_s = DetectorFeatureLoss_s(x_hat, x), and the L2 norm of
           d(lambda * L_s)/d x_hat; for reference the same norm for the pixel part
           MSE + 0.15 * (1 - MS-SSIM) that every arm shares;
  proves   the four backbones are four different networks (parameter count, weight hash), so
           the loss class's silent fallback to yolov8m.pt did not trigger.

The loss module and its arguments are exactly those train.py builds for --loss perceptual_yolo.

Usage (inside the semvideo Docker image, cwd = repo root, where the .pt files live):
    python scripts/diag_supervisor_loss_scale.py --val-dir /data/dji/frames_512_v1/val \
        --out experiments/diag_supervisor_scale_2026_10_01/result.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from semvideo.channel.awgn import AWGNChannel
from semvideo.models.deep_jscc import DeepJSCC
from semvideo.training.detector_feature_loss import DetectorFeatureLoss
from semvideo.training.losses import ms_ssim_loss

SUPERVISORS = {"v8m": "yolov8m.pt", "v8x": "yolov8x.pt", "v11m": "yolo11m.pt", "v10m": "yolov10m.pt"}
PIXEL_ARMS = {  # the pixel-only arm of each seed of the supervisor-ablation table
    "42": "r1pixel_s42_today_20260822_025747",
    "1": "r1pixel_s1_20260822_031516",
    "1337": "r1pixel_s1337_20260822_035301",
    "7": "r1pixel_s7_20260824_091941",
    "2024": "r1pixel_s2024_20260826_065721",
}
CODEC_PX = 256
LAMBDA = 0.5


def load_codec(path: str, device: str) -> DeepJSCC:
    net = DeepJSCC(channel_ratio=1 / 6, base_channels=64).to(device).eval()
    sd = torch.load(path, map_location="cpu", weights_only=False)
    state = sd.get("model_state_dict", sd.get("model_state", sd)) if isinstance(sd, dict) else sd
    missing, _ = net.load_state_dict(state, strict=False)
    if len(missing) > 0.3 * len(net.state_dict()):
        raise RuntimeError(f"architecture mismatch for {path}")
    return net


def load_frame(path: str, device: str) -> torch.Tensor:
    import cv2
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    return F.interpolate(t, (CODEC_PX, CODEC_PX), mode="bilinear", align_corners=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--exp-dir", default="experiments")
    ap.add_argument("--n", type=int, default=96, help="frames per seed (linspace over the split)")
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device

    paths = sorted(glob.glob(args.val_dir + "/*.jpg") + glob.glob(args.val_dir + "/*.png"))
    idx = np.linspace(0, len(paths) - 1, min(args.n, len(paths)), dtype=int)
    paths = [paths[int(i)] for i in idx]

    losses, ident = {}, {}
    for name, w in SUPERVISORS.items():
        if not Path(w).exists():
            raise SystemExit(f"{w} not in cwd: the loss would silently fall back to yolov8m.pt")
        m = DetectorFeatureLoss(weights_path=w, device=dev)
        n_par = sum(p.numel() for p in m.backbone.parameters())
        h = hashlib.sha256()
        for p in list(m.backbone.parameters())[:4]:
            h.update(p.detach().cpu().numpy().tobytes())
        ident[name] = {"weights": w, "backbone_params": int(n_par), "head_hash": h.hexdigest()[:16]}
        losses[name] = m
    if len({v["head_hash"] for v in ident.values()}) != len(SUPERVISORS):
        raise SystemExit("two supervisors load the same backbone weights")

    rec = {s: {"loss": [], "grad": []} for s in SUPERVISORS}
    rec["pixel_part"] = {"loss": [], "grad": []}
    for seed, run in PIXEL_ARMS.items():
        net = load_codec(f"{args.exp_dir}/{run}/best.pt", dev)
        for k, p in enumerate(paths):
            x = load_frame(p, dev)
            torch.manual_seed(1000 + k)
            ch = AWGNChannel(snr_db=args.snr).to(dev)
            with torch.no_grad():
                x_hat0 = net(x, channel=ch).clamp(0, 1)
            # pixel part shared by every arm
            xh = x_hat0.clone().requires_grad_(True)
            lp = F.mse_loss(xh, x) + 0.15 * ms_ssim_loss(xh, x)
            (g,) = torch.autograd.grad(lp, xh)
            rec["pixel_part"]["loss"].append(float(lp))
            rec["pixel_part"]["grad"].append(float(g.norm()))
            for s, m in losses.items():
                xh = x_hat0.clone().requires_grad_(True)
                ls = m(xh, x)
                (g,) = torch.autograd.grad(LAMBDA * ls, xh)
                rec[s]["loss"].append(float(ls))
                rec[s]["grad"].append(float(g.norm()))
        print(f"seed {seed}: done ({len(paths)} frames)", flush=True)

    summary = {}
    for k, v in rec.items():
        l, g = np.array(v["loss"]), np.array(v["grad"])
        summary[k] = {"loss_mean": float(l.mean()), "loss_sd": float(l.std()),
                      "grad_norm_mean": float(g.mean()), "grad_norm_sd": float(g.std()), "n": int(len(l))}
    ref = summary["v8m"]
    for s in SUPERVISORS:
        summary[s]["loss_ratio_to_v8m"] = summary[s]["loss_mean"] / ref["loss_mean"]
        summary[s]["grad_ratio_to_v8m"] = summary[s]["grad_norm_mean"] / ref["grad_norm_mean"]
        summary[s]["grad_ratio_to_pixel_part"] = summary[s]["grad_norm_mean"] / summary["pixel_part"]["grad_norm_mean"]
    out = {"setting": {"lambda": LAMBDA, "codec_px": CODEC_PX, "snr_db": args.snr,
                       "frames_per_seed": len(paths), "seeds": list(PIXEL_ARMS),
                       "inputs": "pixel-only arm reconstructions (supervisor-independent)"},
           "supervisors": ident, "summary": summary, "per_frame": rec}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for s in list(SUPERVISORS) + ["pixel_part"]:
        r = summary[s]
        extra = (f"  loss x{r['loss_ratio_to_v8m']:.2f}  grad x{r['grad_ratio_to_v8m']:.2f} of v8m, "
                 f"x{r['grad_ratio_to_pixel_part']:.2f} of pixel part") if s in SUPERVISORS else ""
        print(f"{s:10s} loss {r['loss_mean']:.4f}  |grad| {r['grad_norm_mean']:.4e}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
