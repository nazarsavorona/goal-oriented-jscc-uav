"""Goal-oriented metric: DINOv2 feature similarity between raw and reconstructed.

Replaces detection-mAP (which proved unreliable on our high-altitude drone
footage where vehicles are 1.5-2.5% of frame width). DINOv2
ViT-S/14 self-supervised features are robust, dense, and don't require
ground-truth labels of any kind.

Two metrics per (model, SNR) cell:
  1. **CLS cosine similarity**: 384-d global semantic feature; 1.0 = identical.
     Captures "is this the same scene?"
  2. **Patch cosine similarity**: per-token average over 37×37 grid; 1.0 = identical.
     Captures "is the spatial structure preserved at the patch level?"

References:
  - Oquab et al., DINOv2 (Meta AI 2023, arxiv 2304.07193)
  - Used by DiT-JSCC, Adapt-ICMH, SemCLIP literature for goal-oriented eval
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from semvideo.channel.awgn import AWGNChannel
from semvideo.channel.fading import RayleighFadingChannel
from semvideo.models.deep_jscc import DeepJSCC


SOURCE_VIDEO = "data/source_video.mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DINOv2 ViT-S/14 expects multiples of 14. 518 = 14*37 → 37x37 grid (1369 patches).
DINO_RES = 518


CHECKPOINTS = [
    ("P12", "experiments/p12_deepjscc_mixed_512_*/best.pt", 512),
    ("P13", "experiments/p13_deepjscc_yolofeat_512_*/best.pt", 512),
    ("P14", "experiments/p14_deepjscc_visdronefeat_512_*/best.pt", 512),
    ("P15", "experiments/p15_deepjscc_ch3_visdronefeat_512_*/best.pt", 512),
]


def build_jscc(ck_path: str, native_size: int):
    matches = sorted(glob.glob(ck_path))
    if not matches:
        return None, None
    ck_path = matches[-1]
    ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    n = cfg.get("model")
    if n == "deepjscc":
        m = DeepJSCC(channel_ratio=cfg.get("channel_ratio", 1/6),
                     base_channels=64).to(DEVICE)
    else:
        return None, None
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, native_size


def jscc_reconstruct(model, native_size, frame_1080, snr_db, channel="awgn"):
    h, w = frame_1080.shape[:2]
    small = cv2.resize(frame_1080, (native_size, native_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    if channel == "awgn":
        ch = AWGNChannel(snr_db=float(snr_db)).to(DEVICE)
    elif channel == "rayleigh":
        ch = RayleighFadingChannel(snr_db=float(snr_db)).to(DEVICE)
    else:
        raise ValueError(channel)
    with torch.no_grad():
        rec = model(x, channel=ch, snr_db=float(snr_db)).clamp(0, 1)
    rec_np = (rec.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    rec_bgr = cv2.cvtColor(rec_np, cv2.COLOR_RGB2BGR)
    return cv2.resize(rec_bgr, (w, h), interpolation=cv2.INTER_CUBIC)


class DINOv2FeatureExtractor:
    """Wrapper around DINOv2 ViT-S/14, yielding (cls_token, patch_tokens)."""

    def __init__(self):
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", verbose=False,
        ).eval().to(DEVICE)
        # ImageNet normalisation (DINOv2 is trained on ImageNet)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((DINO_RES, DINO_RES), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def features(self, frame_bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(rgb).unsqueeze(0).to(DEVICE)
        out = self.model.forward_features(x)
        cls = out["x_norm_clstoken"]              # (1, 384)
        patches = out["x_norm_patchtokens"]       # (1, 1369, 384)
        return cls.squeeze(0), patches.squeeze(0)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity. For patch tokens, mean-cosine across positions."""
    if a.dim() == 1:
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    # (N, D) – average per-position cosine
    return float(F.cosine_similarity(a, b, dim=-1).mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=30)
    parser.add_argument("--snrs", type=int, nargs="+", default=[0, 5, 10, 15, 20])
    parser.add_argument("--channels", nargs="+", default=["awgn", "rayleigh"])
    parser.add_argument("--frame-start", type=int, default=10850)
    parser.add_argument("--frame-end", type=int, default=11800)
    parser.add_argument("--source", default=SOURCE_VIDEO)
    parser.add_argument("--out", default="experiments/dinov2_metric_2026_05_02")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DINOv2 ViT-S/14...")
    dino = DINOv2FeatureExtractor()

    # Sample frames
    cap = cv2.VideoCapture(args.source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(args.frame_start, min(args.frame_end, total - 1),
                           args.n_frames, dtype=int).tolist()
    raw_frames = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if ok:
            raw_frames.append((int(fi), f))
    cap.release()
    print(f"Sampled {len(raw_frames)} frames")

    # Reference DINOv2 features – compute once per frame
    print("Computing DINOv2 features on raw 1080p frames...")
    raw_cls = []
    raw_patches = []
    for fi, f in raw_frames:
        cls, p = dino.features(f)
        raw_cls.append(cls)
        raw_patches.append(p)

    # Bicubic-only controls – establish the resolution-loss baseline
    rows = []
    print("\nBicubic controls (no codec):")
    for ctrl_native in (256, 512):
        clsims, patchsims = [], []
        for (fi, f), (rc, rp) in zip(raw_frames, zip(raw_cls, raw_patches)):
            H, W = f.shape[:2]
            small = cv2.resize(f, (ctrl_native, ctrl_native), interpolation=cv2.INTER_AREA)
            up = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
            cls, p = dino.features(up)
            clsims.append(cosine_sim(rc, cls))
            patchsims.append(cosine_sim(rp, p))
        for snr in args.snrs:
            for chan in args.channels:
                rows.append({
                    "model": f"bicubic{ctrl_native}", "snr_db": snr, "channel": chan,
                    "dino_cls_sim": float(np.mean(clsims)),
                    "dino_patch_sim": float(np.mean(patchsims)),
                    "dino_cls_std": float(np.std(clsims)),
                    "dino_patch_std": float(np.std(patchsims)),
                    "n_frames": len(clsims),
                })
        print(f"  bicubic{ctrl_native}: cls_sim={np.mean(clsims):.4f}±{np.std(clsims):.4f}, "
              f"patch_sim={np.mean(patchsims):.4f}±{np.std(patchsims):.4f}")

    # Per-checkpoint eval
    for label, ck_glob, native in CHECKPOINTS:
        m, native_size = build_jscc(ck_glob, native)
        if m is None:
            print(f"\n  {label}: SKIP (no checkpoint matched)")
            continue
        print(f"\n--- {label} (native={native_size}) ---")
        for chan in args.channels:
            for snr in args.snrs:
                clsims, patchsims = [], []
                for (fi, f), (rc, rp) in zip(raw_frames, zip(raw_cls, raw_patches)):
                    rec = jscc_reconstruct(m, native_size, f, snr, channel=chan)
                    cls, p = dino.features(rec)
                    clsims.append(cosine_sim(rc, cls))
                    patchsims.append(cosine_sim(rp, p))
                rows.append({
                    "model": label, "snr_db": snr, "channel": chan,
                    "dino_cls_sim": float(np.mean(clsims)),
                    "dino_patch_sim": float(np.mean(patchsims)),
                    "dino_cls_std": float(np.std(clsims)),
                    "dino_patch_std": float(np.std(patchsims)),
                    "n_frames": len(clsims),
                })
                print(f"  {label} @{chan} SNR={snr}: cls_sim={np.mean(clsims):.4f}, "
                      f"patch_sim={np.mean(patchsims):.4f}")
        del m
        torch.cuda.empty_cache()

    # CSV
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved -> {csv_path}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for chan in args.channels:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for label, _, _ in CHECKPOINTS + [("bicubic256", "", 0), ("bicubic512", "", 0)]:
            rs = sorted([r for r in rows if r["model"] == label and r["channel"] == chan],
                        key=lambda r: r["snr_db"])
            if not rs: continue
            xs = [r["snr_db"] for r in rs]
            for ax, k in zip(axes, ["dino_cls_sim", "dino_patch_sim"]):
                ys = [r[k] for r in rs]
                ax.plot(xs, ys, marker="o", linewidth=2, label=label)
        for ax, t in zip(axes, ["DINOv2 [CLS] (global)", "DINOv2 patch (mean spatial)"]):
            ax.set_xlabel("Channel SNR (dB)")
            ax.set_ylabel("Cosine similarity to raw")
            ax.set_title(f"{t} – {chan} channel")
            ax.grid(alpha=0.3)
            ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"dinov2_{chan}.png", dpi=140, bbox_inches="tight")
        plt.close()
    print(f"Plots in {out_dir}")


if __name__ == "__main__":
    main()
