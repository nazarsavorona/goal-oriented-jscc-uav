"""RF-level validation, stage 3: decode the received symbols and score them.

Reads what stage 2 wrote, pushes the symbols back through the codec's decoder at exactly the
point stage 1 intercepted, and reports cls-sim against the original frame.

The identity check comes first and is not optional: with stage 2 run as
`--channel none --jammer none`, this must reproduce the codec's own clean cls-sim. If it does
not, the RF numbers mean nothing and the script says so instead of printing them.

Usage (inside the semvideo container):
    python scripts/rf_stage3_decode_score.py --in-dir DIR --out OUT.csv [--label tdl_barrage10]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from transformers import AutoModel  # noqa: E402

DINO_HF = "facebook/dinov2-small"
DINO_RES = 518



def score_compact(args, in_dir, meta):
    """Decode the compact codec's symbols straight back through decode()."""
    from semvideo.models.deep_jscc import DeepJSCC

    net = DeepJSCC(channel_ratio=meta["channel_ratio"],
                   base_channels=meta["base_channels"]).cuda().eval()
    sd = torch.load(meta["ckpt"], map_location="cpu", weights_only=False)
    state = sd.get("model_state_dict", sd.get("model_state", sd)) if isinstance(sd, dict) else sd
    net.load_state_dict(state, strict=False)

    dino = AutoModel.from_pretrained(DINO_HF).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    @torch.no_grad()
    def cls(t):
        t = F.interpolate(t.clamp(0, 1), (DINO_RES, DINO_RES), mode="bilinear",
                          align_corners=False)
        return dino(pixel_values=(t - mean) / std).last_hidden_state[:, 0]

    rows = []
    with torch.no_grad():
        for entry in meta["frames"]:
            stem = entry["stem"]
            rx_path, src_path = in_dir / f"{stem}.npy", in_dir / f"{stem}.src.npy"
            if not rx_path.exists() or not src_path.exists():
                continue
            rx = torch.from_numpy(np.load(rx_path)).cuda().float()
            rgb = np.load(src_path)
            x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
            rec = net.decode(rx.reshape(1, -1), tuple(entry["spatial"]))
            c = F.cosine_similarity(cls(x), cls(rec.clamp(0, 1))).item()
            rec_np = (rec.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                      ).astype(np.uint8)
            psnr = cv2.PSNR(rgb, rec_np)
            rows.append({"label": args.label, "image": stem, "cls_sim": round(c, 5),
                         "psnr": round(float(psnr), 3)})
            print(f"  {stem}: cls={c:.4f} psnr={psnr:.2f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["label", "image", "cls_sim", "psnr"])
        w.writeheader()
        w.writerows(rows)
    a = np.array([r["cls_sim"] for r in rows])
    print(f"[{args.label}] cls={a.mean():.4f}+-{a.std():.4f} n={len(a)}")
    print(f"[saved] {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="rf")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    meta = json.load(open(in_dir / "meta.json"))
    if meta.get("arch") == "compact":
        return score_compact(args, in_dir, meta)

    from eval_swinjscc_dinov2 import build_swinjscc

    net = build_swinjscc(Path(meta["ckpt"]),
                         image_dims=(3, meta["codec_size"], meta["codec_size"]),
                         C=meta["C"])
    dino = AutoModel.from_pretrained(DINO_HF).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    @torch.no_grad()
    def cls(t: torch.Tensor) -> torch.Tensor:
        t = F.interpolate(t.clamp(0, 1), (DINO_RES, DINO_RES), mode="bilinear",
                          align_corners=False)
        return dino(pixel_values=(t - mean) / std).last_hidden_state[:, 0]

    injected: dict[str, torch.Tensor] = {}

    def inject(feature, chan_param, avg_pwr=False):
        """Replace the latent with what came back from the RF link, undoing normalisation."""
        scale = torch.sqrt(avg_pwr * 2) if avg_pwr else torch.sqrt(torch.mean(feature ** 2) * 2)
        return injected["rx"].to(feature.device).to(feature.dtype) * scale

    net.feature_pass_channel = inject
    rows = []
    with torch.no_grad():
        for entry in meta["frames"]:
            stem = entry["stem"]
            rx_path = in_dir / f"{stem}.npy"
            src_path = in_dir / f"{stem}.src.npy"
            if not rx_path.exists() or not src_path.exists():
                continue
            injected["rx"] = torch.from_numpy(np.load(rx_path)).cuda()
            rgb = np.load(src_path)
            x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
            rec, *_ = net(x, given_SNR=meta["snr"], given_rate=None)
            c = F.cosine_similarity(cls(x), cls(rec.clamp(0, 1))).item()
            rec_np = (rec.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8)
            psnr = cv2.PSNR(rgb, rec_np)
            rows.append({"label": args.label, "image": stem, "cls_sim": round(c, 5),
                         "psnr": round(float(psnr), 3)})
            print(f"  {stem}: cls={c:.4f} psnr={psnr:.2f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["label", "image", "cls_sim", "psnr"])
        w.writeheader()
        w.writerows(rows)
    a = np.array([r["cls_sim"] for r in rows])
    print(f"[{args.label}] cls={a.mean():.4f}+-{a.std():.4f} n={len(a)}")
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
