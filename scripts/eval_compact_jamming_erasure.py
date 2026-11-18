"""Compact 5.5 M DeepJSCC codec across the jamming and erasure grid.

The headline objective result is measured on the compact 5.5 M codec, while the jamming,
erasure, cross-corpus and deployment results are measured on the SwinJSCC-capacity codec.
This script evaluates the compact codec under jamming and erasure as well.

It uses the same channel modules as the large-codec sweeps, so the two capacities are
compared through the same threat implementations.

Usage:
    python scripts/eval_compact_jamming_erasure.py --ckpt CKPT --val-dir DIR \
        --out OUT.csv [--n 375] [--codec-size 256] [--snr 20]
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from transformers import AutoModel  # noqa: E402

from semvideo.channel.awgn import AWGNChannel  # noqa: E402
from semvideo.channel.jamming import JammingChannel  # noqa: E402
from semvideo.channel.packet_loss import BernoulliErasureChannel  # noqa: E402
from semvideo.channel.jamming_reactive import ReactiveJammer  # noqa: E402
from semvideo.models.deep_jscc import DeepJSCC  # noqa: E402

DINO_HF = "facebook/dinov2-small"
DINO_RES = 518


class _AwgnThen(torch.nn.Module):
    """Background AWGN followed by a second impairment, matching the large-codec path.

    The jamming sweeps on the large codec run the jammer on top of an AWGN link at the
    stated SNR; the reactive jammer does not add background noise itself, so it is composed
    here explicitly rather than being handed an SNR it ignores.
    """

    def __init__(self, snr_db: float, second: torch.nn.Module | None,
                 pass_snr: bool = True) -> None:
        super().__init__()
        self.awgn = AWGNChannel(snr_db=snr_db)
        self.second = second
        self.snr_db = float(snr_db)
        # The erasure layer treats snr_db as a residual-noise setting for surviving
        # symbols; the paper's erasure evaluation is blind zero-fill with no extra
        # noise on top of the link, so it must not receive it.
        self.pass_snr = pass_snr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.awgn(x)
        if self.second is None:
            return y
        if self.pass_snr:
            return self.second(y, snr_db=self.snr_db)
        return self.second(y)


def build_channel(kind: str, snr_db: float, level: float) -> torch.nn.Module:
    """kind in {none, barrage, pulsed, single_tone, swept, reactive, erasure}."""
    if kind == "none":
        return _AwgnThen(snr_db, None)
    if kind == "erasure":
        # Blind zero-fill erasure on top of the AWGN link, as in the large-codec sweeps:
        # the erasure layer itself adds no further noise (snr_db=None).
        return _AwgnThen(snr_db, BernoulliErasureChannel(p=level, snr_db=None),
                         pass_snr=False)
    if kind == "reactive":
        return _AwgnThen(
            snr_db,
            ReactiveJammer(jnr_db=level, detect_threshold=0.5, react_latency=8,
                           hold_duration=64),
        )
    # The parametric jammers carry their own background AWGN term.
    return JammingChannel(jammer_type=kind, jnr_db=level, snr_db=snr_db)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=375)
    ap.add_argument("--codec-size", type=int, default=256)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--channel-ratio", type=float, default=1 / 6)
    ap.add_argument("--base-channels", type=int, default=64)
    ap.add_argument("--jnr-grid", type=float, nargs="+", default=[0.0, 5.0, 10.0, 15.0])
    ap.add_argument("--erasure-grid", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    ap.add_argument("--jammers", nargs="+",
                    default=["none", "barrage", "pulsed", "single_tone", "swept", "reactive"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    net = DeepJSCC(channel_ratio=args.channel_ratio, base_channels=args.base_channels).cuda().eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = sd.get("model_state_dict", sd.get("model_state", sd)) if isinstance(sd, dict) else sd
    miss, unexp = net.load_state_dict(state, strict=False)
    assert not miss and not unexp, f"arch mismatch: {len(miss)} missing, {len(unexp)} unexpected"
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[setup] {args.ckpt}: {n_params:,} parameters, codec {args.codec_size} px")

    dino = AutoModel.from_pretrained(DINO_HF).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    @torch.no_grad()
    def cls(x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x.clamp(0, 1), (DINO_RES, DINO_RES), mode="bilinear",
                          align_corners=False)
        return dino(pixel_values=(x - mean) / std).last_hidden_state[:, 0]

    paths = sorted(glob.glob(args.val_dir + "/*.jpg") + glob.glob(args.val_dir + "/*.png"))
    if not paths:
        raise SystemExit(f"no frames under {args.val_dir}")
    idx = np.linspace(0, len(paths) - 1, min(args.n, len(paths)), dtype=int)
    paths = [paths[int(i)] for i in idx]

    # Cache inputs and their reference features once.
    frames, refs = [], []
    with torch.no_grad():
        for p in paths:
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
            t = F.interpolate(t, (args.codec_size, args.codec_size), mode="bilinear",
                              align_corners=False)
            frames.append(t)
            refs.append(cls(t))
    print(f"[setup] {len(frames)} frames cached")

    cells: list[tuple[str, float]] = []
    for jam in args.jammers:
        if jam == "none":
            cells.append(("none", 0.0))
        else:
            cells.extend((jam, j) for j in args.jnr_grid)
    cells.extend(("erasure", e) for e in args.erasure_grid)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["ckpt", "params", "channel", "level", "snr", "image", "cls_sim"])
        stem = Path(args.ckpt).parent.name
        for kind, level in cells:
            ch = build_channel(kind, args.snr, level).cuda().eval()
            sims = []
            with torch.no_grad():
                for t, ref in zip(frames, refs):
                    rec = net(t, ch)
                    c = F.cosine_similarity(ref, cls(rec.clamp(0, 1))).item()
                    sims.append(c)
                    w.writerow([stem, n_params, kind, level, args.snr,
                                Path(paths[len(sims) - 1]).stem, round(c, 5)])
            a = np.array(sims)
            print(f"  {kind:>12s} level={level:>5.2f}  cls={a.mean():.4f}+-{a.std():.4f}  "
                  f"n={len(a)}", flush=True)
    print(f"[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
