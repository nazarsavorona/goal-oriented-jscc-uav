"""RF-level validation, stage 1: dump the codec's coded latent symbols to disk.

Every jammer and every impairment in the paper's main sweeps
acts on the encoder's latent symbols, not on a modulated RF waveform, so those results cannot
be read as RF-level claims. An RF-level check needs modulation, pulse shaping, OFDM and a jammer
that acts on the waveform -- which means Sionna, and Sionna cannot share this container
(installing it pulls a torch that breaks the codec's torchvision/torchaudio).

So the pipeline is split across three stages that hand data over as .npy files:

    stage 1 (this file, codec container) : frames -> encoder -> normalised latent symbols
    stage 2 (Sionna container)           : symbols -> QAM -> OFDM -> channel+jammer -> symbols
    stage 3 (codec container)            : symbols -> decoder -> reconstruction -> cls-sim

Stage 1 also records the per-tensor power normalisation the codec applies internally, so
stage 3 can undo it exactly and the round trip is lossless when stage 2 is a no-op. That
no-op round trip is the pipeline's own correctness check: run stage 2 with --channel none and
stage 3 must reproduce the codec's clean cls-sim to numerical precision.

Usage (inside the semvideo container):
    python scripts/rf_stage1_dump_latents.py --ckpt CKPT --val-dir DIR --out-dir DIR [--n 50]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")



def dump_compact(args, out_dir):
    """The 5.5M convolutional codec: encode() already yields power-normalised symbols."""
    import cv2 as _cv2
    from semvideo.models.deep_jscc import DeepJSCC

    net = DeepJSCC(channel_ratio=args.channel_ratio,
                   base_channels=args.base_channels).cuda().eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = sd.get("model_state_dict", sd.get("model_state", sd)) if isinstance(sd, dict) else sd
    miss, unexp = net.load_state_dict(state, strict=False)
    assert not miss and not unexp, f"arch mismatch: {len(miss)} missing, {len(unexp)} unexpected"

    paths = sorted(glob.glob(args.val_dir + "/*.jpg") + glob.glob(args.val_dir + "/*.png"))
    paths = paths[:args.n]
    if not paths:
        raise SystemExit(f"no frames under {args.val_dir}")

    meta = {"ckpt": args.ckpt, "codec_size": args.codec_size, "C": args.C,
            "snr": args.snr, "arch": "compact", "channel_ratio": args.channel_ratio,
            "base_channels": args.base_channels, "frames": []}
    with torch.no_grad():
        for p in paths:
            bgr = _cv2.imread(p)
            if bgr is None:
                continue
            bgr = _cv2.resize(bgr, (args.codec_size, args.codec_size),
                              interpolation=_cv2.INTER_AREA)
            rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
            z = net.encoder(x)
            spatial = list(z.shape[1:])
            tx = net.encode(x)
            stem = Path(p).stem
            np.save(out_dir / f"{stem}.npy", tx.cpu().numpy().astype(np.float32))
            np.save(out_dir / f"{stem}.src.npy", rgb)
            meta["frames"].append({"stem": stem, "scale": 1.0,
                                   "shape": list(tx.shape), "spatial": spatial})
            print(f"  {stem}: {list(tx.shape)} spatial={spatial}", flush=True)

    json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)
    n_sym = int(np.prod(meta["frames"][0]["shape"])) if meta["frames"] else 0
    print(f"[saved] {len(meta['frames'])} frames x {n_sym} real symbols -> {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/experiments/p49_p46_awgn_finetune/best.pt")
    ap.add_argument("--val-dir", default="/data/dji/frames_512_v1/val")
    ap.add_argument("--out-dir", default="/workspace/experiments/rf_stage1")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--codec-size", type=int, default=512)
    ap.add_argument("--C", type=int, default=96)
    ap.add_argument("--arch", default="swinjscc", choices=["swinjscc", "compact"],
                    help="swinjscc: the SwinJSCC-capacity codec. compact: the 5.5M convolutional "
                         "codec, whose encode/decode need no channel monkey-patch.")
    ap.add_argument("--channel-ratio", type=float, default=1 / 6)
    ap.add_argument("--base-channels", type=int, default=64)
    ap.add_argument("--snr", type=float, default=20.0,
                    help="SNR the encoder is conditioned on (the codec is SNR-adaptive).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.arch == "compact":
        return dump_compact(args, out_dir)

    from eval_swinjscc_dinov2 import build_swinjscc  # proven loader

    net = build_swinjscc(Path(args.ckpt), image_dims=(3, args.codec_size, args.codec_size),
                         C=args.C)

    paths = sorted(glob.glob(args.val_dir + "/*.jpg") + glob.glob(args.val_dir + "/*.png"))[:args.n]
    if not paths:
        raise SystemExit(f"no frames under {args.val_dir}")

    captured: dict[str, object] = {}
    original_fpc = net.feature_pass_channel

    def capture(feature, chan_param, avg_pwr=False):
        """Intercept at exactly the point the jamming sweeps intercept."""
        if avg_pwr:
            tx = feature / torch.sqrt(avg_pwr * 2)
            scale = torch.sqrt(avg_pwr * 2)
        else:
            pwr = torch.mean(feature ** 2) * 2
            tx = feature / torch.sqrt(pwr)
            scale = torch.sqrt(pwr)
        captured["tx"] = tx.detach().cpu().numpy().astype(np.float32)
        captured["scale"] = float(scale)
        captured["shape"] = list(tx.shape)
        # Pass through untouched: stage 1 only observes.
        return tx * scale

    net.feature_pass_channel = capture
    meta = {"ckpt": args.ckpt, "codec_size": args.codec_size, "C": args.C,
            "snr": args.snr, "frames": []}
    with torch.no_grad():
        for p in paths:
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            bgr = cv2.resize(bgr, (args.codec_size, args.codec_size),
                             interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
            net(x, given_SNR=args.snr, given_rate=None)
            stem = Path(p).stem
            np.save(out_dir / f"{stem}.npy", captured["tx"])
            # The frame itself travels with the symbols so stage 3 needs no second read.
            np.save(out_dir / f"{stem}.src.npy", rgb)
            meta["frames"].append({"stem": stem, "scale": captured["scale"],
                                   "shape": captured["shape"]})
            print(f"  {stem}: {captured['shape']} scale={captured['scale']:.4f}", flush=True)

    json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)
    net.feature_pass_channel = original_fpc
    n_sym = int(np.prod(meta["frames"][0]["shape"])) if meta["frames"] else 0
    print(f"[saved] {len(meta['frames'])} frames x {n_sym} real symbols -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
