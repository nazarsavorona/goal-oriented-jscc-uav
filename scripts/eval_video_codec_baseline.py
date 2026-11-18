"""Genuine video-codec baselines (H.264 / H.265) over the separated channel.

The codec of the paper is frame-wise image JSCC, and the classical baseline of the main
tables encodes each frame as its own single-frame MP4 -- intra-only, so it exploits no
temporal redundancy either. Neither is a comparison against a real video pipeline.

This encodes a *sequence* through libx264 / libx265 with inter-frame prediction at a matched
average bitrate, then measures two things the frame-wise comparison cannot show:

  1. clean-channel quality at matched bpp -- where temporal prediction helps the classical
     codec and the comparison is genuinely unfavourable to us, and
  2. what one corrupted packet does to the rest of the GOP -- error propagation, which is the
     failure mode a learned codec does not have and the reason the comparison is not simply
     "the video codec wins".

Sequence caveat: the archived DJI validation frames are sampled every 30th frame (1 fps), so
the temporal correlation available here is weaker than a real 30 fps downlink would offer.
That biases the comparison *in the video codec's disfavour* on point 1 and is stated as a
limit rather than hidden; point 2 does not depend on the frame rate.

Usage:
    python scripts/eval_video_codec_baseline.py --frames-dir DIR --out OUT.csv \
        [--codecs libx264 libx265] [--target-bpp 1.17] [--gop 12] [--n 120]
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from transformers import AutoModel  # noqa: E402

DINO_HF = "facebook/dinov2-small"
DINO_RES = 518


def encode_sequence(frames: list[np.ndarray], codec: str, bitrate: int, gop: int,
                    path: Path) -> int:
    """Encode RGB frames as a real video stream. Returns file size in bytes."""
    import av

    container = av.open(str(path), mode="w")
    h, w = frames[0].shape[:2]
    stream = container.add_stream(codec, rate=30)
    stream.width, stream.height = w, h
    stream.pix_fmt = "yuv420p"
    stream.bit_rate = bitrate
    stream.gop_size = gop
    for f in frames:
        frame = av.VideoFrame.from_ndarray(f, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path.stat().st_size


def decode_sequence(path: Path) -> list[np.ndarray]:
    import av

    out = []
    container = av.open(str(path))
    for frame in container.decode(video=0):
        out.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return out


def corrupt_stream(path: Path, out_path: Path, drop_fraction: float, seed: int) -> None:
    """Drop a fraction of the compressed packets: the separated pipeline's failure mode.

    A channel that loses a packet does not politely lose a frame; the decoder carries the
    damage forward until the next keyframe. This is what error propagation means and it is
    invisible in a frame-wise comparison.
    """
    import av

    rng = np.random.default_rng(seed)
    src = av.open(str(path))
    dst = av.open(str(out_path), mode="w")
    in_stream = src.streams.video[0]
    out_stream = dst.add_stream_from_template(in_stream)
    for packet in src.demux(in_stream):
        if packet.dts is None:
            continue
        # Keyframes survive: dropping them would make the comparison about container
        # recovery rather than about prediction carrying damage forward.
        if not packet.is_keyframe and rng.random() < drop_fraction:
            continue
        packet.stream = out_stream
        dst.mux(packet)
    dst.close()
    src.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--codecs", nargs="+", default=["libx264", "libx265"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--target-bpp", type=float, default=1.17,
                    help="Average bits per pixel to match (the JPEG q=70 operating point "
                         "the separated baseline uses).")
    ap.add_argument("--gop", type=int, default=12)
    ap.add_argument("--drop-fractions", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.10])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.frames_dir + "/*.jpg") + glob.glob(args.frames_dir + "/*.png"))
    if not paths:
        raise SystemExit(f"no frames under {args.frames_dir}")
    paths = paths[: args.n]
    frames = []
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        bgr = cv2.resize(bgr, (args.size, args.size), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    print(f"[setup] {len(frames)} frames at {args.size} px, GOP {args.gop}")

    dino = AutoModel.from_pretrained(DINO_HF).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    @torch.no_grad()
    def cls(rgb: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
        t = F.interpolate(t.clamp(0, 1), (DINO_RES, DINO_RES), mode="bilinear",
                          align_corners=False)
        return dino(pixel_values=(t - mean) / std).last_hidden_state[:, 0]

    refs = [cls(f) for f in frames]
    # bits/pixel -> bits/second at the 30 fps the container declares.
    px = args.size * args.size
    bitrate = int(args.target_bpp * px * 30)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["codec", "drop_fraction", "gop", "bpp_actual", "frame_idx", "cls_sim"])
        for codec in args.codecs:
            with tempfile.TemporaryDirectory() as td:
                clean = Path(td) / f"{codec}.mp4"
                nbytes = encode_sequence(frames, codec, bitrate, args.gop, clean)
                bpp = nbytes * 8 / (len(frames) * px)
                print(f"\n[{codec}] {nbytes} bytes, {bpp:.3f} bpp actual "
                      f"(target {args.target_bpp})")
                for drop in args.drop_fractions:
                    if drop == 0.0:
                        dec = decode_sequence(clean)
                    else:
                        damaged = Path(td) / f"{codec}_d{drop}.mp4"
                        try:
                            corrupt_stream(clean, damaged, drop, args.seed)
                            dec = decode_sequence(damaged)
                        except Exception as exc:  # noqa: BLE001
                            # Remuxing a damaged stream needs a decoder for that codec to be
                            # present (AV1 needs libdav1d, which this image lacks). Report and
                            # move on rather than losing the clean-channel measurement.
                            print(f"  drop={drop:<5.2f} SKIPPED ({type(exc).__name__}: {exc})",
                                  flush=True)
                            continue
                    sims = []
                    for i, ref in enumerate(refs):
                        if i >= len(dec):
                            # Lost tail: the decoder produced fewer frames than were sent.
                            sims.append(0.0)
                            w.writerow([codec, drop, args.gop, round(bpp, 4), i, 0.0])
                            continue
                        c = F.cosine_similarity(ref, cls(dec[i])).item()
                        sims.append(c)
                        w.writerow([codec, drop, args.gop, round(bpp, 4), i, round(c, 5)])
                    a = np.array(sims)
                    print(f"  drop={drop:<5.2f} cls={a.mean():.4f}+-{a.std():.4f} "
                          f"n={len(a)} decoded={len(dec)}", flush=True)
    print(f"[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
