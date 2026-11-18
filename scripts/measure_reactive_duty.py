"""Measure the reactive jammer's duty cycle and its total injected power.

The JNR-sweep reactive setting is said to inject less total power than an
equal-JNR barrage jammer "because only gated symbols are hit". This script checks
that statement. The two families reference JNR to different quantities:

  barrage   jammer_power = noise_var * JNR_lin,  noise_var = P / SNR_lin
            -> jammer-to-NOISE, total injected = JNR_lin * P/SNR_lin * N
  reactive  per_symbol_std = sqrt(energy_i * JNR_lin) on gated symbols
            -> jammer-to-SIGNAL per jammed symbol, total = JNR_lin * sum_gated energy_i

so  total_reactive / total_barrage = (sum_gated energy_i / (N * P)) * SNR_lin,
i.e. the claim holds only if the gated energy fraction is below 1/SNR_lin
(1% at SNR = 20 dB). This script measures that fraction on real codec latents.

Usage:
    python measure_reactive_duty.py --ckpt CKPT --val-dir DIR --n 50 \
        --snr 20 --threshold 0.5 --latency 8 --hold 64
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

from semvideo.channel.jamming_reactive import ReactiveJammer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--latency", type=int, default=8)
    ap.add_argument("--hold", type=int, default=64)
    ap.add_argument("--codec-size", type=int, default=512)
    ap.add_argument("--C", type=int, default=96)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from eval_swinjscc_dinov2 import build_swinjscc

    net = build_swinjscc(args.ckpt, image_dims=(3, args.codec_size, args.codec_size), C=args.C)
    jam = ReactiveJammer(jnr_db=10.0, detect_threshold=args.threshold,
                         react_latency=args.latency, hold_duration=args.hold)

    paths = sorted(glob.glob(args.val_dir + "/*.jpg") + glob.glob(args.val_dir + "/*.png"))
    idx = np.linspace(0, len(paths) - 1, min(args.n, len(paths)), dtype=int)
    paths = [paths[int(i)] for i in idx]

    # Capture the pre-channel latent by hooking the codec's channel stage.
    latents: list[torch.Tensor] = []
    original = net.feature_pass_channel

    def capture(z, *a, **k):
        latents.append(z.detach().flatten(1).clone())
        return original(z, *a, **k)

    net.feature_pass_channel = capture

    with torch.no_grad():
        for p in paths:
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            side = min(bgr.shape[:2])
            y0, x0 = (bgr.shape[0] - side) // 2, (bgr.shape[1] - side) // 2
            crop = cv2.resize(bgr[y0:y0 + side, x0:x0 + side],
                              (args.codec_size, args.codec_size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
            net(t, given_SNR=args.snr, given_rate=None)

    net.feature_pass_channel = original
    if not latents:
        raise SystemExit("no latents captured")

    duty_list, energy_frac_list = [], []
    for z in latents:
        # unit-power normalise as the codec's channel stage does
        z = z / z.pow(2).mean().clamp(min=1e-12).sqrt()
        energy = z.abs() ** 2
        mean_e = float(energy.mean().clamp(min=1e-12).item())
        mask = jam._build_jam_mask(energy, mean_e)
        duty_list.append(float(mask.float().mean().item()))
        energy_frac_list.append(float((energy * mask).sum().item()
                                      / energy.sum().clamp(min=1e-12).item()))

    duty = float(np.mean(duty_list))
    efrac = float(np.mean(energy_frac_list))
    snr_lin = 10.0 ** (args.snr / 10.0)
    ratio = efrac * snr_lin  # total_reactive / total_barrage at equal nominal JNR

    print(f"[setup] n={len(latents)} frames, {latents[0].numel()} symbols/frame, "
          f"threshold={args.threshold}x mean, latency={args.latency}, hold={args.hold}")
    print(f"duty cycle (fraction of symbols gated)      : {duty:.4f}  ({100*duty:.1f}%)")
    print(f"fraction of total latent energy that is gated: {efrac:.4f}  ({100*efrac:.1f}%)")
    print(f"SNR={args.snr:.0f} dB -> SNR_lin={snr_lin:.0f}")
    print(f"total injected power, reactive / equal-JNR barrage = {ratio:.1f}x")
    verdict = ("BELOW barrage (paper's claim holds)" if ratio < 1.0
               else f"ABOVE barrage by {ratio:.0f}x (paper's claim is FALSE)")
    print(f"VERDICT: {verdict}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "n_frames": len(latents), "symbols_per_frame": int(latents[0].numel()),
            "threshold_x_mean": args.threshold, "react_latency": args.latency,
            "hold_duration": args.hold, "snr_db": args.snr,
            "duty_cycle": round(duty, 5), "gated_energy_fraction": round(efrac, 5),
            "reactive_over_barrage_total_power": round(ratio, 3),
            "claim_total_below_barrage_holds": bool(ratio < 1.0),
        }, indent=1))
        print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
