"""Paired detector-feature-vs-pixel-only cls-sim lift with a per-frame bootstrap CI.

Loads BOTH 5.5M DeepJSCC checkpoints (detector-feature = perceptual_yolo, pixel-only = mse),
runs each DJI val frame through BOTH with an identical per-frame AWGN noise realization
(true paired design), records the per-frame cls-sim difference, and bootstraps the mean
lift -> 95% CI.

Run inside semvideo:latest:
    python scripts/eval_r1_paired.py <detfeat_ckpt> <pixel_ckpt> <val_dir> [n] [snr]
"""
import sys
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from semvideo.models.deep_jscc import DeepJSCC  # noqa: E402
from semvideo.channel.awgn import AWGNChannel  # noqa: E402
from transformers import AutoModel  # noqa: E402

detf_ckpt, pix_ckpt, valdir = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 375
SNR = int(sys.argv[5]) if len(sys.argv) > 5 else 20


def load(ckpt):
    net = DeepJSCC(channel_ratio=1 / 6, base_channels=64).cuda().eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = sd.get("model_state_dict", sd.get("model_state", sd)) if isinstance(sd, dict) else sd
    miss, unexp = net.load_state_dict(state, strict=False)
    assert len(miss) < 0.3 * len(net.state_dict()), f"arch mismatch {ckpt}: {len(miss)} missing"
    return net


detf, pix = load(detf_ckpt), load(pix_ckpt)
dino = AutoModel.from_pretrained("facebook/dinov2-small").cuda().eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()


def cls(x):
    x = F.interpolate(x.clamp(0, 1), (518, 518), mode="bilinear", align_corners=False)
    return dino(pixel_values=(x - mean) / std).last_hidden_state[:, 0]


paths = sorted(glob.glob(valdir + "/*.jpg") + glob.glob(valdir + "/*.png"))
idx = np.linspace(0, len(paths) - 1, min(n, len(paths)), dtype=int)
paths = [paths[int(i)] for i in idx]
print(f"[r1] detfeat={detf_ckpt.split('/')[-1]} pixel={pix_ckpt.split('/')[-1]} "
      f"n={len(paths)} SNR={SNR}", flush=True)

cd, cp, diffs = [], [], []
with torch.no_grad():
    for k, path in enumerate(paths):
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        img = F.interpolate(img, (256, 256), mode="bilinear", align_corners=False)
        ref = cls(img)
        torch.manual_seed(1000 + k)  # identical channel noise for both arms (paired)
        rec_d = detf(img, channel=AWGNChannel(snr_db=SNR).cuda(), snr_db=SNR).clamp(0, 1)
        torch.manual_seed(1000 + k)
        rec_p = pix(img, channel=AWGNChannel(snr_db=SNR).cuda(), snr_db=SNR).clamp(0, 1)
        a = F.cosine_similarity(ref, cls(rec_d)).item()
        b = F.cosine_similarity(ref, cls(rec_p)).item()
        cd.append(a); cp.append(b); diffs.append(a - b)

cd, cp, diffs = np.array(cd), np.array(cp), np.array(diffs)
print(f"detector-feature cls-sim = {cd.mean():.4f} +- {cd.std():.4f}", flush=True)
print(f"pixel-only       cls-sim = {cp.mean():.4f} +- {cp.std():.4f}", flush=True)
print(f"PAIRED LIFT (detfeat - pixel) = {diffs.mean():+.4f}", flush=True)

rng = np.random.default_rng(0)
B = 20000
boot = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(B)])
lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
print(f"  paired bootstrap 95% CI = [{lo:+.4f}, {hi:+.4f}]  excludes_0={lo > 0 or hi < 0}", flush=True)
# paired t-style: fraction of frames where detfeat wins
print(f"  frames detfeat>pixel = {(diffs > 0).mean():.1%}  (n={len(diffs)})", flush=True)
print("R1_PAIRED_DONE")
