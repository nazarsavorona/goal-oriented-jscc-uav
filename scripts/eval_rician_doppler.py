"""Rician / Doppler / imperfect-CSI sensitivity of the learned codec.

The paper models fading as per-symbol Rayleigh with perfect-CSI zero forcing, whereas a
practical UAV air-to-ground link has a dominant line of sight, Doppler-
correlated fading and an imperfect channel estimate. This sweeps all three in the latent
domain, which brackets the assumption; it does not replace an RF-level simulation.

K -> 0 recovers the manuscript's Rayleigh limit and larger K is a stronger line of sight;
rho is the lag-1 correlation of the diffuse component (0 = the paper's independent draw);
csi_err is the standard deviation of multiplicative error on the estimate used for equalising.

Usage (inside the semvideo container, with /data and /third_party mounted):
    python scripts/eval_rician_doppler.py --ckpt CKPT --val-dir DIR --out OUT.csv
"""

import sys, csv, glob, json
from pathlib import Path
sys.path.insert(0, "src")
import cv2, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel

# Reuse the proven SwinJSCC loader and the jamming patcher's power-normalised channel hook.
sys.path.insert(0, "scripts")
from eval_swinjscc_dinov2 import build_swinjscc

DEV = "cuda"
import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt", default="/workspace/experiments/p49_p46_awgn_finetune/best.pt")
_ap.add_argument("--val-dir", default="/data/dji/frames_512_v1/val")
_ap.add_argument("--out", default="/workspace/experiments/rev1_fading/rician_doppler.csv")
_ap.add_argument("--n", type=int, default=200)
_args = _ap.parse_args()
CKPT, VAL, OUT = _args.ckpt, _args.val_dir, Path(_args.out)


class RicianDoppler(torch.nn.Module):
    """Rician fading with a Doppler-correlated envelope and optional CSI error.

    K -> inf is the pure-LoS limit and K = 0 is Rayleigh, so sweeping K brackets the
    manuscript's per-symbol Rayleigh assumption against an air-to-ground link that has a
    dominant line of sight. rho is the lag-1 correlation of the diffuse component: rho = 0
    reproduces the paper's independent-per-symbol draw, rho -> 1 is a slowly varying channel
    the equaliser tracks well. csi_err is the standard deviation of the multiplicative error
    on the channel estimate used for zero-forcing.
    """

    def __init__(self, k_db, rho, snr_db, csi_err=0.0):
        super().__init__()
        self.k = 10 ** (k_db / 10.0)
        self.rho = rho
        self.snr_db = snr_db
        self.csi_err = csi_err

    def forward(self, x, snr_db=None):
        snr = self.snr_db if snr_db is None else snr_db
        flat = x.reshape(1, -1)
        n = flat.shape[-1]
        # Diffuse component with an AR(1) envelope (Doppler correlation).
        g = torch.randn(2, n, device=flat.device)
        if self.rho > 0:
            a = np.sqrt(1 - self.rho ** 2)
            for i in range(1, n):
                g[:, i] = self.rho * g[:, i - 1] + a * g[:, i]
        diffuse = (g[0] + 1j * g[1]) / np.sqrt(2.0)
        los = np.sqrt(self.k / (self.k + 1.0))
        h = los + np.sqrt(1.0 / (self.k + 1.0)) * diffuse
        # Zero-forcing with an imperfect estimate of h.
        h_hat = h
        if self.csi_err > 0:
            e = (torch.randn(2, n, device=flat.device) * self.csi_err)
            h_hat = h * (1 + (e[0] + 1j * e[1]) / np.sqrt(2.0))
        p = float((flat ** 2).mean())
        noise_var = p / (10 ** (snr / 10.0))
        noise = torch.randn_like(flat) * np.sqrt(noise_var)
        y = flat * h.abs().to(flat.dtype) + noise
        y = y / h_hat.abs().clamp(min=1e-3).to(flat.dtype)
        return y.reshape(x.shape)


net = build_swinjscc(Path(CKPT), image_dims=(3, 512, 512), C=96)
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)


@torch.no_grad()
def cls(t):
    t = F.interpolate(t.clamp(0, 1), (518, 518), mode="bilinear", align_corners=False)
    return dino(pixel_values=(t - mean) / std).last_hidden_state[:, 0]


paths = sorted(glob.glob(VAL + "/*.jpg"))[:_args.n]
frames, refs = [], []
with torch.no_grad():
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(bgr, (512, 512)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEV) / 255.0
        frames.append(t)
        refs.append(cls(t))
print(f"[setup] {len(frames)} frames", flush=True)

orig_fpc = net.feature_pass_channel
CELLS = []
for k_db in (-100, 0, 6, 10):          # -100 dB ~ Rayleigh, 10 dB ~ strong LoS
    for rho in (0.0, 0.9, 0.99):
        CELLS.append((k_db, rho, 0.0))
for csi in (0.05, 0.15):
    CELLS.append((6, 0.9, csi))

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp)
    w.writerow(["k_db", "rho", "csi_err", "snr", "image", "cls_sim"])
    for k_db, rho, csi in CELLS:
        ch = RicianDoppler(k_db, rho, 20.0, csi).to(DEV)

        def patched(feature, chan_param, avg_pwr=False, _ch=ch):
            # Same unit-power normalisation SwinJSCC's own Channel applies, copied from
            # eval_jamming_cross_corpus.JammingPatcher so the two paths are comparable.
            if avg_pwr:
                tx = feature / torch.sqrt(avg_pwr * 2)
                scale = torch.sqrt(avg_pwr * 2)
            else:
                pwr = torch.mean(feature ** 2) * 2
                tx = feature / torch.sqrt(pwr)
                scale = torch.sqrt(pwr)
            shape = tx.shape
            out = _ch(tx.reshape(1, -1), snr_db=float(chan_param)).reshape(shape)
            return out * scale

        net.feature_pass_channel = patched
        sims = []
        with torch.no_grad():
            for t, ref, p in zip(frames, refs, paths):
                rec, *_ = net(t, given_SNR=20.0, given_rate=None)
                c = F.cosine_similarity(ref, cls(rec.clamp(0, 1))).item()
                sims.append(c)
                w.writerow([k_db, rho, csi, 20.0, Path(p).stem, round(c, 5)])
        net.feature_pass_channel = orig_fpc
        a = np.array(sims)
        print(f"  K={k_db:>5} dB  rho={rho:<5} csi={csi:<5} cls={a.mean():.4f}+-{a.std():.4f} "
              f"n={len(a)}", flush=True)
