"""The deployed-codec panels of the qualitative montage.

The companion script ``build_recon_montage_figure.py`` renders the matched 5.5 M pair,
which is the right arm for the distillation *ablation* but is not the codec this paper
deploys. A montage built only from that pair would tell a reader the system is blurry when
the deployed two-stage codec sits at cls-sim 0.99. So this script renders the same five
channel conditions on the same display frames through the two-stage cross-corpus SwinJSCC
codec at 512 px, and a local compositor merges the two into one figure.

Runs in the SwinJSCC mount layout, because ``eval_swinjscc_dinov2`` chdir's into
third_party/SwinJSCC at import::

    docker run --rm --runtime=nvidia \
      -v $HOME/semvideo:/repo/semvideo -v $HOME/semvideo/third_party:/repo/third_party \
      -v $HOME/data:/data -w /repo/semvideo semvideo:latest \
      python -u scripts/build_recon_montage_swin.py --ckpt /repo/semvideo/experiments/... \
        --out-dir /repo/semvideo/experiments/recon_montage_2026_08_21/swin

Every checkpoint path must be absolute for that reason.

Gate. Four of the five rows have a published cell for this checkpoint (cliff table and
erasure table). All four must reproduce, or the script refuses to write panels: the whole
value of the figure is that it shows the same codec the tables measure.
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
import torch.nn.functional as F

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from eval_erasure_cross_corpus import ErasurePatcher  # noqa: E402
from eval_jamming_cross_corpus import JammingPatcher, center_square_resize  # noqa: E402
from eval_swinjscc_dinov2 import build_swinjscc  # noqa: E402

DEVICE = "cuda"
CODEC_PX = 512

# key, given_SNR, jammer (None = clean), JNR, erasure p, published cls-sim for this
# checkpoint. Published values: cliff table (no-jam SNR 20 / SNR 0, barrage JNR 10 at
# SNR 10) and the erasure table's general-codec column at p = 0.30, nominal SNR 10.
# 2026-08-27: the AWGN and barrage rows moved one grid cell deeper (5 -> 0 dB and
# SNR 15 -> 10) so the separated column's grey outage panels agree with the REAL
# BP decoder the paper's LDPC table now reports, not only with the analytic model.
ROWS = [
    ("clean", 20.0, None, None, None, 0.991),
    ("awgn0", 0.0, None, None, None, 0.934),
    ("barrage", 10.0, "barrage", 10.0, None, 0.861),
    ("reactive", 15.0, "reactive", 10.0, None, None),
    ("erasure30", 10.0, None, None, 0.30, 0.880),
]
GATE_TOL = 0.015


class ClsSim:
    """DINOv2-small class-token cosine similarity, the paper's goal metric."""

    def __init__(self) -> None:
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").to(DEVICE).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    @torch.no_grad()
    def cls(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
        x = F.interpolate(x, (518, 518), mode="bilinear", align_corners=False)
        return self.model(pixel_values=(x - self.mean) / self.std).last_hidden_state[:, 0]

    @torch.no_grad()
    def sim(self, a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
        return float(F.cosine_similarity(self.cls(a_bgr), self.cls(b_bgr)).item())


@torch.no_grad()
def reconstruct(net, jam_patcher, era_patcher, crop_bgr: np.ndarray, snr_db: float,
                jammer: str | None, jnr_db: float | None,
                p_erase: float | None) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    if jammer is not None:
        with jam_patcher(jammer_type=jammer, jnr_db=float(jnr_db)):
            rec, *_ = net(x, given_SNR=snr_db, given_rate=None)
    elif p_erase:
        era_patcher.p = float(p_erase)
        era_patcher.mode = "iid"
        with era_patcher:
            rec, *_ = net(x, given_SNR=snr_db, given_rate=None)
        era_patcher.p = 0.0
    else:
        rec, *_ = net(x, given_SNR=snr_db, given_rate=None)
    rec = rec.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor((rec * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="absolute path to the two-stage codec")
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gate-n", type=int, default=100,
                    help="frames for the published-cell gate")
    ap.add_argument("--display-frames", default="40,150,300")
    ap.add_argument("--skip-gate", action="store_true", help="debugging only")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = sorted(glob.glob(args.val_dir + "/*.jpg") + glob.glob(args.val_dir + "/*.png"))
    if not paths:
        raise SystemExit(f"no frames under {args.val_dir}")
    # Same deterministic 375-frame subsample the montage script uses, so display
    # indices refer to the same frames in both halves of the figure.
    idx = np.linspace(0, len(paths) - 1, min(375, len(paths)), dtype=int)
    paths = [paths[int(i)] for i in idx]

    net = build_swinjscc(Path(args.ckpt), image_dims=(3, CODEC_PX, CODEC_PX), C=96)
    jam_patcher = JammingPatcher(net)
    era_patcher = ErasurePatcher(net)
    metric = ClsSim()
    print(f"[swin] {Path(args.ckpt).name}, {CODEC_PX}px, {len(paths)} val frames", flush=True)

    # ---- gate against the published cells
    gate_paths = [paths[int(i)] for i in
                  np.linspace(0, len(paths) - 1, min(args.gate_n, len(paths)), dtype=int)]
    gate_report = {}
    for key, snr, jammer, jnr, p_er, published in ROWS:
        sims = []
        for k, path in enumerate(gate_paths):
            crop = center_square_resize(cv2.imread(path), CODEC_PX)
            torch.manual_seed(3000 + k)
            rec = reconstruct(net, jam_patcher, era_patcher, crop, snr, jammer, jnr, p_er)
            sims.append(metric.sim(crop, rec))
        got = float(np.mean(sims))
        gate_report[key] = {"measured": got, "published": published, "n": len(gate_paths)}
        if published is None:
            print(f"[gate] {key:9s} measured {got:.4f} (no published cell)", flush=True)
            continue
        delta = abs(got - published)
        ok = delta <= GATE_TOL
        gate_report[key]["ok"] = ok
        print(f"[gate] {key:9s} measured {got:.4f} published {published:.4f} "
              f"|delta|={delta:.4f} tol={GATE_TOL} -> {'OK' if ok else 'MISMATCH'}",
              flush=True)
        if not ok and not args.skip_gate:
            raise SystemExit(f"GATE FAILED on {key}: this checkpoint does not reproduce "
                             "the published cell -- refusing to write panels")
    print("[gate] PASS", flush=True)

    # ---- panels for the display frames
    panels = {}
    for fi in [int(x) for x in args.display_frames.split(",") if x.strip()]:
        crop = center_square_resize(cv2.imread(paths[fi]), CODEC_PX)
        cv2.imwrite(str(out / f"frame{fi:04d}_original.png"), crop)
        panels[str(fi)] = {"source": Path(paths[fi]).name, "rows": {}}
        for key, snr, jammer, jnr, p_er, _ in ROWS:
            torch.manual_seed(3000 + fi)
            rec = reconstruct(net, jam_patcher, era_patcher, crop, snr, jammer, jnr, p_er)
            fn = out / f"frame{fi:04d}_{key}_twostage.png"
            cv2.imwrite(str(fn), rec)
            sim = metric.sim(crop, rec)
            panels[str(fi)]["rows"][key] = {"file": fn.name, "cls_sim": sim}
            print(f"[panel] frame {fi} {key:9s} cls-sim {sim:.3f} -> {fn.name}", flush=True)

    (out / "panels.json").write_text(json.dumps(
        {"checkpoint": args.ckpt, "codec_px": CODEC_PX, "gate": gate_report,
         "panels": panels}, indent=2), encoding="utf-8")
    print("\nSWIN_PANELS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
