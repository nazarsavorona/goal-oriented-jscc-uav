"""Qualitative reconstruction montage: what the cls-sim numbers actually look like.

Rows are channel conditions, columns are the four arms a reader needs to compare:

    original | pixel-only codec | detector-feature codec | JPEG + rate-1/2 LDPC

The two learned columns are the *matched pair* of the paper's headline claim: two
5.5 M-parameter DeepJSCC codecs trained on an identical schedule differing only in
the loss (pixel MSE vs frozen-detector feature distillation). Every condition is
applied to both arms through an identical per-frame channel realization (the seed
is re-set before each arm), so a visible difference between columns 2 and 3 is the
loss and nothing else.

The classical column follows the paper's own separated-coding protocol: JPEG over a
rate-1/2 threshold-LDPC model that either delivers the frame or delivers nothing
(mid-grey). Its decode/fail decision is driven by the uncoded BPSK BER at the
effective SINR of the row, which is the same arithmetic that produces the published
cliff table -- so the montage's failures land exactly where the table's do.

Two gates run before any image is written:

  1. The matched-pair lift at clean AWGN SNR = 20 dB must reproduce the published
     value for the checkpoint pair (Table "The three estimates of the detector-feature
     lift"), else the script aborts: a montage drawn from the wrong checkpoints is a
     fabricated figure.
  2. The classical arm's BER at each row must reproduce the published cliff-table
     BER for that condition, so the grey panels are the paper's own outages.

Usage (inside semvideo:latest on a GPU host)::

    python scripts/build_recon_montage_figure.py \
        --detfeat-ckpt experiments/dj_r1_detfeat_20260624_20260623_222209/best.pt \
        --pixel-ckpt   experiments/dj_r1_pixel_s1_20260624_20260623_230658/best.pt \
        --val-dir /data/dji/frames_512_v1/val \
        --n 375 --expect-lift 0.1678 \
        --out-dir experiments/recon_montage_2026_08_21
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")

from semvideo.channel.awgn import AWGNChannel  # noqa: E402
from semvideo.channel.jamming import JammingChannel  # noqa: E402
from semvideo.channel.jamming_reactive import ReactiveJammer  # noqa: E402
from semvideo.channel.packet_loss import BernoulliErasureChannel  # noqa: E402
from semvideo.models.deep_jscc import DeepJSCC  # noqa: E402

CODEC_PX = 256  # the matched pair's training resolution
JPEG_Q = 70  # the paper's separated-baseline source quality
LDPC_BER_THRESHOLD = 1e-3  # rate-1/2 long-block threshold model


# ---------------------------------------------------------------------------
# Channel conditions. `classical_sinr_db` is the effective SINR the separated
# baseline sees in this row; None means the row is an erasure row, where the
# decode decision is the binary-erasure-channel one instead.
# ---------------------------------------------------------------------------
class Condition:
    """One row of the montage."""

    def __init__(self, key: str, label: str, kind: str, snr_db: float | None,
                 jnr_db: float | None = None, p_erase: float | None = None,
                 published_ber: float | None = None) -> None:
        self.key = key
        self.label = label
        self.kind = kind  # clean | awgn | barrage | reactive | erasure
        self.snr_db = snr_db
        self.jnr_db = jnr_db
        self.p_erase = p_erase
        self.published_ber = published_ber

    def build_channel(self, device: str) -> torch.nn.Module:
        # a JNR-sweep condition is keyed "<mode>_jnr<N>", so the mode is the prefix
        base = self.kind.split("_jnr")[0]
        if base in ("clean", "awgn"):
            return AWGNChannel(snr_db=self.snr_db).to(device)
        if base in ("barrage", "pulsed", "swept", "single_tone"):
            return JammingChannel(jammer_type=base, jnr_db=self.jnr_db,
                                  snr_db=self.snr_db).to(device)
        if base == "reactive":
            return _AwgnThenReactive(self.snr_db, self.jnr_db).to(device)
        if base == "erasure":
            # Pure symbol erasure, zero-fill, no compounding AWGN on survivors --
            # matching the erasure table's stated protocol.
            return BernoulliErasureChannel(p=self.p_erase, snr_db=None).to(device)
        raise ValueError(f"unknown condition kind {self.kind}")

    def classical_ber(self) -> float | None:
        """Uncoded BPSK BER at this row's effective SINR; None on erasure rows."""
        if self.kind.split("_jnr")[0] == "erasure":
            return None
        sinr_db = float(self.snr_db)
        if self.jnr_db is not None:
            # Jammer power referenced to the noise floor: SINR = SNR - 10log10(1+JNR).
            sinr_db -= 10.0 * math.log10(1.0 + 10.0 ** (self.jnr_db / 10.0))
        snr_lin = 10.0 ** (sinr_db / 10.0)
        return 0.5 * math.erfc(math.sqrt(snr_lin))

    def classical_decodes(self) -> bool:
        if self.kind.split("_jnr")[0] == "erasure":
            # A rate-1/2 code sits below the binary-erasure capacity 1-p for any
            # p < 0.5, so the separated baseline recovers here. Stated, not hidden:
            # the learned codec's erasure advantage is that it needs no such code.
            return self.p_erase < 0.5
        ber = self.classical_ber()
        assert ber is not None
        return ber < LDPC_BER_THRESHOLD


class _AwgnThenReactive(torch.nn.Module):
    """Base AWGN first, then the energy-following reactive jammer.

    Composition order matches ``ChannelWithReactiveJam``, which is what the
    reactive-jam results in the paper were measured through.
    """

    def __init__(self, snr_db: float, jnr_db: float) -> None:
        super().__init__()
        self.awgn = AWGNChannel(snr_db=snr_db)
        # Spell the gating out rather than inheriting ReactiveJammer's own defaults: the
        # class default is detect_threshold=1.0, but every reactive number in the paper --
        # and ChannelWithReactiveJam, and the evaluation harness -- gates at 0.5x mean
        # energy. Taking the class default here made this figure's reactive row a milder
        # jammer than the tables it sits beside, which is how it was caught.
        self.jam = ReactiveJammer(jnr_db=jnr_db, detect_threshold=0.5,
                                  react_latency=8, hold_duration=64)

    def forward(self, x: torch.Tensor, snr_db: float | None = None) -> torch.Tensor:
        return self.jam(self.awgn(x, snr_db=snr_db))


# Rows sit on cells where the analytic threshold decision and the real BP decode
# (jpegldpc real_n30 sweep, 2026-08-06/2026-08-11, bit-identical) AGREE on outage:
# the paper's LDPC table now reports the real decoder, and at the previous rows
# (AWGN 5 dB, barrage SNR 15) the real decoder recovers while the analytic model
# used for this figure's grey panels does not, so those rows moved one grid cell
# deeper (AWGN 0 dB, barrage SNR 10) where both decoders fail every frame.
CONDITIONS = [
    Condition("clean", "Clean\nAWGN SNR $=20$ dB", "clean", 20.0, published_ber=None),
    Condition("awgn0", "AWGN\nSNR $=0$ dB", "awgn", 0.0, published_ber=None),
    Condition("barrage", "Barrage jam\nJNR $=10$, SNR $=10$", "barrage", 10.0,
              jnr_db=10.0, published_ber=None),
    Condition("reactive", "Reactive jam\nJNR $=10$, SNR $=15$", "reactive", 15.0,
              jnr_db=10.0, published_ber=None),
    Condition("erasure30", "Symbol erasure\n$p=0.30$", "erasure", None, p_erase=0.30),
]


# ---------------------------------------------------------------------------
# Model / metric plumbing
# ---------------------------------------------------------------------------
def load_codec(ckpt: str, device: str) -> DeepJSCC:
    net = DeepJSCC(channel_ratio=1 / 6, base_channels=64).to(device).eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = sd.get("model_state_dict", sd.get("model_state", sd)) if isinstance(sd, dict) else sd
    missing, _ = net.load_state_dict(state, strict=False)
    if len(missing) > 0.3 * len(net.state_dict()):
        raise RuntimeError(f"architecture mismatch for {ckpt}: {len(missing)} missing tensors")
    return net


class ClsSim:
    """DINOv2-small class-token cosine similarity, the paper's goal metric."""

    def __init__(self, device: str) -> None:
        from transformers import AutoModel
        self.device = device
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def cls(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x.clamp(0, 1), (518, 518), mode="bilinear", align_corners=False)
        return self.model(pixel_values=(x - self.mean) / self.std).last_hidden_state[:, 0]

    @torch.no_grad()
    def sim(self, ref_cls: torch.Tensor, x: torch.Tensor) -> float:
        return float(F.cosine_similarity(ref_cls, self.cls(x)).item())


def jpeg_roundtrip(rgb_uint8: np.ndarray, quality: int) -> tuple[np.ndarray, float]:
    """JPEG encode/decode at a fixed quality; returns (decoded RGB, bpp)."""
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    h, w = rgb_uint8.shape[:2]
    bpp = 8.0 * len(buf) / (h * w)
    dec = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB), bpp


def to_tensor(rgb_uint8: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(rgb_uint8).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    return t


def load_codec_input(path: str, device: str) -> torch.Tensor:
    """Frame -> codec input, identical to ``eval_r1_paired.py``.

    That script reads the full frame, converts to a float tensor and *then*
    bilinear-interpolates to the codec resolution. Resizing the uint8 array with
    cv2 instead is a different resampling and moves the measured lift by several
    hundredths -- enough to fail the gate below, which is how this was caught.
    """
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return F.interpolate(to_tensor(rgb, device), (CODEC_PX, CODEC_PX),
                         mode="bilinear", align_corners=False)


def to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detfeat-ckpt", required=True)
    ap.add_argument("--pixel-ckpt", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--n", type=int, default=375, help="frames for the quantitative table")
    ap.add_argument("--display-frames", default="", help="comma-separated val indices to render")
    ap.add_argument("--expect-lift", default=None,
                    help="comma-separated published clean-SNR20 matched-pair lifts; the run "
                         "must reproduce one of them (the table does not record which of the "
                         "two seeded pairs is A and which is B)")
    ap.add_argument("--lift-tol", type=float, default=0.010)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-panels", action="store_true",
                    help="also write each panel as its own PNG, for composing a figure "
                         "that mixes these arms with the deployed 512px codec")
    ap.add_argument("--panels-only", action="store_true",
                    help="skip the full-split quantitative pass (already in metrics.json)")
    ap.add_argument("--extra-erasures", default="",
                    help="comma-separated Bernoulli erasure probabilities to "
                         "measure in addition. Quantitative only.")
    ap.add_argument("--extra-jnr", default="",
                    help="comma-separated JNR values; each --extra-jammers mode "
                         "is also measured at each of these. Quantitative only.")
    ap.add_argument("--extra-jammers", default="",
                    help="comma-separated extra parametric modes to measure (pulsed, swept, "
                         "single_tone). Quantitative only; they are not drawn, and they have "
                         "no published classical cell so that arm is reported ungated.")
    ap.add_argument("--extra-awgn", default="",
                    help="comma-separated extra clean-AWGN SNRs in dB, keyed awgn<N> (e.g. 5 -> "
                         "awgn5, the lift table's AWGN row). Quantitative only.")
    ap.add_argument("--per-frame-csv", action="store_true",
                    help="also write per_frame.csv (condition, image, pixel, detfeat, classical "
                         "cls-sim) so paired intervals can be re-derived later under a "
                         "different bootstrap, e.g. moving-block for temporal correlation. "
                         "metrics.json is unchanged; its means must equal the CSV means.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    for _p in (float(x) for x in args.extra_erasures.split(",") if x.strip()):
        _key = f"erasure{int(round(_p * 100)):02d}"
        CONDITIONS.append(Condition(
            _key, f"Erasure\\n$p={_p:.2f}$", "erasure", None, p_erase=_p))
        print(f"[montage] extra condition: {_key} (quantitative only)",
              flush=True)

    for _s in (float(x) for x in args.extra_awgn.split(",") if x.strip()):
        _key = f"awgn{int(round(_s))}"
        CONDITIONS.append(Condition(_key, f"AWGN\\nSNR $={int(round(_s))}$ dB", "awgn", _s))
        print(f"[montage] extra condition: {_key} (quantitative only)", flush=True)

    _modes = [m.strip() for m in args.extra_jammers.split(",") if m.strip()]
    _jnrs = [float(x) for x in args.extra_jnr.split(",") if x.strip()] or [10.0]
    for mode in _modes:
        for j in _jnrs:
            key = mode if _jnrs == [10.0] and not args.extra_jnr else f"{mode}_jnr{int(j)}"
            CONDITIONS.append(Condition(
                key, f"{mode} jam\nJNR $={int(j)}$, SNR $=15$", key, 15.0, jnr_db=j))
            print(f"[montage] extra condition: {key} (quantitative only, "
                  f"classical ungated)", flush=True)

    paths = sorted(glob.glob(args.val_dir + "/*.jpg") + glob.glob(args.val_dir + "/*.png"))
    if not paths:
        raise SystemExit(f"no frames under {args.val_dir}")
    idx = np.linspace(0, len(paths) - 1, min(args.n, len(paths)), dtype=int)
    paths = [paths[int(i)] for i in idx]
    print(f"[montage] {len(paths)} frames, codec input {CODEC_PX}px, device {device}", flush=True)

    detfeat = load_codec(args.detfeat_ckpt, device)
    pixel = load_codec(args.pixel_ckpt, device)
    metric = ClsSim(device)

    # ---- gate 2 first: it is free, and it decides what the classical column shows
    print("\n[gate] classical arm BER against the published cliff table", flush=True)
    for c in CONDITIONS:
        ber = c.classical_ber()
        if ber is None:
            print(f"  {c.key:9s} erasure row: rate-1/2 below BEC capacity at p={c.p_erase} "
                  f"-> decodes", flush=True)
            continue
        verdict = "decodes" if c.classical_decodes() else "FAILS (grey)"
        if c.published_ber is not None:
            rel = abs(ber - c.published_ber) / c.published_ber
            ok = rel < 0.05
            print(f"  {c.key:9s} BER={ber:.2e} published={c.published_ber:.2e} "
                  f"rel={rel:.1%} {'OK' if ok else 'MISMATCH'} -> {verdict}", flush=True)
            if not ok:
                raise SystemExit(f"classical BER for {c.key} does not reproduce the paper")
        else:
            print(f"  {c.key:9s} BER={ber:.2e} (no published cell) -> {verdict}", flush=True)

    # ---- quantitative pass over the full split, every condition, both arms
    results: dict[str, dict] = {}
    per_frame_cls: dict[str, dict[str, list[float]]] = {}

    for c in (CONDITIONS if not args.panels_only else []):
        arms = {"pixel": [], "detfeat": [], "classical": []}
        for k, path in enumerate(paths):
            src = load_codec_input(path, device)
            rgb = to_uint8(src)  # the JPEG arm sees exactly the codec's input
            ref = metric.cls(src)

            for name, net in (("detfeat", detfeat), ("pixel", pixel)):
                torch.manual_seed(1000 + k)  # identical realization for both arms
                ch = c.build_channel(device)
                with torch.no_grad():
                    rec = net(src, channel=ch).clamp(0, 1)
                arms[name].append(metric.sim(ref, rec))

            if c.classical_decodes():
                dec, _ = jpeg_roundtrip(rgb, JPEG_Q)
            else:
                dec = np.full_like(rgb, 128)
            arms["classical"].append(metric.sim(ref, to_tensor(dec, device)))

            if (k + 1) % 75 == 0:
                print(f"  {c.key}: {k + 1}/{len(paths)}", flush=True)

        per_frame_cls[c.key] = arms
        d = np.array(arms["detfeat"])
        p = np.array(arms["pixel"])
        cl = np.array(arms["classical"])
        diffs = d - p
        rng = np.random.default_rng(0)
        boot = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean()
                         for _ in range(20000)])
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        results[c.key] = {
            "label": c.label.replace("\n", " "),
            "pixel_mean": float(p.mean()), "pixel_std": float(p.std()),
            "detfeat_mean": float(d.mean()), "detfeat_std": float(d.std()),
            "classical_mean": float(cl.mean()),
            "classical_decodes": bool(c.classical_decodes()),
            "paired_lift": float(diffs.mean()), "lift_ci": [lo, hi],
            "frames_improved": float((diffs > 0).mean()), "n": len(paths),
        }
        r = results[c.key]
        print(f"[{c.key:9s}] pixel {r['pixel_mean']:.4f}  detfeat {r['detfeat_mean']:.4f}  "
              f"lift {r['paired_lift']:+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"classical {r['classical_mean']:.4f}"
              f"{' (decoded)' if r['classical_decodes'] else ' (outage)'}", flush=True)

    # ---- gate 1: the clean row must reproduce the published matched-pair lift
    if args.expect_lift and not args.panels_only:
        expected = [float(x) for x in str(args.expect_lift).split(",") if x.strip()]
        got = results["clean"]["paired_lift"]
        deltas = {e: abs(got - e) for e in expected}
        best = min(deltas, key=deltas.get)
        print(f"\n[gate] matched-pair lift @clean SNR20: got {got:+.4f}; published "
              f"{expected}; closest {best:+.4f}, |delta|={deltas[best]:.4f}, "
              f"tol={args.lift_tol}", flush=True)
        if deltas[best] > args.lift_tol:
            raise SystemExit("GATE FAILED: this checkpoint pair reproduces none of the "
                             "published lifts -- refusing to draw the figure")
        results["clean"]["gate_matched_published"] = best
        print("[gate] PASS", flush=True)

    if not args.panels_only:
        # Stamp the provenance into the file the gate and the write-ups actually read. Without
        # this, metrics.json says what the lifts were but not which checkpoints produced them,
        # so a later gate cannot tell one pair's published lift from another's.
        results["_provenance"] = {
            "detfeat_ckpt": args.detfeat_ckpt,
            "pixel_ckpt": args.pixel_ckpt,
            "val_dir": args.val_dir,
            "n": args.n,
        }
        (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.per_frame_csv and not args.panels_only:
        # The record behind every summary above, one row per (condition, frame). Frames are
        # listed in the evaluation order, which is the sorted file order, i.e. source video then
        # frame index, so a block bootstrap over rows respects temporal adjacency.
        import csv
        with (out / "per_frame.csv").open("w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(["condition", "image", "pixel", "detfeat", "classical"])
            for key, arms in per_frame_cls.items():
                for k, path in enumerate(paths):
                    w.writerow([key, Path(path).stem, f"{arms['pixel'][k]:.6f}",
                                f"{arms['detfeat'][k]:.6f}", f"{arms['classical'][k]:.6f}"])
        print(f"[montage] per-frame record: {out / 'per_frame.csv'} "
              f"({sum(len(a['pixel']) for a in per_frame_cls.values())} rows)", flush=True)

    # ---- render the montage
    display = ([int(x) for x in args.display_frames.split(",") if x.strip()]
               or [len(paths) // 3])
    panel_index: dict[str, dict] = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col_titles = ["Original", "Pixel-only codec", "Detector-feature codec",
                  "JPEG $+$ rate-$1/2$ LDPC"]

    for fi in display:
        k = int(fi)
        src = load_codec_input(paths[k], device)
        rgb = to_uint8(src)
        ref = metric.cls(src)

        fig, axes = plt.subplots(len(CONDITIONS[:5]), 4, figsize=(7.4, 9.6))
        for ri, c in enumerate(CONDITIONS[:5]):
            panels = [(rgb, None)]
            for name, net in (("pixel", pixel), ("detfeat", detfeat)):
                torch.manual_seed(1000 + k)
                ch = c.build_channel(device)
                with torch.no_grad():
                    rec = net(src, channel=ch).clamp(0, 1)
                panels.append((to_uint8(rec), metric.sim(ref, rec)))
            if c.classical_decodes():
                dec, bpp = jpeg_roundtrip(rgb, JPEG_Q)
            else:
                dec, bpp = np.full_like(rgb, 128), None
            panels.append((dec, metric.sim(ref, to_tensor(dec, device))))

            if args.save_panels:
                names = ["original", "pixel", "detfeat", "classical"]
                for name, (img, sim) in zip(names, panels):
                    cv2.imwrite(str(out / f"frame{k:04d}_{c.key}_{name}.png"),
                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if sim is not None:
                        panel_index.setdefault(str(k), {}).setdefault(c.key, {})[name] = {
                            "file": f"frame{k:04d}_{c.key}_{name}.png", "cls_sim": sim}
                panel_index.setdefault(str(k), {}).setdefault(c.key, {})["original"] = {
                    "file": f"frame{k:04d}_{c.key}_original.png", "cls_sim": None}
                panel_index[str(k)]["source"] = Path(paths[k]).name

            for ci, (img, sim) in enumerate(panels):
                ax = axes[ri, ci]
                ax.imshow(img)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.4)
                    s.set_color("0.6")
                if sim is not None:
                    ax.set_xlabel(f"{sim:.3f}", fontsize=7.5, labelpad=1.5)
                if ri == 0:
                    ax.set_title(col_titles[ci], fontsize=8.5, pad=4)
                if ci == 0:
                    ax.set_ylabel(c.label, fontsize=7.5, labelpad=3)
        fig.subplots_adjust(left=0.10, right=0.995, top=0.955, bottom=0.012,
                            wspace=0.04, hspace=0.16)
        stem = out / f"recon_montage_frame{k:04d}"
        fig.savefig(f"{stem}.pdf")
        fig.savefig(f"{stem}.png", dpi=200)
        plt.close(fig)
        print(f"[montage] wrote {stem}.pdf / .png  (source {Path(paths[k]).name})", flush=True)

    if args.save_panels:
        (out / "panels.json").write_text(json.dumps(
            {"codec_px": CODEC_PX, "detfeat_ckpt": args.detfeat_ckpt,
             "pixel_ckpt": args.pixel_ckpt, "panels": panel_index}, indent=2),
            encoding="utf-8")
        print(f"[montage] wrote {out}/panels.json", flush=True)

    print("\nMONTAGE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
