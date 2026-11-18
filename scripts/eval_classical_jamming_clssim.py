"""Separated JPEG+LDPC baseline vs AWGN SNR under every jammer mode -- cls-sim + PSNR.

WHY THIS EXISTS
---------------
The paper evaluates the codec against a separated JPEG+LDPC baseline under AWGN
and five jamming modes and states that JPEG+LDPC cliffs out. This script produces
the per-mode table behind that statement, with a per-cell n.

It writes NO new codec and NO new LDPC. Everything is reused:

  * JPEG source codec + mid-grey "cliff" frame + bpp accounting
        <- scripts/run_classical_clssim_sweep.py
  * real LDPC matrices / encode / BP-decode (pyldpc)
        <- src/semvideo/baselines/ldpc_real.py  (_get_ldpc, bytes_to_bits, bits_to_bytes)
  * the five jammer models, byte-identical to the ones the JSCC arm is scored under
        <- src/semvideo/channel/jamming.py           (barrage / pulsed / swept / single_tone)
        <- src/semvideo/channel/jamming_reactive.py  (reactive DRFM proxy)
  * the DINOv2 ViT-S/14 cls-token cosine + PSNR of the jamming table
        <- scripts/eval_benchmark_sweep.py           (DINOv2Extractor, cossim, psnr)
  * the analytic BER-threshold LDPC model
        <- scripts/baseline_under_jamming.py         (average_ber_under_jamming)

TWO LDPC ARMS
-------------
--ldpc-arm real
    JPEG bytes -> bits -> pyldpc LDPC encode -> BPSK +-1 -> the SAME JammingChannel
    the codec is scored under -> belief-propagation decode -> bytes -> JPEG decode
    (mid-grey cliff frame if the JPEG decoder rejects the corrupted bitstream).
    The BP decoder is handed the *realized* effective SINR, i.e. a jam-aware
    receiver that knows the total interference power. That is deliberately
    generous to the classical scheme.

--ldpc-arm threshold
    The analytic model: a hard cliff at a fixed post-channel BER
    threshold. Kept so the real arm can be compared against it.

A NOTE ON THE REACTIVE MODE, WHICH IS THE INTERESTING ONE
---------------------------------------------------------
The reactive jammer follows *per-symbol emission energy*. A BPSK codeword is
constant-modulus: every symbol carries identical energy. Its energy detector
therefore has nothing to discriminate on, so the automaton latches on and the
mode degenerates to always-on barrage. That is a structural property of pointing
an energy-following jammer at a digital constant-modulus baseline, and it is
reported rather than hidden. `--reactive-add-awgn` controls whether background
AWGN is added on top for that mode (the JSCC arm's convention omits it, which
would leave the separated pipeline with no SNR axis at all).

Usage (inside the semvideo Docker image; pyldpc is pip-installed at entry):
    python eval_classical_jamming_clssim.py \
        --frames-dir /data/dji/frames_512_v1/val --label dji \
        --n-frames 100 --quality 70 \
        --snr-grid 0 5 10 15 20 25 --jnr 10 \
        --jammers none barrage pulsed swept single_tone reactive \
        --ldpc-arm real --out experiments/jpegldpc_jamming
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

SEMVIDEO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEMVIDEO_ROOT / "scripts"))
sys.path.insert(0, str(SEMVIDEO_ROOT / "src"))

# --- reused: JPEG codec + cliff frame + bpp (the classical-baseline harness) ---
from run_classical_clssim_sweep import (  # noqa: E402
    bpsk_ber,
    cliff_frame,
    jpeg_bpp,
    jpeg_decode,
    jpeg_encode,
)

# --- reused: the exact metric path of the paper's jamming table ---
from eval_benchmark_sweep import DINOv2Extractor, cossim, psnr  # noqa: E402

# --- reused: real LDPC stack ---
from semvideo.baselines.ldpc_real import _get_ldpc, bits_to_bytes, bytes_to_bits  # noqa: E402

# --- reused: the five jammers the JSCC arm is scored under ---
from semvideo.channel.jamming import JammingChannel  # noqa: E402
from semvideo.channel.jamming_reactive import ReactiveJammer  # noqa: E402

# --- reused: the analytic threshold model the paper currently relies on ---
from baseline_under_jamming import average_ber_under_jamming  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Threshold-LDPC operating points (label, code_rate, BER threshold) -- as in
# run_classical_clssim_sweep.LDPC_OPERATING_POINTS.
BER_THRESHOLDS = {"1/2": 1e-3, "3/4": 1e-4}


# ---------------------------------------------------------------------------
# Channel application on a real BPSK symbol vector
# ---------------------------------------------------------------------------
def apply_channel(
    x: np.ndarray,
    mode: str,
    snr_db: float,
    jnr_db: float,
    duty_cycle: float,
    reactive_add_awgn: bool,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Push a +-1 BPSK symbol vector through the project's channel models.

    Args:
        x: (N,) float array of +-1 coded symbols.
        mode: none | barrage | pulsed | swept | single_tone | reactive.
        snr_db: background AWGN Es/N0 in dB.
        jnr_db: jammer-to-noise ratio in dB.
        duty_cycle: pulsed duty cycle.
        reactive_add_awgn: add background AWGN under the reactive mode too.
        seed: torch seed for this cell (reproducibility).

    Returns:
        (y, snr_eff_db) where snr_eff_db is the realized signal-to-total-
        interference ratio, handed to the BP decoder as a jam-aware receiver.
    """
    torch.manual_seed(seed)
    xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)  # (1, N)

    if mode == "none":
        sigma = 10.0 ** (-snr_db / 20.0)
        yt = xt + torch.randn_like(xt) * sigma
    elif mode == "reactive":
        jam = ReactiveJammer(
            jnr_db=jnr_db, detect_threshold=0.5, react_latency=8, hold_duration=64,
        )
        yt = jam(xt, snr_db=snr_db)
        if reactive_add_awgn:
            sigma = 10.0 ** (-snr_db / 20.0)
            yt = yt + torch.randn_like(yt) * sigma
    else:
        ch = JammingChannel(
            jammer_type=mode, jnr_db=jnr_db, duty_cycle=duty_cycle, snr_db=snr_db,
        )
        yt = ch(xt, snr_db=snr_db)

    y = yt.squeeze(0).numpy().astype(np.float64)
    resid = y - x
    var_eff = float(np.mean(resid ** 2))
    sig_pow = float(np.mean(x ** 2))
    snr_eff_db = 10.0 * np.log10(max(sig_pow / max(var_eff, 1e-12), 1e-12))
    return y, snr_eff_db


def reactive_duty(x: np.ndarray, jnr_db: float, snr_db: float, seed: int) -> float:
    """Fraction of symbols the reactive automaton actually gates on.

    Diagnostic for the constant-modulus degeneracy noted in the module docstring.
    """
    torch.manual_seed(seed)
    xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
    jam = ReactiveJammer(
        jnr_db=jnr_db, detect_threshold=0.5, react_latency=8, hold_duration=64,
    )
    energy = xt.detach().abs() ** 2
    mean_e = float(energy.mean().clamp(min=1e-12).item())
    mask = jam._build_jam_mask(energy, mean_e)
    return float(mask.float().mean().item())


# ---------------------------------------------------------------------------
# Real LDPC round trip for one frame
# ---------------------------------------------------------------------------
def ldpc_encode_frame(payload: bytes, code_rate: str, n_block: int) -> dict:
    """LDPC-encode a JPEG payload once; reused across every (mode, SNR) cell.

    The generator from ``_get_ldpc`` is systematic with the message in the first
    k codeword positions (verified: ``get_message(G, c)[:k] == c[:k]``), so the
    receiver can slice instead of running pyldpc's per-codeword back-substitution
    -- which otherwise costs ~112 s per frame at this payload size.

    Returns a dict with the +-1 symbol stream and the geometry needed to decode.
    """
    import pyldpc

    H, G, k = _get_ldpc(code_rate, n_block, seed=0)
    n = G.shape[0]
    ref_bits = bytes_to_bits(payload)
    bits = ref_bits
    pad = (-bits.size) % k
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    n_blocks = bits.size // k
    msgs = bits.reshape(n_blocks, k).astype(np.int64)
    d = np.asarray(pyldpc.utils.binaryproduct(G, msgs.T)).astype(np.int64) % 2  # (n, n_blocks)
    x = (1.0 - 2.0 * d.astype(np.float32)).T.reshape(-1)                       # block-major +-1
    return {
        "H": H, "n": n, "k": k, "n_blocks": n_blocks,
        "x": x, "ref_bits": ref_bits, "n_ref": int(ref_bits.size),
    }


def ldpc_channel_decode(
    enc: dict,
    mode: str,
    snr_db: float,
    jnr_db: float,
    duty_cycle: float,
    bp_iter: int,
    reactive_add_awgn: bool,
    seed: int,
) -> tuple[bytes, float, float, float]:
    """Pre-encoded codeword -> jammed channel -> batched BP decode -> bytes.

    Returns:
        (decoded_bytes, ber, snr_eff_db, jam_duty)
    """
    import pyldpc

    x = enc["x"].astype(np.float64)
    y, snr_eff_db = apply_channel(
        x, mode, snr_db, jnr_db, duty_cycle, reactive_add_awgn, seed
    )
    duty = reactive_duty(x, jnr_db, snr_db, seed) if mode == "reactive" else float("nan")

    # pyldpc decodes codewords in parallel when y is (n_code, n_messages).
    y2d = y.reshape(enc["n_blocks"], enc["n"]).T
    try:
        cw = np.asarray(pyldpc.decode(enc["H"], y2d, snr=snr_eff_db, maxiter=bp_iter))
    except Exception as exc:  # noqa: BLE001
        # Hard-decision fallback. Kept for compatibility with the archived runs, but never silent:
        # pyldpc allocates all codewords of a frame at once, so on a small-memory machine large
        # payloads raise MemoryError and the frame would otherwise look like a channel failure.
        print(f"[warn] BP decode failed ({type(exc).__name__}: {str(exc)[:80]}) "
              f"-> hard-decision fallback for this frame", file=sys.stderr, flush=True)
        cw = (y2d < 0).astype(np.int64)
    if cw.ndim == 1:
        cw = cw.reshape(-1, 1)

    out_bits = cw[: enc["k"], :].T.reshape(-1)[: enc["n_ref"]].astype(np.uint8) & 1
    ref = enc["ref_bits"]
    ber = float(np.mean(out_bits != ref)) if ref.size else 0.0
    return bits_to_bytes(out_bits), ber, snr_eff_db, duty


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def load_frames_bgr(frames_dir: Path, n_frames: int, size: int) -> list[tuple[str, np.ndarray]]:
    """Sorted center-cropped square frames at `size`, matching the other harnesses."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in frames_dir.rglob("*") if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"no frames under {frames_dir}")
    if 0 < n_frames < len(paths):
        idx = np.linspace(0, len(paths) - 1, n_frames, dtype=int)
        paths = [paths[int(i)] for i in idx]
    out: list[tuple[str, np.ndarray]] = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        crop = bgr[y0:y0 + side, x0:x0 + side]
        if crop.shape[0] != size:
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        out.append((p.stem, crop))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--quality", type=int, default=70, help="JPEG quality (paper quotes q=70).")
    ap.add_argument("--snr-grid", nargs="+", type=float, default=[0, 5, 10, 15, 20, 25])
    ap.add_argument("--jnr", type=float, default=10.0)
    ap.add_argument("--duty-cycle", type=float, default=0.5)
    ap.add_argument(
        "--jammers", nargs="+",
        default=["none", "barrage", "pulsed", "swept", "single_tone", "reactive"],
    )
    ap.add_argument("--ldpc-arm", choices=["real", "threshold"], default="real")
    ap.add_argument("--code-rate", choices=["1/3", "1/2", "2/3", "3/4"], default="1/2")
    ap.add_argument("--n-block", type=int, default=1024)
    ap.add_argument("--bp-iter", type=int, default=30)
    ap.add_argument("--reactive-add-awgn", action="store_true", default=True)
    ap.add_argument("--no-reactive-add-awgn", dest="reactive_add_awgn", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frames = load_frames_bgr(args.frames_dir, args.n_frames, args.size)
    h, w = frames[0][1].shape[:2]
    print(f"[data] {len(frames)} frames at {h}x{w} from {args.frames_dir}", flush=True)

    print("[load] DINOv2 ViT-S/14 (the jamming table's metric path)...", flush=True)
    dino = DINOv2Extractor()

    # Source-code every frame once at the fixed quality.
    payloads: dict[str, bytes] = {}
    bpps: list[float] = []
    refs: dict[str, torch.Tensor] = {}
    for stem, bgr in frames:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        blob = jpeg_encode(rgb, quality=args.quality)
        payloads[stem] = blob
        bpps.append(jpeg_bpp(blob, h, w))
        refs[stem] = dino.features(bgr)[0]
    mean_bpp = float(np.mean(bpps))
    print(f"[src ] JPEG q={args.quality}: mean bpp={mean_bpp:.4f}, "
          f"mean payload={np.mean([len(b) for b in payloads.values()]):.0f} B", flush=True)

    # LDPC-encode every payload ONCE; the codeword is reused for all (mode, SNR)
    # cells, so only the channel + BP decode is repeated per cell.
    encoded: dict[str, dict] = {}
    if args.ldpc_arm == "real":
        t_enc = time.time()
        for stem in payloads:
            encoded[stem] = ldpc_encode_frame(payloads[stem], args.code_rate, args.n_block)
        any_enc = next(iter(encoded.values()))
        print(f"[ldpc] rate {args.code_rate}, n={any_enc['n']}, k={any_enc['k']}, "
              f"{any_enc['n_blocks']} blocks/frame, encoded {len(encoded)} frames "
              f"in {time.time() - t_enc:.1f}s", flush=True)

    # Channel-perfect (source-coding-only) reference cell.
    clean_cls, clean_psnr = [], []
    for stem, bgr in frames:
        dec = jpeg_decode(payloads[stem])
        rec_bgr = cv2.cvtColor(dec, cv2.COLOR_RGB2BGR) if dec is not None else cliff_frame(bgr.shape)
        clean_cls.append(cossim(refs[stem], dino.features(rec_bgr)[0]))
        clean_psnr.append(psnr(bgr, rec_bgr))
    print(f"[src ] channel-perfect JPEG: cls={np.mean(clean_cls):.4f}+-{np.std(clean_cls):.4f} "
          f"PSNR={np.mean(clean_psnr):.2f}", flush=True)

    # Cliff floor.
    cliff_cls, cliff_psnr = [], []
    for stem, bgr in frames:
        cf = cliff_frame(bgr.shape)
        cliff_cls.append(cossim(refs[stem], dino.features(cf)[0]))
        cliff_psnr.append(psnr(bgr, cf))
    print(f"[cliff] mid-grey floor: cls={np.mean(cliff_cls):.4f}+-{np.std(cliff_cls):.4f} "
          f"PSNR={np.mean(cliff_psnr):.2f}", flush=True)

    pf_path = args.out / "per_frame.csv"
    sm_path = args.out / "summary.csv"
    fp = open(pf_path, "w", newline="")
    pw = csv.writer(fp)
    pw.writerow([
        "label", "ldpc_arm", "code_rate", "jammer", "jnr_db", "snr_db", "image",
        "cls_sim", "psnr", "ber", "snr_eff_db", "decode_ok", "jam_duty",
    ])

    rows: list[dict] = []
    for mode in args.jammers:
        jnr = 0.0 if mode == "none" else args.jnr
        for snr in args.snr_grid:
            t0 = time.time()
            cls_v, psnr_v, ber_v, eff_v, ok_v, duty_v = [], [], [], [], [], []
            for fi, (stem, bgr) in enumerate(frames):
                if args.ldpc_arm == "real":
                    dec_bytes, ber, snr_eff, duty = ldpc_channel_decode(
                        encoded[stem], mode, float(snr), jnr, args.duty_cycle,
                        args.bp_iter, args.reactive_add_awgn, args.seed + 1000 * fi,
                    )
                    dec = jpeg_decode(dec_bytes)
                else:
                    # The paper's analytic model: hard cliff at a BER threshold.
                    if mode == "none":
                        ber = bpsk_ber(float(snr))
                    elif mode == "reactive":
                        # degenerates to always-on barrage on constant-modulus BPSK
                        ber = average_ber_under_jamming(float(snr), jnr, "barrage")
                    else:
                        key = "single_tone" if mode == "single_tone" else mode
                        key = key if key in {"barrage", "pulsed", "single_tone"} else "barrage"
                        ber = average_ber_under_jamming(float(snr), jnr, key)
                    snr_eff, duty = float("nan"), float("nan")
                    dec = jpeg_decode(payloads[stem]) if ber < BER_THRESHOLDS[args.code_rate] else None

                if dec is None:
                    rec_bgr = cliff_frame(bgr.shape)
                    ok = 0
                else:
                    rec_bgr = cv2.cvtColor(dec, cv2.COLOR_RGB2BGR)
                    if rec_bgr.shape != bgr.shape:
                        rec_bgr = cv2.resize(rec_bgr, (w, h))
                    ok = 1
                c = cossim(refs[stem], dino.features(rec_bgr)[0])
                ps = psnr(bgr, rec_bgr)
                cls_v.append(c); psnr_v.append(ps); ber_v.append(ber)
                eff_v.append(snr_eff); ok_v.append(ok); duty_v.append(duty)
                pw.writerow([
                    args.label, args.ldpc_arm, args.code_rate, mode, jnr, snr, stem,
                    round(c, 5), round(ps, 3), f"{ber:.3e}",
                    round(snr_eff, 3) if np.isfinite(snr_eff) else "",
                    ok, round(duty, 4) if np.isfinite(duty) else "",
                ])
            fp.flush()
            row = {
                "label": args.label, "ldpc_arm": args.ldpc_arm, "code_rate": args.code_rate,
                "jammer": mode, "jnr_db": jnr, "snr_db": snr, "n": len(cls_v),
                "cls_mean": round(float(np.mean(cls_v)), 5),
                "cls_std": round(float(np.std(cls_v)), 5),
                "psnr_mean": round(float(np.mean(psnr_v)), 3),
                "psnr_std": round(float(np.std(psnr_v)), 3),
                "ber_mean": f"{float(np.mean(ber_v)):.3e}",
                "snr_eff_mean": round(float(np.nanmean(eff_v)), 3) if np.any(np.isfinite(eff_v)) else "",
                "decode_ok_frac": round(float(np.mean(ok_v)), 4),
                "jam_duty_mean": round(float(np.nanmean(duty_v)), 4) if np.any(np.isfinite(duty_v)) else "",
                "jpeg_q": args.quality, "source_bpp": round(mean_bpp, 5),
                "wall_s": round(time.time() - t0, 1),
            }
            rows.append(row)
            print(f"  [{mode:>12s} JNR={jnr:>4.1f} SNR={snr:>5.1f}] "
                  f"cls={row['cls_mean']:.4f}+-{row['cls_std']:.4f} "
                  f"PSNR={row['psnr_mean']:>6.2f} BER={row['ber_mean']} "
                  f"ok={row['decode_ok_frac']:.2f} ({row['wall_s']}s)", flush=True)
    fp.close()

    with open(sm_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    meta = {
        "frames_dir": str(args.frames_dir), "label": args.label,
        "n_frames": len(frames), "size": args.size, "jpeg_quality": args.quality,
        "source_bpp": mean_bpp, "ldpc_arm": args.ldpc_arm, "code_rate": args.code_rate,
        "n_block": args.n_block, "bp_iter": args.bp_iter, "jnr_db": args.jnr,
        "duty_cycle": args.duty_cycle, "jammers": args.jammers,
        "snr_grid": args.snr_grid, "reactive_add_awgn": bool(args.reactive_add_awgn),
        "metric": "DINOv2 ViT-S/14 cls-token cosine (eval_benchmark_sweep.DINOv2Extractor)",
        "channel_perfect_jpeg": {
            "cls_mean": float(np.mean(clean_cls)), "cls_std": float(np.std(clean_cls)),
            "psnr_mean": float(np.mean(clean_psnr)), "n": len(clean_cls),
        },
        "cliff_floor": {
            "cls_mean": float(np.mean(cliff_cls)), "cls_std": float(np.std(cliff_cls)),
            "psnr_mean": float(np.mean(cliff_psnr)), "n": len(cliff_cls),
        },
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[saved] {pf_path}\n[saved] {sm_path}\n[saved] {args.out / 'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
