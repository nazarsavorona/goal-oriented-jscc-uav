"""RF-level validation, stage 2: carry the codec's symbols over a simulated OFDM link.

Runs in its OWN container (Sionna's torch requirement conflicts with the codec image), reads
the .npy symbol files stage 1 wrote, and puts them through a frequency-domain OFDM link:

    real latent symbols -> 4-bit quantisation -> 16-QAM mapping -> OFDM resource grid
      -> per-resource-element TDL channel with Doppler + AWGN (Sionna OFDMChannel)
      -> jammer added to the RECEIVED resource grid -> LMMSE equalisation with the true channel
      -> QAM demapping -> dequantisation -> real symbols back out

Everything runs on the frequency-domain resource grid: there is no OFDM modulator, IFFT,
cyclic-prefix insertion, pulse shaping or time-domain sample stream (--cp-len only sets the
OFDM-symbol duration at which the channel's time variation is sampled). The jammers are added
to the received grid after the channel and the AWGN, before equalisation; the equaliser and
the demapper are given the thermal-noise variance only. The manuscript's
latent-domain reactive jammer gates on channel-input symbol energy with a latency/hold
automaton; the reactive mode here is a memoryless gate on the power of each received
resource element.
A follower that senses a time-domain signal is NOT simulated by this script, so it cannot say
how such a follower would rank against barrage.

Usage (inside a Sionna container):
    pip install sionna
    python scripts/rf_stage2_sionna_link.py --in-dir DIR --out-dir DIR \\
        --channel tdl --jammer barrage --jnr 10 --snr 20

--channel none --jammer none is the identity round trip that stage 3 uses to prove the
pipeline is lossless before any impairment is believed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def to_complex(sym: np.ndarray) -> np.ndarray:
    """The codec emits real symbols; pair them into complex ones for the modulator."""
    flat = sym.reshape(-1)
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, flat.dtype)])
    return (flat[0::2] + 1j * flat[1::2]).astype(np.complex64)


def to_real(cx: np.ndarray, n: int, shape) -> np.ndarray:
    out = np.empty(cx.size * 2, np.float32)
    out[0::2] = cx.real
    out[1::2] = cx.imag
    return out[:n].reshape(shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--channel", default="tdl", choices=["none", "awgn", "tdl"])
    ap.add_argument("--tdl-model", default="A", choices=["A", "B", "C", "D", "E"],
                    help="3GPP TDL profile; D and E are the LoS ones, closest to air-to-ground.")
    ap.add_argument("--delay-spread", type=float, default=100e-9)
    ap.add_argument("--doppler", type=float, default=200.0,
                    help="Maximum Doppler in Hz; 200 Hz is a fast UAV at S-band.")
    ap.add_argument("--jammer", default="none",
                    choices=["none", "barrage", "reactive", "pulsed"])
    ap.add_argument("--jnr", type=float, default=10.0)
    ap.add_argument("--reactive-threshold", type=float, default=0.5,
                    help="Reactive mode: jam every received resource element whose power exceeds "
                         "this multiple of the frame's mean element power (memoryless; no latency, no hold).")
    ap.add_argument("--duty", type=float, default=0.5,
                    help="Pulsed mode: fraction of the received grid jammed as one contiguous "
                         "run in stored order (consecutive OFDM symbols, all subcarriers), "
                         "with the per-element variance divided by the duty cycle.")
    ap.add_argument("--reactive-power-reference", default="followed",
                    choices=["followed", "matched_barrage"],
                    help="followed: per-element variance equal to barrage, so the "
                         "follower spends less total energy (the original convention). "
                         "matched_barrage: scale the gated injection so total injected "
                         "energy equals barrage at the same JNR.")
    ap.add_argument("--fft-size", type=int, default=512)
    ap.add_argument("--cp-len", type=int, default=64)
    ap.add_argument("--bits-per-symbol", type=int, default=4, help="16-QAM by default.")
    ap.add_argument("--subcarrier-spacing", type=float, default=30e3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Sionna 2.x is torch-based; there is no TensorFlow in this stack.
    import sionna  # noqa: F401
    import sionna.phy  # noqa: F401
    import torch
    from sionna.phy.channel import OFDMChannel
    from sionna.phy.channel.tr38901 import TDL
    from sionna.phy.mapping import Constellation, Demapper, Mapper
    from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    # Sionna keeps its own numpy/python/torch generators; torch.manual_seed does not
    # reach them, so without this the channel and noise draws are not reproducible.
    sionna.phy.config.seed = args.seed

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(in_dir / "meta.json"))

    constellation = Constellation("qam", args.bits_per_symbol)
    mapper = Mapper(constellation=constellation)
    demapper = Demapper("app", constellation=constellation)

    jam_stats: dict[str, float] = {}

    def jam_waveform(x: np.ndarray, noise_var: float) -> np.ndarray:
        """Add interference to the received frequency-domain grid (after channel and AWGN)."""
        if args.jammer == "none":
            return x
        jnr_lin = 10.0 ** (args.jnr / 10.0)
        p_jam = noise_var * jnr_lin
        n = x.size
        j = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
        j *= np.sqrt(p_jam / 2.0)
        if args.jammer == "barrage":
            return x + j
        if args.jammer == "pulsed":
            mask = np.zeros(n, bool)
            on = int(n * args.duty)
            start = rng.integers(0, max(1, n - on))
            mask[start:start + on] = True
            return x + np.where(mask, j / np.sqrt(args.duty), 0)
        # reactive: memoryless gate on the power of each received resource element
        p = np.abs(x) ** 2
        gate = p > args.reactive_threshold * p.mean()
        frac = float(gate.mean())
        jam_stats["gate_fraction"] = frac
        if args.reactive_power_reference == "matched_barrage" and frac > 0:
            # equalise total injected energy against barrage, as the pulsed mode does
            j = j / np.sqrt(frac)
        jam_stats["power_ratio_vs_barrage"] = (
            1.0 if args.reactive_power_reference == "matched_barrage" else frac)
        return x + np.where(gate, j, 0)

    stats = {"cfg": vars(args), "frames": [], "jam": jam_stats}
    for entry in meta["frames"]:
        stem = entry["stem"]
        sym = np.load(in_dir / f"{stem}.npy")
        shape, n_real = sym.shape, sym.size
        cx = to_complex(sym)

        # Bits -> QAM is the wrong direction here: the codec already produced analog symbols,
        # so they are carried as the constellation's soft payload. The honest way to put an
        # analog latent through a digital PHY is to quantise it to bits, send those, and
        # dequantise -- which is what a real digital downlink would do to a learned codec.
        q_levels = 2 ** args.bits_per_symbol
        lo, hi = float(cx.real.min()), float(cx.real.max())
        span = max(hi - lo, 1e-9)

        def quantise(v):
            idx = np.clip(((v - lo) / span * (q_levels - 1)).round(), 0, q_levels - 1)
            return idx.astype(np.int32)

        def dequantise(idx):
            return idx.astype(np.float32) / (q_levels - 1) * span + lo

        idx_i, idx_q = quantise(cx.real), quantise(cx.imag)
        bits = np.concatenate([
            ((idx_i[:, None] >> np.arange(args.bits_per_symbol)[None, ::-1]) & 1),
            ((idx_q[:, None] >> np.arange(args.bits_per_symbol)[None, ::-1]) & 1),
        ]).reshape(-1).astype(np.float32)

        n_sym_needed = int(np.ceil(bits.size / args.bits_per_symbol))
        pad = n_sym_needed * args.bits_per_symbol - bits.size
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, np.float32)])

        rg = ResourceGrid(num_ofdm_symbols=int(np.ceil(n_sym_needed / args.fft_size)),
                          fft_size=args.fft_size, subcarrier_spacing=args.subcarrier_spacing,
                          num_tx=1, num_streams_per_tx=1, cyclic_prefix_length=args.cp_len)
        rg_mapper = ResourceGridMapper(rg)
        cap = rg.num_data_symbols
        bits_t = torch.tensor(
            np.pad(bits, (0, max(0, cap * args.bits_per_symbol - bits.size)))[
                : cap * args.bits_per_symbol], dtype=torch.float32).reshape(1, 1, 1, -1)
        x_map = mapper(bits_t)
        x_rg = rg_mapper(x_map)

        noise_var = 10.0 ** (-args.snr / 10.0)
        if args.channel == "none":
            y = x_rg
            h = None
        else:
            tdl = TDL(model=args.tdl_model, delay_spread=args.delay_spread,
                      carrier_frequency=2.4e9, min_speed=0.0,
                      max_speed=args.doppler * 3e8 / 2.4e9)
            ofdm_channel = OFDMChannel(tdl, rg, add_awgn=True, normalize_channel=True,
                                       return_channel=True)
            y, h = ofdm_channel(x_rg, noise_var)

        y_np = y.detach().cpu().numpy().reshape(-1)
        y_np = jam_waveform(y_np, noise_var)
        y_t = torch.tensor(y_np, dtype=torch.complex64,
                           device=y.device).reshape(y.shape)

        # Equalise. Without this the per-subcarrier channel rotates and scales every QAM
        # point and the demapper returns near-random bits -- the first TDL run scored 0.48
        # with NO jammer, which is the signature of an unequalised receiver rather than a
        # hard channel. LMMSE with the true h matches the manuscript's perfect-CSI
        # assumption, so the RF comparison inherits the same idealisation and no more.
        if h is not None:
            h_eq = h.reshape(y_t.shape).to(y_t.device)
            y_t = (y_t * h_eq.conj()) / (h_eq.abs() ** 2 + noise_var)

        llr = demapper(y_t.reshape(1, 1, 1, -1), noise_var)
        llr_np = llr.detach().cpu().numpy().reshape(-1)
        # Sionna's app demapper returns POSITIVE llr for bit 1 (verified against a
        # map->demap round trip: llr > 0 gives 100% bit accuracy, llr < 0 gives 0%).
        hard = (llr_np > 0).astype(np.int32)[: bits.size - pad if pad else None]
        hard = hard[: idx_i.size * args.bits_per_symbol * 2]
        half = hard.size // 2
        w = args.bits_per_symbol
        rec_i = (hard[:half].reshape(-1, w) << np.arange(w)[::-1]).sum(1)
        rec_q = (hard[half:].reshape(-1, w) << np.arange(w)[::-1]).sum(1)
        k = min(rec_i.size, rec_q.size, cx.size)
        rec = dequantise(rec_i[:k]) + 1j * dequantise(rec_q[:k])
        if k < cx.size:
            rec = np.concatenate([rec, cx[k:]])

        out = to_real(rec, n_real, shape)
        np.save(out_dir / f"{stem}.npy", out.astype(np.float32))
        err = float(np.mean((out - sym) ** 2))
        stats["frames"].append({"stem": stem, "symbol_mse": err})
        print(f"  {stem}: symbol MSE {err:.6f}", flush=True)

    json.dump(stats, open(out_dir / "stage2_meta.json", "w"), indent=1)
    import shutil
    shutil.copy(in_dir / "meta.json", out_dir / "meta.json")
    for entry in meta["frames"]:
        shutil.copy(in_dir / f"{entry['stem']}.src.npy", out_dir / f"{entry['stem']}.src.npy")
    print(f"[saved] {len(stats['frames'])} frames -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
