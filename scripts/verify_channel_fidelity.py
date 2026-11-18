"""Numerical fidelity checks for the channel-impairment models.

Feeds known inputs through each channel and compares empirical statistics to
the analytical ground truth. This is the 'proof' layer behind the code audit:
distributions and power conventions must match textbook physical-layer models.

Run (CPU is enough):
    python scripts/verify_channel_fidelity.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats

from semvideo.channel.awgn import AWGNChannel
from semvideo.channel.fading import RayleighFadingChannel
from semvideo.channel.packet_loss import BernoulliErasureChannel, BurstErasureChannel

torch.manual_seed(0)
np.random.seed(0)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((PASS if ok else FAIL, name, detail))


# ---------------------------------------------------------------- AWGN
def test_awgn() -> None:
    N = 2_000_000
    # unit-power real signal
    x = torch.randn(1, N)
    x = x / x.pow(2).mean().sqrt()
    for snr_db in (0.0, 10.0, 20.0):
        ch = AWGNChannel(snr_db=snr_db)
        ch.eval()
        y = ch(x, snr_db=snr_db)
        noise = (y - x)
        emp_snr = x.pow(2).mean().item() / noise.pow(2).mean().item()
        target = 10 ** (snr_db / 10)
        rel = abs(emp_snr - target) / target
        check(
            f"AWGN SNR={snr_db:.0f}dB  empirical Px/Pn",
            rel < 0.02,
            f"empirical={emp_snr:.3f}  analytical(10^(SNR/10))={target:.3f}  rel.err={rel*100:.2f}%",
        )
    # noise gaussianity (KS on standardized residual)
    ch = AWGNChannel(snr_db=10.0)
    y = ch(x, snr_db=10.0)
    n = (y - x).flatten().numpy()
    n = (n - n.mean()) / n.std()
    ks = stats.kstest(n[:200000], "norm")
    check("AWGN noise ~ Gaussian (KS)", ks.pvalue > 0.01,
          f"KS stat={ks.statistic:.4f} p={ks.pvalue:.3f}")
    # complex-vs-real convention note: per-real-dim noise var should equal P/10
    ch = AWGNChannel(snr_db=10.0)
    nv = (ch(x, snr_db=10.0) - x).pow(2).mean().item()
    check("AWGN per-dim noise var == P/10 (matches complex Es/N0)",
          abs(nv - 0.1) / 0.1 < 0.02,
          f"per-dim noise var={nv:.4f}  expected P/10=0.1 (P=1)")


# ---------------------------------------------------------------- Rayleigh
def test_rayleigh() -> None:
    N = 2_000_000
    torch.manual_seed(1)
    x = torch.ones(1, N)  # constant signal isolates |h|
    ch = RayleighFadingChannel(snr_db=100.0, equalize=False)  # negligible noise
    ch.eval()
    with torch.no_grad():
        h_mag = ch(x).flatten().numpy()  # y = |h| * 1 + negligible noise
    # E[|h|^2] == 1 (unit-power fading)
    check("Rayleigh E[|h|^2] == 1", abs((h_mag ** 2).mean() - 1) < 0.01,
          f"E[|h|^2]={(h_mag**2).mean():.4f}  expected 1.0")
    # |h| ~ Rayleigh(sigma=1/sqrt2), fully specified null
    cvm = stats.cramervonmises(h_mag[:200000], "rayleigh", args=(0, 1 / np.sqrt(2)))
    check("Rayleigh |h| ~ Rayleigh(1/sqrt2) (Cramer-von Mises, fully specified)",
          cvm.pvalue > 0.01,
          f"W2={cvm.statistic:.4f} p={cvm.pvalue:.4f}")
    # per-symbol independence (lag-1 autocorr ~ 0 -> i.i.d. fast-fading, NOT block fading)
    ac = np.corrcoef(h_mag[:-1], h_mag[1:])[0, 1]
    check("Rayleigh fading is PER-SYMBOL i.i.d. (lag-1 autocorr ~ 0)", abs(ac) < 0.01,
          f"lag-1 autocorr={ac:.4f} (==0 confirms optimistic fast-fading / perfect-interleaving limit, "
          f"NOT realistic block fading)")
    # average post-equalization SNR preserved == snr_linear
    torch.manual_seed(2)
    xs = torch.randn(1, N); xs = xs / xs.pow(2).mean().sqrt()
    ch = RayleighFadingChannel(snr_db=10.0, equalize=True)
    y = ch(xs, snr_db=10.0)
    err = (y - xs)
    eff_snr = xs.pow(2).mean().item() / err.pow(2).mean().item()
    # post-ZF noise is amplified by 1/|h|^2 (E[1/|h|^2] diverges) -> effective SNR < nominal; report it
    check("Rayleigh post-ZF effective SNR < nominal (deep-fade amplification)",
          eff_snr < 10 ** (10 / 10),
          f"effective Px/Perr={eff_snr:.3f}  nominal=10.0 (ZF noise enhancement expected)")


# ---------------------------------------------------------------- Erasure
def test_bernoulli_erasure() -> None:
    N = 4_000_000
    x = torch.randn(1, N)
    for p in (0.3, 0.4):
        ch = BernoulliErasureChannel(p=p)
        y = ch(x, p=p)
        emp = (y == 0).float().mean().item()
        check(f"Bernoulli erasure p={p}: empirical loss rate",
              abs(emp - p) < 0.005,
              f"empirical={emp:.4f}  target p={p}")
    # granularity: erasures are per-symbol independent (lag-1 autocorr of mask ~ 0)
    ch = BernoulliErasureChannel(p=0.3)
    mask = (ch(x, p=0.3) != 0).float().flatten().numpy()
    ac = np.corrcoef(mask[:-1], mask[1:])[0, 1]
    check("Bernoulli erasure is PER-SYMBOL (lag-1 autocorr ~ 0; symbol puncturing, not packet loss)",
          abs(ac) < 0.005, f"lag-1 autocorr={ac:.4f}")


def test_burst_erasure() -> None:
    N = 200_000
    x = torch.randn(1, N)
    ch = BurstErasureChannel(g_to_b=0.01, b_to_g=0.10, p_good=0.0, p_bad=1.0)
    y = ch(x)
    mask_lost = (y == 0).float().flatten().numpy()
    emp_loss = mask_lost.mean()
    stat_loss = ch.stationary_loss_rate()
    check("Burst (Gilbert-Elliott) empirical loss == stationary_loss_rate()",
          abs(emp_loss - stat_loss) < 0.02,
          f"empirical={emp_loss:.4f}  analytical stationary={stat_loss:.4f}")
    # burstiness: lag-1 autocorr of loss mask should be POSITIVE and large
    ac = np.corrcoef(mask_lost[:-1], mask_lost[1:])[0, 1]
    check("Burst erasure is BURSTY (lag-1 autocorr >> 0)", ac > 0.3,
          f"lag-1 autocorr={ac:.4f} (positive => packet-like bursts)")
    # mean burst length ~ 1/b_to_g = 10
    runs = []
    cur = 0
    for v in mask_lost:
        if v > 0.5:
            cur += 1
        elif cur > 0:
            runs.append(cur); cur = 0
    if cur > 0:
        runs.append(cur)
    mbl = float(np.mean(runs)) if runs else 0.0
    check("Burst mean length ~ 1/b_to_g (=10)", abs(mbl - 10) < 3,
          f"empirical mean burst len={mbl:.2f}  expected ~10")


if __name__ == "__main__":
    print("=" * 78)
    print("CHANNEL FIDELITY: empirical vs analytical ground truth")
    print("=" * 78)
    test_awgn()
    test_rayleigh()
    test_bernoulli_erasure()
    test_burst_erasure()
    print()
    n_pass = sum(1 for r in results if r[0] == PASS)
    for status, name, detail in results:
        print(f"[{status}] {name}\n        {detail}")
    print("=" * 78)
    print(f"SUMMARY: {n_pass}/{len(results)} checks passed")
    print("=" * 78)
