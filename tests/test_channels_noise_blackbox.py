"""Black-box contract tests for the NOISE channel models.

Covered public APIs (imported only through their package entry points):

  - ``semvideo.channel.awgn.AWGNChannel``
  - ``semvideo.channel.fading.RayleighFadingChannel``
  - ``semvideo.channel.jamming.JammingChannel`` (4 jammer types)
  - ``semvideo.channel.jamming_reactive.ReactiveJammer``

Discipline
----------
Each channel is treated as a BLACK BOX. We import only the public class,
construct it through its documented constructor, and assert on observable
input -> output contracts and on the mathematical / domain properties that the
docstrings and the project verification rules ("Verification &
Proof-Checking") require:

  (a) output shape and dtype equal the input;
  (b) gradients flow back to the input tensor (the channel is differentiable);
  (c) higher SNR -> strictly lower mean-squared error vs the input
      (monotone, swept over several SNR points);
  (d) the AWGN additive noise empirically matches N(0, sigma^2) for the
      configured SNR, checked with a Kolmogorov-Smirnov test on a large
      sample (the analytical-distribution rule);
  (e) the same ``torch.manual_seed`` reproduces an identical output, a
      different seed produces a different output;
  (f) JammingChannel: higher JNR -> more distortion; the pulsed ``duty_cycle``
      in [0, 1] controls the active fraction statistically; an unknown
      ``jammer_type`` raises ``ValueError``.

No private attributes, no internal call-order assertions, no "golden" tensors
captured from the current implementation. Every assertion is derived from the
spec / domain math, so it would still hold for any correct re-implementation.

All tests run on small CPU tensors with explicit seeds and tolerances, so the
suite stays deterministic and green on a laptop with no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from semvideo.channel.awgn import AWGNChannel
from semvideo.channel.fading import RayleighFadingChannel
from semvideo.channel.jamming import JammingChannel
from semvideo.channel.jamming_reactive import ReactiveJammer

# scipy provides the Kolmogorov-Smirnov test required by the project's
# "compare channel output to analytical distribution" verification rule.
scipy_stats = pytest.importorskip("scipy.stats")


# A small, well-separated SNR ladder. The points are far apart so that the
# monotonicity contract is testable with a modest Monte-Carlo budget despite
# the heavy-tailed Rayleigh equaliser (residual noise scaled by 1 / |h|).
SNR_LADDER = [0.0, 5.0, 10.0, 15.0, 20.0]


def _seed_all(seed: int) -> None:
    """Seed every RNG the channels can draw from."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def _mean_mse_vs_snr(
    channel: torch.nn.Module,
    x: torch.Tensor,
    snr_db: float,
    *,
    n_trials: int,
    base_seed: int,
) -> float:
    """Monte-Carlo mean of ||channel(x) - x||^2 at a fixed SNR.

    Averaging over many independent noise draws turns the (random) channel
    output into a stable estimate of the expected distortion, which is the
    quantity the monotonicity contract is actually about.
    """
    total = 0.0
    for t in range(n_trials):
        _seed_all(base_seed + t)
        y = channel(x, snr_db=snr_db)
        total += torch.mean((y - x) ** 2).item()
    return total / n_trials


# Factory list of the "SNR-driven additive noise" channels. These three share
# the contract that distortion is governed by an SNR parameter, so the
# shape / dtype / gradient / monotonicity / reproducibility contracts can be
# parametrised over all of them.
SNR_CHANNEL_FACTORIES = {
    "awgn": lambda: AWGNChannel(snr_db=10.0),
    "rayleigh": lambda: RayleighFadingChannel(snr_db=10.0, equalize=True),
    "jamming_barrage": lambda: JammingChannel(
        jammer_type="barrage", jnr_db=0.0, snr_db=10.0
    ),
}

# All NOISE channels, including the reactive jammer, for the
# generic shape / dtype / gradient / reproducibility contracts that do not
# depend on an SNR sweep.
ALL_CHANNEL_FACTORIES = {
    **SNR_CHANNEL_FACTORIES,
    "jamming_single_tone": lambda: JammingChannel(
        jammer_type="single_tone", jnr_db=5.0, snr_db=10.0
    ),
    "jamming_swept": lambda: JammingChannel(
        jammer_type="swept", jnr_db=5.0, snr_db=10.0
    ),
    "jamming_pulsed": lambda: JammingChannel(
        jammer_type="pulsed", jnr_db=5.0, duty_cycle=0.5, snr_db=10.0
    ),
    # react_latency=0 with detect_threshold=0 guarantees the automaton engages,
    # so the jammer is exercised (not a silent passthrough) in these contracts.
    "reactive": lambda: ReactiveJammer(
        jnr_db=10.0, detect_threshold=0.0, react_latency=0, hold_duration=256
    ),
}


# --------------------------------------------------------------------------- #
# Contract (a): output shape + dtype == input.
# --------------------------------------------------------------------------- #
class TestShapeAndDtype:
    @pytest.mark.parametrize("name", list(ALL_CHANNEL_FACTORIES))
    def test_shape_and_dtype_preserved(self, name: str) -> None:
        channel = ALL_CHANNEL_FACTORIES[name]().eval()
        _seed_all(0)
        x = torch.randn(4, 256)  # (B, N) channel-symbol convention
        y = channel(x)
        assert y.shape == x.shape, f"{name}: shape changed {x.shape} -> {y.shape}"
        assert y.dtype == x.dtype, f"{name}: dtype changed {x.dtype} -> {y.dtype}"
        assert torch.isfinite(y).all(), f"{name}: produced non-finite values"

    @pytest.mark.parametrize("name", list(ALL_CHANNEL_FACTORIES))
    def test_dtype_preserved_float64(self, name: str) -> None:
        # The contract is "dtype == input", so a float64 input must come back
        # float64. This catches any implicit float32 cast in the noise draw.
        channel = ALL_CHANNEL_FACTORIES[name]().eval()
        _seed_all(1)
        x = torch.randn(4, 128, dtype=torch.float64)
        y = channel(x)
        assert y.dtype == torch.float64, f"{name}: lost float64 -> {y.dtype}"
        assert y.shape == x.shape


# --------------------------------------------------------------------------- #
# Contract (b): gradients flow back to the input (channel is differentiable).
# --------------------------------------------------------------------------- #
class TestDifferentiable:
    @pytest.mark.parametrize("name", list(ALL_CHANNEL_FACTORIES))
    def test_gradient_flows_to_input(self, name: str) -> None:
        channel = ALL_CHANNEL_FACTORIES[name]().train()
        _seed_all(3)
        x = torch.randn(4, 256, requires_grad=True)
        out = channel(x)
        out.sum().backward()
        # The project verification rule: "gradients must flow through
        # channel layers (assert param.grad is not None)".
        assert x.grad is not None, f"{name}: no gradient reached the input"
        assert torch.isfinite(x.grad).all(), f"{name}: non-finite gradient"
        assert (x.grad != 0).any(), f"{name}: gradient is identically zero"


# --------------------------------------------------------------------------- #
# Contract (c): higher SNR -> strictly lower expected MSE (monotone sweep).
# --------------------------------------------------------------------------- #
class TestMonotoneInSNR:
    @pytest.mark.parametrize("name", list(SNR_CHANNEL_FACTORIES))
    def test_mse_strictly_decreases_with_snr(self, name: str) -> None:
        channel = SNR_CHANNEL_FACTORIES[name]().eval()
        _seed_all(10)
        x = torch.randn(8, 1024)
        # Rayleigh's 1/|h| equaliser has heavy tails, so it needs more Monte
        # Carlo trials than the additive-only channels for a stable mean.
        n_trials = 200 if name == "rayleigh" else 60
        mse = [
            _mean_mse_vs_snr(channel, x, snr, n_trials=n_trials, base_seed=1000)
            for snr in SNR_LADDER
        ]
        # PSNR (hence MSE) must be monotone in SNR -- the project flags a
        # violation as a bug.
        for lo, hi, m_lo, m_hi in zip(
            SNR_LADDER[:-1], SNR_LADDER[1:], mse[:-1], mse[1:]
        ):
            assert m_hi < m_lo, (
                f"{name}: MSE not strictly decreasing across SNR "
                f"{lo} -> {hi} dB ({m_lo:.5f} -> {m_hi:.5f}); full sweep={mse}"
            )


# --------------------------------------------------------------------------- #
# Contract (d): AWGN additive noise ~ N(0, sigma^2) via Kolmogorov-Smirnov.
# --------------------------------------------------------------------------- #
class TestAWGNDistribution:
    def test_additive_noise_is_gaussian(self) -> None:
        # sigma^2 = signal_power / 10^(snr/10), signal_power = mean(x^2).
        # The added noise is recoverable as (output - input); under the spec it
        # must be zero-mean Gaussian with that variance.
        snr_db = 10.0
        channel = AWGNChannel(snr_db=snr_db).eval()
        _seed_all(123)
        x = torch.randn(4, 25_000)  # large N for a powerful goodness-of-fit test
        signal_power = torch.mean(x**2).item()
        sigma = (signal_power / 10.0 ** (snr_db / 10.0)) ** 0.5

        noise = (channel(x) - x).flatten().numpy()

        # Standardise by the analytical sigma and compare to a standard normal.
        cvm = scipy_stats.cramervonmises(noise / sigma, "norm")
        assert cvm.pvalue > 0.01, (
            f"AWGN noise fails Cramer-von Mises GOF vs N(0, sigma^2) after analytical "
            f"standardisation: W2={cvm.statistic:.4f}, p={cvm.pvalue:.4g}"
        )
        # Empirical moments should also match the analytical ones.
        assert abs(float(noise.mean())) < 0.02 * sigma, "noise not zero-mean"
        assert abs(float(noise.std()) - sigma) < 0.05 * sigma, (
            f"empirical sigma {noise.std():.4f} != analytical {sigma:.4f}"
        )

    @pytest.mark.parametrize("snr_db", [0.0, 6.0, 12.0, 20.0])
    def test_noise_variance_tracks_snr(self, snr_db: float) -> None:
        # The empirical noise variance must follow signal_power / snr_linear
        # for whatever SNR is requested -- the core AWGN power relation.
        channel = AWGNChannel(snr_db=snr_db).eval()
        _seed_all(int(snr_db) + 50)
        x = torch.randn(4, 20_000)
        signal_power = torch.mean(x**2).item()
        expected_var = signal_power / 10.0 ** (snr_db / 10.0)
        noise = (channel(x) - x)
        emp_var = torch.var(noise, unbiased=False).item()
        assert abs(emp_var - expected_var) < 0.08 * expected_var, (
            f"SNR={snr_db}: empirical noise var {emp_var:.5f} "
            f"!= analytical {expected_var:.5f}"
        )


# --------------------------------------------------------------------------- #
# Contract (e): same seed -> identical output, different seed -> different.
# --------------------------------------------------------------------------- #
class TestReproducibility:
    @pytest.mark.parametrize("name", list(ALL_CHANNEL_FACTORIES))
    def test_same_seed_identical_diff_seed_differs(self, name: str) -> None:
        channel = ALL_CHANNEL_FACTORIES[name]().eval()
        _seed_all(0)
        x = torch.randn(4, 256)

        _seed_all(42)
        out_a = channel(x).clone()
        _seed_all(42)
        out_b = channel(x).clone()
        _seed_all(43)
        out_c = channel(x).clone()

        # Reproducibility rule: identical seed -> bit-identical output.
        assert torch.equal(out_a, out_b), (
            f"{name}: same seed produced different outputs"
        )
        # A different seed must perturb the (stochastic) channel.
        assert not torch.equal(out_a, out_c), (
            f"{name}: different seeds produced identical output (not random)"
        )


# --------------------------------------------------------------------------- #
# Contract (f): JammingChannel JNR, pulsed duty cycle, unknown-type error.
# --------------------------------------------------------------------------- #
class TestJammingChannel:
    @pytest.mark.parametrize(
        "jammer_type", ["barrage", "single_tone", "swept", "pulsed"]
    )
    def test_higher_jnr_more_distortion(self, jammer_type: str) -> None:
        # JNR is the jammer-to-noise ratio; turning it up injects more
        # interference power, so the expected distortion must grow with JNR.
        _seed_all(20)
        x = torch.randn(8, 1024)
        jnr_ladder = [0.0, 5.0, 10.0, 15.0, 20.0]
        mse = []
        for jnr in jnr_ladder:
            # pulsed: full duty so every trial actually carries the jammer;
            # otherwise the active fraction adds variance to the mean.
            kwargs = {"duty_cycle": 1.0} if jammer_type == "pulsed" else {}
            channel = JammingChannel(
                jammer_type=jammer_type, jnr_db=jnr, snr_db=10.0, **kwargs
            ).eval()
            total = 0.0
            n_trials = 80
            for t in range(n_trials):
                _seed_all(2000 + t)
                total += torch.mean((channel(x) - x) ** 2).item()
            mse.append(total / n_trials)
        for lo, hi, m_lo, m_hi in zip(
            jnr_ladder[:-1], jnr_ladder[1:], mse[:-1], mse[1:]
        ):
            assert m_hi > m_lo, (
                f"{jammer_type}: distortion not increasing across JNR "
                f"{lo} -> {hi} dB ({m_lo:.5f} -> {m_hi:.5f}); sweep={mse}"
            )

    @pytest.mark.parametrize("duty_cycle", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_pulsed_duty_cycle_controls_active_fraction(
        self, duty_cycle: float
    ) -> None:
        # The partial-band pulsed jammer masks symbol BLOCKS within each row
        # at constant time-average power: a fraction ~ duty_cycle of symbols
        # is hit, with per-symbol peak power jammer_power / duty_cycle. With
        # JNR far above the AWGN floor, jammed symbols are separable at the
        # per-symbol level (peak >= 1e4x the floor for every tested duty,
        # while an unjammed symbol exceeds 10x the floor with chi-square
        # probability ~0.0016).
        snr_db, jnr_db = 30.0, 40.0
        channel = JammingChannel(
            jammer_type="pulsed",
            jnr_db=jnr_db,
            duty_cycle=duty_cycle,
            snr_db=snr_db,
        ).eval()
        _seed_all(7)
        x = torch.randn(4000, 64)  # many rows for a tight empirical fraction
        signal_power = torch.mean(x**2).item()
        noise_floor = signal_power / 10.0 ** (snr_db / 10.0)
        y = channel(x)
        sym_dev = (y - x) ** 2
        active_fraction = (sym_dev > 10 * noise_floor).float().mean().item()
        assert abs(active_fraction - duty_cycle) < 0.05, (
            f"duty_cycle={duty_cycle}: empirical jammed-symbol fraction "
            f"{active_fraction:.3f} too far from configured value"
        )

    def test_duty_cycle_is_monotone_in_active_fraction(self) -> None:
        # Cross-check: a larger duty_cycle never yields fewer active rows.
        snr_db, jnr_db = 30.0, 40.0
        fractions = []
        for dc in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            channel = JammingChannel(
                jammer_type="pulsed", jnr_db=jnr_db, duty_cycle=dc, snr_db=snr_db
            ).eval()
            _seed_all(99)
            x = torch.randn(4000, 64)
            noise_floor = torch.mean(x**2).item() / 10.0 ** (snr_db / 10.0)
            y = channel(x)
            row_mse = torch.mean((y - x) ** 2, dim=1)
            fractions.append(
                (row_mse > 100 * noise_floor).float().mean().item()
            )
        for f_lo, f_hi in zip(fractions[:-1], fractions[1:]):
            assert f_hi >= f_lo - 1e-9, (
                f"active fraction not monotone in duty_cycle: {fractions}"
            )

    def test_unknown_jammer_type_raises(self) -> None:
        # The docstring enumerates exactly four jammer types; anything else is
        # a programming error and must raise ValueError (not silently pass).
        channel = JammingChannel(jammer_type="definitely_not_a_jammer")
        _seed_all(0)
        x = torch.randn(2, 64)
        with pytest.raises(ValueError):
            channel(x)


# --------------------------------------------------------------------------- #
# Reactive jammer: domain-specific JNR -> distortion contract.
# --------------------------------------------------------------------------- #
class TestReactiveJammers:
    def test_reactive_higher_jnr_more_distortion(self) -> None:
        # With detect_threshold=0 and react_latency=0 the automaton engages on
        # the first symbol and holds, so the whole batch is jammed and the
        # added noise power -- hence distortion -- must grow with JNR.
        _seed_all(30)
        x = torch.randn(8, 512)
        jnr_ladder = [0.0, 10.0, 20.0]
        mse = []
        for jnr in jnr_ladder:
            channel = ReactiveJammer(
                jnr_db=jnr,
                detect_threshold=0.0,
                react_latency=0,
                hold_duration=512,
            ).eval()
            total = 0.0
            for t in range(30):
                _seed_all(3000 + t)
                total += torch.mean((channel(x) - x) ** 2).item()
            mse.append(total / 30)
        for m_lo, m_hi in zip(mse[:-1], mse[1:]):
            assert m_hi > m_lo, (
                f"ReactiveJammer distortion not increasing with JNR: {mse}"
            )

    def test_reactive_idle_when_silent(self) -> None:
        # A reactive (follower) jammer should not fire on an all-zero input:
        # there is no transmitter energy to react to, so the output must equal
        # the input. This is the defining behaviour distinguishing a reactive
        # jammer from an always-on barrage jammer.
        channel = ReactiveJammer(
            jnr_db=20.0, detect_threshold=1.0, react_latency=8, hold_duration=64
        ).eval()
        _seed_all(0)
        x = torch.zeros(4, 256)
        y = channel(x)
        assert torch.equal(y, x), "reactive jammer fired on a silent channel"
