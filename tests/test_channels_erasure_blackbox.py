"""Black-box contract tests for the ERASURE / packet-loss channel family.

Modules under test (imported ONLY through their public API):
  - ``semvideo.channel.packet_loss``      – ``BernoulliErasureChannel``, ``BurstErasureChannel``
  - ``semvideo.channel.erasure_wrapped``  – ``ChannelWithErasure``
  - ``semvideo.channel.reactive_jam_wrapped`` – ``ChannelWithReactiveJam``

These tests treat each class as a BLACK BOX. They assert observable
input -> output contracts and domain properties derived from the
module docstrings-as-spec and the project verification rules
("Verification & Proof-Checking"):

  (a) output shape is preserved;
  (b) the empirical erasure fraction approximates the requested probability
      over many trials (statistical, with tolerance);
  (c) p = 0 -> output equals the clean-channel baseline;
  (d) erased positions are actually zeroed/dropped per the spec;
  (e) burst erasures are contiguous runs (spec), not i.i.d.;
  (f) determinism under a fixed seed.

No private attribute is asserted on, no internal call order is checked, and no
"golden" tensor produced by the current implementation is pasted back in. Every
property is independently implied by the documented contract or the channel
math. Tests are deterministic: torch + numpy are seeded, tensors are small CPU
tensors, and tolerances are generous enough to absorb Monte-Carlo noise.

Style note: en-dash only, never em-dash.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from semvideo.channel.erasure_wrapped import ChannelWithErasure
from semvideo.channel.packet_loss import (
    BernoulliErasureChannel,
    BurstErasureChannel,
)
from semvideo.channel.reactive_jam_wrapped import ChannelWithReactiveJam


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Seed every RNG the channels can touch, for deterministic tests."""
    torch.manual_seed(0)
    np.random.seed(0)


class _FakeSwinJSCCChannel(nn.Module):
    """Mimic the SwinJSCC ``Channel.forward(feature, chan_param, avg_pwr)``.

    The wrappers (``ChannelWithErasure`` / ``ChannelWithReactiveJam``) document
    that they forward exactly this 3-argument signature to the base channel.
    We use a deterministic additive-AWGN stand-in so the "clean-channel
    baseline" is reproducible under a fixed seed (contract c).
    """

    def __init__(self) -> None:
        super().__init__()
        self.channel_type = "awgn"

    def forward(
        self, feature: torch.Tensor, chan_param: float, avg_pwr: bool = False
    ) -> torch.Tensor:
        snr_linear = 10.0 ** (chan_param / 10.0)
        sig_power = (feature * feature).mean()
        noise_var = sig_power / snr_linear
        return feature + torch.randn_like(feature) * noise_var.sqrt()


def _zero_fraction(t: torch.Tensor, tol: float = 1e-6) -> float:
    """Fraction of entries that are (numerically) exactly zero."""
    return (t.abs() < tol).float().mean().item()


def _max_run_length(flags: np.ndarray) -> int:
    """Longest contiguous run of True in a 1-D boolean array."""
    best = run = 0
    for v in flags:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def _num_runs(flags: np.ndarray) -> int:
    """Number of maximal contiguous True-runs in a 1-D boolean array."""
    flags = np.asarray(flags, dtype=bool)
    if flags.size == 0:
        return 0
    # A run starts where True and the previous entry is False (or at index 0).
    starts = flags & np.concatenate(([True], ~flags[:-1]))
    return int(starts.sum())


# ===========================================================================
# BernoulliErasureChannel
# ===========================================================================


class TestBernoulliErasureChannel:
    """I.i.d. per-symbol Bernoulli erasure: erased -> 0, fraction ~= p."""

    def test_a_output_shape_preserved(self) -> None:
        """Contract (a): output shape equals input shape, for several shapes."""
        ch = BernoulliErasureChannel(p=0.3, snr_db=None)
        for shape in [(2, 64), (4, 8, 16), (3, 5, 7, 9)]:
            x = torch.randn(*shape)
            y = ch(x)
            assert y.shape == x.shape

    def test_c_p_zero_is_clean_passthrough(self) -> None:
        """Contract (c): p = 0 reproduces the clean input exactly (lossless)."""
        ch = BernoulliErasureChannel(p=0.0, snr_db=None)
        x = torch.randn(4, 256)
        y = ch(x)
        assert torch.allclose(y, x)

    def test_c_negative_p_also_passthrough(self) -> None:
        """Spec: ``resolved_p <= 0`` -> all-ones mask, i.e. clean passthrough."""
        ch = BernoulliErasureChannel(p=0.3, snr_db=None)
        x = torch.randn(4, 256)
        y = ch(x, p=0.0)
        assert torch.allclose(y, x)

    def test_d_erased_positions_are_exactly_zero(self) -> None:
        """Contract (d): with p = 1 every symbol is dropped (exactly zero)."""
        ch = BernoulliErasureChannel(p=1.0, snr_db=None)
        x = torch.randn(4, 256) + 5.0  # strictly nonzero signal
        y = ch(x)
        assert torch.count_nonzero(y).item() == 0

    def test_d_surviving_symbols_unchanged_no_residual_noise(self) -> None:
        """Spec: with ``snr_db=None`` survivors are multiplied by 1 (untouched).

        Every output entry is therefore either exactly 0 (erased) or exactly
        equal to the input (survived) – a hard-erasure channel.
        """
        ch = BernoulliErasureChannel(p=0.5, snr_db=None)
        x = torch.randn(8, 512) + 3.0
        y = ch(x)
        erased = y.abs() < 1e-12
        # Survivors must match the input bit-for-bit (multiplicative mask of 1).
        assert torch.allclose(y[~erased], x[~erased])

    def test_b_empirical_fraction_approximates_p(self) -> None:
        """Contract (b): mean erased fraction ~= p over many symbols/trials.

        Erasure is i.i.d. Bernoulli(p), so the fraction of zeros concentrates
        on ``p`` by the law of large numbers. Use a large symbol count and a
        tolerance that comfortably exceeds the binomial standard error.
        """
        x = torch.randn(64, 2000) + 4.0  # nonzero so survivors are never ~0
        for p in (0.1, 0.25, 0.5, 0.75):
            ch = BernoulliErasureChannel(p=p, snr_db=None)
            fracs = [_zero_fraction(ch(x)) for _ in range(5)]
            mean_frac = float(np.mean(fracs))
            assert abs(mean_frac - p) < 0.02, (
                f"p={p}: empirical erased fraction {mean_frac:.4f} too far "
                f"from requested {p}"
            )

    def test_b_random_p_sampling_in_range(self) -> None:
        """Spec: ``p=None`` draws p ~ U[p_min, p_max]; mean fraction ~= mean p.

        With p sampled uniformly in [p_min, p_max] each forward, the expected
        erased fraction equals (p_min + p_max) / 2.
        """
        ch = BernoulliErasureChannel(p=None, p_min=0.2, p_max=0.6, snr_db=None)
        x = torch.randn(16, 4000) + 4.0
        fracs = [_zero_fraction(ch(x)) for _ in range(40)]
        mean_frac = float(np.mean(fracs))
        # Each draw is in [0.2, 0.6]; the empirical fraction must stay near it.
        assert all(0.2 - 0.05 <= f <= 0.6 + 0.05 for f in fracs)
        assert abs(mean_frac - 0.4) < 0.03

    def test_f_determinism_under_fixed_seed(self) -> None:
        """Contract (f): same seed -> identical output (reproducibility rule)."""
        ch = BernoulliErasureChannel(p=0.4, snr_db=None)
        x = torch.randn(4, 1024) + 2.0
        torch.manual_seed(123)
        y1 = ch(x)
        torch.manual_seed(123)
        y2 = ch(x)
        assert torch.equal(y1, y2)

    def test_gradient_flows_through_mask(self) -> None:
        """Verification rule: channel layers must be differentiable in ``x``.

        Gradient flows through the multiplicative mask to surviving symbols.
        """
        ch = BernoulliErasureChannel(p=0.3, snr_db=None)
        # Build a leaf tensor with grad tracking (offset baked in before the
        # requires_grad flag so ``x`` stays a leaf and ``x.grad`` populates).
        x = (torch.randn(4, 512) + 2.0).requires_grad_(True)
        y = ch(x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0.0

    def test_residual_snr_keeps_erased_exactly_zero(self) -> None:
        """Spec: even with residual AWGN, erased symbols stay at exactly zero.

        The docstring states noise is added only on surviving symbols
        (``noise * mask``), so erased positions remain hard zeros.
        """
        ch = BernoulliErasureChannel(p=1.0, snr_db=10.0)
        x = torch.randn(4, 256) + 5.0
        y = ch(x)
        assert torch.count_nonzero(y).item() == 0


# ===========================================================================
# BurstErasureChannel (Gilbert-Elliott)
# ===========================================================================


class TestBurstErasureChannel:
    """Gilbert-Elliott burst erasure: contiguous runs, analytic loss rate."""

    def test_a_output_shape_preserved(self) -> None:
        """Contract (a): output shape equals input shape."""
        ch = BurstErasureChannel(g_to_b=0.05, b_to_g=0.20)
        for shape in [(2, 128), (4, 8, 64), (3, 4, 5, 32)]:
            x = torch.randn(*shape)
            assert ch(x).shape == x.shape

    def test_c_no_transitions_no_loss_is_passthrough(self) -> None:
        """Contract (c): a chain that can never erase reproduces the input.

        With ``g_to_b=0`` the chain starts and stays in Good (pi_B = 0) and
        ``p_good=0`` means Good never erases -> a clean passthrough. This is
        the burst-channel analogue of "p = 0".
        """
        ch = BurstErasureChannel(g_to_b=0.0, b_to_g=0.20, p_good=0.0, p_bad=1.0)
        x = torch.randn(4, 1024)
        y = ch(x)
        assert torch.allclose(y, x)

    def test_d_erased_positions_are_exactly_zero(self) -> None:
        """Contract (d): erased symbols are multiplied by zero (hard drop)."""
        ch = BurstErasureChannel(
            g_to_b=0.10, b_to_g=0.10, p_good=0.0, p_bad=1.0, snr_db=None
        )
        x = torch.randn(8, 2048) + 4.0
        y = ch(x)
        erased = y.abs() < 1e-12
        # Hard erasure: survivors equal the input exactly, erased equal zero.
        assert torch.allclose(y[~erased], x[~erased])
        assert torch.equal(y[erased], torch.zeros_like(y[erased]))

    def test_b_empirical_loss_matches_stationary_rate(self) -> None:
        """Contract (b): mean erased fraction ~= analytic ``stationary_loss_rate``.

        The Gilbert-Elliott stationary loss is
        ``pi_G * p_good + pi_B * p_bad`` with ``pi_B = g_to_b/(g_to_b+b_to_g)``.
        Averaged over many long sequences the empirical fraction concentrates
        on this analytic value (project rule: compare channel output to the
        analytical distribution).
        """
        ch = BurstErasureChannel(
            g_to_b=0.04, b_to_g=0.16, p_good=0.0, p_bad=1.0, snr_db=None
        )
        analytic = ch.stationary_loss_rate()  # = 0.04/0.20 = 0.20
        x = torch.randn(64, 4000) + 4.0
        fracs = [_zero_fraction(ch(x)) for _ in range(5)]
        mean_frac = float(np.mean(fracs))
        assert abs(mean_frac - analytic) < 0.03, (
            f"empirical burst loss {mean_frac:.4f} far from analytic "
            f"{analytic:.4f}"
        )

    def test_e_erasures_are_contiguous_bursts_not_iid(self) -> None:
        """Contract (e): erased symbols cluster into contiguous runs.

        With ``p_good=0`` / ``p_bad=1`` every erased symbol is a Bad-state
        symbol, so erased positions are exactly the Bad-state runs and are
        contiguous by construction. We verify this two ways at a matched loss
        rate against an i.i.d. Bernoulli channel:

          * the longest burst run is markedly longer than i.i.d. would give;
          * the number of distinct runs is markedly smaller (fewer, longer
            bursts) than the many tiny isolated drops of an i.i.d. channel.
        """
        rate = 0.20
        # mean burst length = 1 / b_to_g = 10 symbols; pi_B = 0.20.
        burst = BurstErasureChannel(
            g_to_b=0.05, b_to_g=0.20, p_good=0.0, p_bad=1.0, snr_db=None
        )
        iid = BernoulliErasureChannel(p=rate, snr_db=None)

        n = 8000
        x = torch.randn(1, n) + 4.0

        burst_runs, burst_maxrun = [], []
        iid_runs, iid_maxrun = [], []
        for _ in range(5):
            be = (burst(x).abs() < 1e-12).reshape(-1).cpu().numpy()
            ie = (iid(x).abs() < 1e-12).reshape(-1).cpu().numpy()
            burst_runs.append(_num_runs(be))
            iid_runs.append(_num_runs(ie))
            burst_maxrun.append(_max_run_length(be))
            iid_maxrun.append(_max_run_length(ie))

        # Same overall loss, but bursts must look qualitatively different.
        assert np.mean(burst_maxrun) > 3 * np.mean(iid_maxrun), (
            f"burst max-run {np.mean(burst_maxrun):.1f} should dwarf iid "
            f"{np.mean(iid_maxrun):.1f}"
        )
        assert np.mean(burst_runs) < 0.5 * np.mean(iid_runs), (
            f"burst should have far fewer runs ({np.mean(burst_runs):.0f}) "
            f"than iid ({np.mean(iid_runs):.0f})"
        )

    def test_e_symbol_axis_is_respected(self) -> None:
        """Spec: the Markov chain advances along ``symbol_axis`` only.

        With ``symbol_axis=0`` the contiguous runs must form along axis 0, so a
        column read down axis 0 shows long runs while reading across axis 1
        (independent chains per column) does not bias toward long runs.
        """
        n_rows, n_cols = 4000, 8
        ch = BurstErasureChannel(
            g_to_b=0.05, b_to_g=0.20, p_good=0.0, p_bad=1.0,
            snr_db=None, symbol_axis=0,
        )
        x = torch.randn(n_rows, n_cols) + 4.0
        erased = (ch(x).abs() < 1e-12).cpu().numpy()
        # Runs down each column (the symbol axis) should be long bursts.
        col_maxruns = [_max_run_length(erased[:, c]) for c in range(n_cols)]
        assert float(np.mean(col_maxruns)) > 3.0, (
            f"expected long bursts along symbol_axis=0, got mean max-run "
            f"{np.mean(col_maxruns):.2f}"
        )

    def test_f_determinism_under_fixed_seed(self) -> None:
        """Contract (f): identical output under a fixed seed."""
        ch = BurstErasureChannel(
            g_to_b=0.05, b_to_g=0.20, p_good=0.0, p_bad=1.0, snr_db=None
        )
        x = torch.randn(4, 2000) + 2.0
        torch.manual_seed(7)
        y1 = ch(x)
        torch.manual_seed(7)
        y2 = ch(x)
        assert torch.equal(y1, y2)

    def test_stationary_loss_rate_matches_closed_form(self) -> None:
        """Spec: ``stationary_loss_rate`` equals pi_G*p_good + pi_B*p_bad."""
        g, b, pg, pb = 0.03, 0.12, 0.05, 0.9
        ch = BurstErasureChannel(g_to_b=g, b_to_g=b, p_good=pg, p_bad=pb)
        pi_b = g / (g + b)
        expected = (1.0 - pi_b) * pg + pi_b * pb
        assert abs(ch.stationary_loss_rate() - expected) < 1e-9


# ===========================================================================
# ChannelWithErasure (erasure_wrapped)
# ===========================================================================


class TestChannelWithErasure:
    """Wrapper composing a base channel with post-channel Bernoulli erasure."""

    def test_a_output_shape_preserved(self) -> None:
        """Contract (a): wrapper output shape equals the feature shape."""
        wrapped = ChannelWithErasure(_FakeSwinJSCCChannel(), p_max=0.4).train()
        x = torch.randn(2, 64, 96)
        assert wrapped(x, 10.0).shape == x.shape

    def test_c_p_max_zero_equals_base_channel(self) -> None:
        """Contract (c): p_max = 0 reproduces the base channel exactly.

        The wrapper docstring: "Set ``p_max`` to 0 to disable erasure
        entirely." Under a fixed seed the wrapped output must equal the raw
        base-channel output.
        """
        base = _FakeSwinJSCCChannel()
        wrapped = ChannelWithErasure(base, p_max=0.0).train()
        x = torch.randn(2, 64, 96)
        torch.manual_seed(42)
        expected = base(x, 10.0)
        torch.manual_seed(42)
        actual = wrapped(x, 10.0)
        assert torch.allclose(actual, expected)

    def test_c_eval_mode_skips_erasure(self) -> None:
        """Contract (c) (Dropout-like): eval mode passes the base channel through.

        With ``always_apply=False`` (default) erasure is training-only, so in
        eval the wrapped output equals the base-channel output under one seed.
        """
        base = _FakeSwinJSCCChannel()
        wrapped = ChannelWithErasure(base, p_max=0.4, always_apply=False).eval()
        x = torch.randn(2, 64, 96)
        torch.manual_seed(42)
        expected = base(x, 10.0)
        torch.manual_seed(42)
        actual = wrapped(x, 10.0)
        assert torch.allclose(actual, expected)

    def test_b_train_mode_mean_erasure_near_half_p_max(self) -> None:
        """Contract (b): with p ~ U[0, p_max], mean erased fraction ~= p_max/2.

        Each forward draws p uniformly from [0, p_max]; averaging the erased
        fraction over many forwards concentrates on E[p] = p_max / 2. The base
        AWGN channel is run at very high SNR so survivors stay nonzero and the
        only ~exact zeros are the erased symbols.
        """
        p_max = 0.4
        wrapped = ChannelWithErasure(
            _FakeSwinJSCCChannel(), p_max=p_max, always_apply=True
        ).eval()
        x = torch.randn(8, 4000, 1) * 5.0 + 5.0
        # SNR = 60 dB -> residual base noise is negligible vs the +5 offset.
        fracs = [_zero_fraction(wrapped(x, 60.0), tol=1e-3) for _ in range(30)]
        mean_frac = float(np.mean(fracs))
        assert abs(mean_frac - p_max / 2.0) < 0.05, (
            f"mean erased fraction {mean_frac:.4f} not near p_max/2="
            f"{p_max / 2:.2f}"
        )

    def test_always_apply_erases_in_eval(self) -> None:
        """Spec: ``always_apply=True`` forces erasure even in eval mode."""
        wrapped = ChannelWithErasure(
            _FakeSwinJSCCChannel(), p_max=0.4, always_apply=True
        ).eval()
        x = torch.randn(8, 2000, 1) * 5.0 + 5.0
        fracs = [_zero_fraction(wrapped(x, 60.0), tol=1e-3) for _ in range(8)]
        assert float(np.mean(fracs)) > 0.05

    def test_f_determinism_under_fixed_seed(self) -> None:
        """Contract (f): identical output under a fixed seed (train mode)."""
        wrapped = ChannelWithErasure(_FakeSwinJSCCChannel(), p_max=0.4).train()
        x = torch.randn(2, 256, 8)
        torch.manual_seed(55)
        y1 = wrapped(x, 15.0)
        torch.manual_seed(55)
        y2 = wrapped(x, 15.0)
        assert torch.equal(y1, y2)

    def test_gradient_flows_through_wrapper(self) -> None:
        """Verification rule: gradients flow through the wrapper to the input."""
        wrapped = ChannelWithErasure(
            _FakeSwinJSCCChannel(), p_max=0.2, always_apply=True
        ).eval()
        x = (torch.randn(2, 64, 16) + 2.0).requires_grad_(True)
        y = wrapped(x, 10.0)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0.0

    def test_rejects_invalid_p_max(self) -> None:
        """Spec: p_max outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError):
            ChannelWithErasure(_FakeSwinJSCCChannel(), p_max=-0.1)
        with pytest.raises(ValueError):
            ChannelWithErasure(_FakeSwinJSCCChannel(), p_max=1.5)


# ===========================================================================
# ChannelWithReactiveJam (reactive_jam_wrapped)
# ===========================================================================


class TestChannelWithReactiveJam:
    """Wrapper composing a base channel with post-channel reactive jamming.

    Reactive jamming ADDS broadband noise to detected symbols rather than
    zeroing them, so the erasure-fraction contract does not apply here; instead
    we test the wrapper's own contracts: shape preservation, ``jam_prob=0``
    passthrough, the Dropout-like eval gate, and seed determinism.
    """

    def test_a_output_shape_preserved(self) -> None:
        """Contract (a): wrapper output shape equals the feature shape."""
        wrapped = ChannelWithReactiveJam(
            _FakeSwinJSCCChannel(), jam_prob=1.0, always_apply=True
        ).eval()
        x = torch.randn(2, 256, 8) + 2.0
        assert wrapped(x, 10.0).shape == x.shape

    def test_c_jam_prob_zero_equals_base_channel(self) -> None:
        """Contract (c): jam_prob = 0 disables jamming (clean base channel).

        Spec: "jam_prob: Probability per forward of applying jam (0.0
        disables)." Under a fixed seed the wrapped output must equal the raw
        base-channel output.
        """
        base = _FakeSwinJSCCChannel()
        wrapped = ChannelWithReactiveJam(
            base, jam_prob=0.0, always_apply=True
        ).train()
        x = torch.randn(2, 256, 8) + 2.0
        torch.manual_seed(42)
        expected = base(x, 10.0)
        torch.manual_seed(42)
        actual = wrapped(x, 10.0)
        assert torch.allclose(actual, expected)

    def test_c_eval_mode_skips_jam(self) -> None:
        """Contract (c) (Dropout-like): eval + default always_apply -> passthrough."""
        base = _FakeSwinJSCCChannel()
        wrapped = ChannelWithReactiveJam(
            base, jam_prob=1.0, always_apply=False
        ).eval()
        x = torch.randn(2, 256, 8) + 2.0
        torch.manual_seed(42)
        expected = base(x, 10.0)
        torch.manual_seed(42)
        actual = wrapped(x, 10.0)
        assert torch.allclose(actual, expected)

    def test_d_jam_perturbs_output_when_applied(self) -> None:
        """Spec: when jam fires it adds noise -> output differs from clean base.

        This is the dual of the erasure contract (d): jammed symbols are
        perturbed (not dropped). We confirm the wrapper actually changes the
        signal vs the base-channel-only output when jamming is forced on.
        """
        base = _FakeSwinJSCCChannel()
        wrapped = ChannelWithReactiveJam(
            base, jam_prob=1.0, jnr_min_db=10.0, jnr_max_db=10.0,
            react_latency=0, hold_duration=64, detect_threshold=0.5,
            always_apply=True,
        ).eval()
        x = torch.randn(2, 512, 8) + 2.0
        torch.manual_seed(42)
        clean = base(x, 60.0)
        torch.manual_seed(42)
        jammed = wrapped(x, 60.0)
        # Some symbols must have been perturbed by the reactive jammer.
        assert not torch.allclose(jammed, clean)

    def test_f_determinism_under_fixed_seed(self) -> None:
        """Contract (f): identical output under a fixed seed."""
        wrapped = ChannelWithReactiveJam(
            _FakeSwinJSCCChannel(), jam_prob=1.0, always_apply=True
        ).eval()
        x = torch.randn(2, 256, 8) + 2.0
        torch.manual_seed(77)
        y1 = wrapped(x, 15.0)
        torch.manual_seed(77)
        y2 = wrapped(x, 15.0)
        assert torch.equal(y1, y2)

    def test_gradient_flows_through_wrapper(self) -> None:
        """Verification rule: gradients flow through the jam wrapper to input."""
        wrapped = ChannelWithReactiveJam(
            _FakeSwinJSCCChannel(), jam_prob=1.0, react_latency=0,
            always_apply=True,
        ).eval()
        x = (torch.randn(2, 256, 8) + 2.0).requires_grad_(True)
        y = wrapped(x, 10.0)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0.0

    def test_rejects_invalid_args(self) -> None:
        """Spec: jam_prob outside [0,1] and jnr_min > jnr_max raise ValueError."""
        with pytest.raises(ValueError):
            ChannelWithReactiveJam(_FakeSwinJSCCChannel(), jam_prob=1.5)
        with pytest.raises(ValueError):
            ChannelWithReactiveJam(_FakeSwinJSCCChannel(), jam_prob=-0.1)
        with pytest.raises(ValueError):
            ChannelWithReactiveJam(
                _FakeSwinJSCCChannel(), jnr_min_db=20.0, jnr_max_db=5.0
            )
