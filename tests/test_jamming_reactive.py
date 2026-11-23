"""Tests for ``ReactiveJammer`` (5th jamming mode, arXiv 2502.02385)."""

from __future__ import annotations

import pytest
import torch

from semvideo.channel.jamming_reactive import ReactiveJammer


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def test_shape_preserved():
    j = ReactiveJammer(jnr_db=10.0)
    x = torch.randn(2, 1024)
    y = j(x)
    assert y.shape == x.shape


def test_zero_jnr_minimal_perturbation():
    """At very low JNR, the jammer adds essentially no noise even when active."""
    # detect_threshold=0.5 ensures the detector fires on energy=1.0 input.
    j = ReactiveJammer(jnr_db=-40.0, detect_threshold=0.5,
                       react_latency=0, hold_duration=128)
    x = torch.ones(2, 1024)
    y = j(x)
    # Noise floor at JNR=-40 is sig_power * 1e-4 -> RMS ~ 0.01 vs signal 1.0
    assert (y - x).abs().mean().item() < 0.1


def test_high_jnr_significantly_perturbs():
    j = ReactiveJammer(jnr_db=20.0, detect_threshold=0.5,
                       react_latency=0, hold_duration=512)
    x = torch.ones(2, 1024)
    y = j(x)
    # At JNR=20 dB the noise RMS is 10x the signal – output should be far from x
    assert (y - x).abs().mean().item() > 1.0


def test_latency_delays_engagement():
    """With high react_latency, only later symbols should be jammed."""
    j = ReactiveJammer(jnr_db=20.0, detect_threshold=0.5,
                       react_latency=200, hold_duration=312)
    x = torch.ones(1, 512)
    y = j(x)
    # First ~200 symbols should be mostly clean (no jam engaged yet)
    early_diff = (y[:, :100] - x[:, :100]).abs().mean().item()
    late_diff = (y[:, 300:500] - x[:, 300:500]).abs().mean().item()
    # If the automaton respects latency, late portion is more disturbed
    assert late_diff > early_diff, (
        f"latency should delay jamming: early={early_diff:.3f}, late={late_diff:.3f}"
    )


def test_per_sample_independent_pattern():
    """Different batch rows must see different jamming patterns (not shared)."""
    j = ReactiveJammer(jnr_db=10.0, detect_threshold=0.5,
                       react_latency=0, hold_duration=64)
    # Two rows with identical content – noise is per-element-random so the
    # jammed outputs should differ even though the on/off mask matches.
    x = torch.ones(2, 1024)
    y = j(x)
    diff = (y[0] - y[1]).abs().mean().item()
    assert diff > 0.01, f"per-sample patterns identical: diff={diff:.6f}"


def test_rejects_invalid_params():
    with pytest.raises(ValueError):
        ReactiveJammer(react_latency=-1)
    with pytest.raises(ValueError):
        ReactiveJammer(hold_duration=0)


# ---------------------------------------------------------------------------
# Power conventions: the mean-referenced variant that trained the reactive-augmented model
# ---------------------------------------------------------------------------


def _ramp_input() -> torch.Tensor:
    """Symbols with very different energies, so the two conventions must differ."""
    torch.manual_seed(1)
    scale = torch.linspace(0.2, 3.0, 2048)
    return torch.randn(4, 2048) * scale


def test_rejects_unknown_power_reference():
    with pytest.raises(ValueError):
        ReactiveJammer(power_reference="peak")


def test_mean_reference_matches_earlier_formula_bit_for_bit():
    """'mean' reproduces the earlier jammer: noise std sqrt(mean energy * JNR), same draw."""
    x = _ramp_input()
    jnr_db = 10.0
    j = ReactiveJammer(jnr_db=jnr_db, detect_threshold=0.5, react_latency=8,
                       hold_duration=64, power_reference="mean")
    torch.manual_seed(123)
    y = j(x)

    # Reference: the earlier implementation, written out.
    energy = x.detach().abs() ** 2
    mean_e = float(energy.mean().clamp(min=1e-12).item())
    mask = j._build_jam_mask(energy, mean_e)
    noise_var = mean_e * 10.0 ** (jnr_db / 10.0)
    torch.manual_seed(123)
    noise = torch.randn_like(x) * (noise_var**0.5)
    y_ref = torch.where(mask, x + noise, x)
    assert torch.equal(y, y_ref)


def test_mean_reference_interference_does_not_follow_symbol_energy():
    """Under 'mean' every gated symbol gets the same variance; under 'followed' it scales."""
    x = _ramp_input()
    energy = x.abs() ** 2
    mean_e = float(energy.mean().item())
    jnr_lin = 10.0 ** (10.0 / 10.0)
    for ref in ("mean", "followed"):
        j = ReactiveJammer(jnr_db=10.0, detect_threshold=0.5, react_latency=0,
                           hold_duration=4096, power_reference=ref)
        torch.manual_seed(7)
        d = j(x) - x
        mask = d != 0
        assert mask.float().mean().item() > 0.5  # latched on for most of the block
        lo = mask & (energy < 0.5 * mean_e)
        hi = mask & (energy > 2.0 * mean_e)
        var_lo = float((d[lo] ** 2).mean())
        var_hi = float((d[hi] ** 2).mean())
        if ref == "mean":
            # same expected variance mean_e * JNR in both energy groups
            assert var_lo == pytest.approx(mean_e * jnr_lin, rel=0.15)
            assert var_hi == pytest.approx(mean_e * jnr_lin, rel=0.15)
        else:
            assert var_hi > 4.0 * var_lo


def test_mean_reference_leaves_ungated_symbols_and_reports_power():
    x = _ramp_input()
    j = ReactiveJammer(jnr_db=10.0, detect_threshold=1.0, react_latency=8,
                       hold_duration=64, power_reference="mean")
    y = j(x, snr_db=10.0)
    energy = x.abs() ** 2
    mask = j._build_jam_mask(energy, float(energy.mean().item()))
    assert torch.equal(y[~mask], x[~mask])
    # injected energy / equal-JNR barrage budget = duty * SNR_linear under this convention
    assert j.last_power_ratio == pytest.approx(j.last_duty * 10.0, rel=1e-5)


def test_wrapper_forwards_power_reference():
    from semvideo.channel.reactive_jam_wrapped import ChannelWithReactiveJam

    class _Identity(torch.nn.Module):
        def forward(self, feature, chan_param, avg_pwr=False):  # noqa: D102
            return feature

    w = ChannelWithReactiveJam(_Identity(), jam_prob=1.0, power_reference="mean")
    assert w.jammer.power_reference == "mean"
    assert "power_reference=mean" in w.extra_repr()
    assert ChannelWithReactiveJam(_Identity()).jammer.power_reference == "followed"
