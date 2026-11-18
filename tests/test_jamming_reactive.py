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
