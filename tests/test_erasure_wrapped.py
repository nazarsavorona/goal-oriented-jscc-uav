"""Tests for ``ChannelWithErasure`` (replaces train-time monkey-patch)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from semvideo.channel.awgn import AWGNChannel
from semvideo.channel.erasure_wrapped import ChannelWithErasure


class _FakeSwinJSCCChannel(nn.Module):
    """Mimics SwinJSCC ``Channel.forward(feature, chan_param, avg_pwr)``."""

    def __init__(self):
        super().__init__()
        self.channel_type = "awgn"

    def forward(self, feature: torch.Tensor, chan_param: float,
                avg_pwr: bool = False) -> torch.Tensor:
        snr_linear = 10.0 ** (chan_param / 10.0)
        sig_power = (feature * feature).mean()
        noise_var = sig_power / snr_linear
        return feature + torch.randn_like(feature) * noise_var.sqrt()


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def test_passthrough_at_p_max_0():
    """p_max=0 must be a perfect passthrough (forward equals base channel)."""
    base = _FakeSwinJSCCChannel()
    wrapped = ChannelWithErasure(base, p_max=0.0).train()
    x = torch.randn(2, 64, 96)
    torch.manual_seed(42)
    expected = base(x, 10.0)
    torch.manual_seed(42)
    actual = wrapped(x, 10.0)
    assert torch.allclose(actual, expected)


def test_erasure_active_in_train_mode():
    """At p_max > 0 in training, *some* run must produce visible erasure.

    Since p ~ U[0, p_max] per forward, a single forward may draw p ≈ 0 and
    show no erasure. Average across several runs to make the expectation
    deterministic.
    """
    base = _FakeSwinJSCCChannel()
    wrapped = ChannelWithErasure(base, p_max=0.40).train()
    x = torch.randn(4, 512, 16) * 5.0
    zero_fractions = []
    for _ in range(8):
        out = wrapped(x, 20.0)
        zero_fractions.append((out.abs() < 1e-6).float().mean().item())
    # Mean expected zero fraction = E[p] = p_max/2 = 0.20. Allow plenty of
    # slack for sampling noise.
    mean_zeros = sum(zero_fractions) / len(zero_fractions)
    assert mean_zeros > 0.05, (
        f"expected average erasure across 8 forwards >0.05, "
        f"got {mean_zeros:.4f} (per-run: {zero_fractions})"
    )


def test_eval_mode_skips_erasure():
    """In eval mode with always_apply=False (default), erasure is bypassed."""
    base = _FakeSwinJSCCChannel()
    wrapped = ChannelWithErasure(base, p_max=0.40, always_apply=False).eval()
    x = torch.randn(2, 64, 96)
    torch.manual_seed(42)
    expected = base(x, 10.0)
    torch.manual_seed(42)
    actual = wrapped(x, 10.0)
    assert torch.allclose(actual, expected), "eval should passthrough"


def test_always_apply_overrides_eval():
    """always_apply=True forces erasure even in eval mode (packet-loss eval)."""
    base = _FakeSwinJSCCChannel()
    wrapped = ChannelWithErasure(base, p_max=0.40, always_apply=True).eval()
    x = torch.randn(4, 512, 16) * 5.0
    zero_fractions = []
    for _ in range(8):
        out = wrapped(x, 20.0)
        zero_fractions.append((out.abs() < 1e-6).float().mean().item())
    mean_zeros = sum(zero_fractions) / len(zero_fractions)
    assert mean_zeros > 0.05, (
        f"always_apply should erase in eval mode; got mean {mean_zeros:.4f}"
    )


def test_state_dict_roundtrip_preserves_config():
    """state_dict() round-trip must preserve the wrapper structure."""
    base = _FakeSwinJSCCChannel()
    w1 = ChannelWithErasure(base, p_max=0.40, always_apply=False)
    sd = w1.state_dict()

    base2 = _FakeSwinJSCCChannel()
    w2 = ChannelWithErasure(base2, p_max=0.40, always_apply=False)
    w2.load_state_dict(sd)
    # Sanity: same params, same buffers, no drift after load.
    for (k1, v1), (k2, v2) in zip(w1.state_dict().items(),
                                    w2.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(v1, v2)


def test_gradient_flows_through_wrapper():
    """Gradients must flow through the wrapper to the base channel's input.

    Use the FakeSwinJSCCChannel which mimics the SwinJSCC ``Channel.forward``
    signature ``(feature, chan_param, avg_pwr)`` – the same signature the
    wrapper assumes from the real SwinJSCC channel.
    """
    base = _FakeSwinJSCCChannel()
    wrapped = ChannelWithErasure(base, p_max=0.2, always_apply=False).eval()
    x = torch.randn(2, 64, 16, requires_grad=True)
    y = wrapped(x, 10.0)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


def test_rejects_invalid_p_max():
    base = _FakeSwinJSCCChannel()
    with pytest.raises(ValueError):
        ChannelWithErasure(base, p_max=-0.1)
    with pytest.raises(ValueError):
        ChannelWithErasure(base, p_max=1.5)
