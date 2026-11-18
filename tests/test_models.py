"""Tests for the model architecture: DeepJSCC."""

import pytest
import torch
import torch.nn as nn

from semvideo.channel.awgn import AWGNChannel
from semvideo.models.deep_jscc import DeepJSCC


class TestDeepJSCC:
    """Tests for the CNN-based DeepJSCC model."""

    @pytest.fixture
    def model(self) -> DeepJSCC:
        return DeepJSCC(channel_ratio=1 / 6, base_channels=64)

    def test_forward_shape(self, model: DeepJSCC) -> None:
        x = torch.randn(2, 3, 128, 128)
        recon = model(x)
        assert recon.shape == (2, 3, 128, 128)

    def test_forward_with_awgn(self, model: DeepJSCC) -> None:
        ch = AWGNChannel(snr_db=10.0)
        x = torch.randn(2, 3, 128, 128)
        recon = model(x, channel=ch, snr_db=10.0)
        assert recon.shape == (2, 3, 128, 128)

    def test_gradient_flow_through_channel(self, model: DeepJSCC) -> None:
        ch = AWGNChannel(snr_db=10.0)
        x = torch.randn(2, 3, 128, 128)
        recon = model(x, channel=ch, snr_db=10.0)
        loss = nn.functional.mse_loss(recon, x)
        loss.backward()
        for name, param in model.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for encoder.{name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for encoder.{name}"
                break

    def test_power_normalization(self, model: DeepJSCC) -> None:
        x = torch.randn(4, 3, 128, 128)
        symbols = model.encode(x)
        power = torch.mean(symbols ** 2, dim=-1)
        assert torch.allclose(power, torch.ones_like(power), atol=0.2)

    def test_compression_stats(self, model: DeepJSCC) -> None:
        stats = model.get_compression_stats((128, 128))
        assert stats["input_pixels"] == 3 * 128 * 128
        assert stats["compression_ratio"] > 10
