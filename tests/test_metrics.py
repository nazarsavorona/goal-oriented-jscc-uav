"""Tests for quality metrics."""

import torch

from semvideo.metrics.psnr_ssim import psnr


class TestPSNR:
    """Tests for PSNR metric."""

    def test_identical_images_infinite_psnr(self) -> None:
        """PSNR of identical images should be infinity."""
        x = torch.rand(1, 3, 64, 64)
        assert psnr(x, x) == float("inf")

    def test_psnr_decreases_with_noise(self) -> None:
        """More noise should result in lower PSNR."""
        x = torch.rand(1, 3, 64, 64)
        noise_small = x + 0.01 * torch.randn_like(x)
        noise_large = x + 0.1 * torch.randn_like(x)
        assert psnr(noise_small, x) > psnr(noise_large, x)

    def test_known_value(self) -> None:
        """Test against a known PSNR value."""
        x = torch.zeros(1, 1, 2, 2)
        y = torch.ones(1, 1, 2, 2) * 0.1  # MSE = 0.01
        expected_psnr = 20.0  # 10 * log10(1/0.01) = 20
        assert abs(psnr(y, x) - expected_psnr) < 0.01
