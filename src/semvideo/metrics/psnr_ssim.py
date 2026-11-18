"""Classic image/video quality metrics: PSNR and SSIM."""

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted tensor.
        target: Ground truth tensor.
        max_val: Maximum pixel value (1.0 for normalized, 255.0 for uint8).

    Returns:
        PSNR value in dB.
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(max_val**2 / mse)).item()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    """Compute Structural Similarity Index (SSIM).

    Implements Wang et al., 2004, "Image quality assessment: from error
    visibility to structural similarity."

    Args:
        pred: Predicted tensor (B, C, H, W).
        target: Ground truth tensor (B, C, H, W).
        window_size: Size of the Gaussian window.
        max_val: Maximum pixel value.

    Returns:
        Mean SSIM value.
    """
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    channels = pred.shape[1]

    # Create Gaussian window
    gauss = torch.arange(window_size, dtype=pred.dtype, device=pred.device).float()
    gauss = gauss - window_size // 2
    gauss = torch.exp(-(gauss**2) / (2 * 1.5**2))
    gauss = gauss / gauss.sum()

    # 2D window via outer product
    window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)  # (K, K)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
    window = window.expand(channels, 1, -1, -1)  # (C, 1, K, K)

    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred**2, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target**2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return ssim_map.mean().item()
