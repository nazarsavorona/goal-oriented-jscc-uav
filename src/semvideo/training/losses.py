"""Loss functions for JSCC training: the three-scale SSIM term of the compact codec.

``ms_ssim_loss`` is the structural term of the paper's compact-codec objective: SSIM with an
11-pixel Gaussian window (sigma 1.5) at full, half and quarter resolution, averaged with
weights 0.5, 0.3 and 0.2. The plain MSE term and the detector-feature term are combined with
it in ``scripts/train.py``.
"""

import torch
import torch.nn.functional as F


def _ssim_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Compute SSIM map (differentiable, returns tensor not float).

    Args:
        pred: (B, C, H, W) in [0, 1].
        target: (B, C, H, W) in [0, 1].
        window_size: Gaussian window size.

    Returns:
        Mean SSIM value as a differentiable scalar tensor.
    """
    c1 = 0.01**2
    c2 = 0.03**2
    channels = pred.shape[1]

    gauss = torch.arange(window_size, dtype=pred.dtype, device=pred.device).float()
    gauss = gauss - window_size // 2
    gauss = torch.exp(-(gauss**2) / (2 * 1.5**2))
    gauss = gauss / gauss.sum()

    window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)

    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)

    sigma1_sq = F.conv2d(pred**2, window, padding=pad, groups=channels) - mu1**2
    sigma2_sq = F.conv2d(target**2, window, padding=pad, groups=channels) - mu2**2
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1 * mu2

    ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_val.mean()


def ms_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    scales: int = 3,
    window_size: int = 11,
) -> torch.Tensor:
    """Multi-scale SSIM loss (1 - MS-SSIM), differentiable.

    Computes SSIM at multiple downsampled resolutions and combines them.
    Following Wang et al. 2003, "Multiscale structural similarity for
    image quality assessment."

    Args:
        pred: (B, C, H, W) in [0, 1].
        target: (B, C, H, W) in [0, 1].
        scales: Number of downsampling scales.
        window_size: SSIM window size.

    Returns:
        1 - MS-SSIM as a differentiable scalar (lower is better).
    """
    weights = [0.5, 0.3, 0.2][:scales]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]  # renormalize so the loss can reach 0 for scales < 3
    total = torch.tensor(0.0, device=pred.device)

    for i in range(scales):
        ssim_val = _ssim_map(pred, target, window_size)
        total = total + weights[i] * ssim_val
        if i < scales - 1:
            pred = F.avg_pool2d(pred, 2)
            target = F.avg_pool2d(target, 2)

    return 1.0 - total
