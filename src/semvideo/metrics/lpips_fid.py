"""Perceptual quality metrics: LPIPS using VGG16 features.

Implements Learned Perceptual Image Patch Similarity (LPIPS) following
Zhang et al. 2018 using VGG16 feature extraction. Uses raw feature L2
distances (no learned linear weights) to avoid extra dependencies.
"""

import torch
import torch.nn as nn
import torchvision.models as models

# ImageNet normalization constants
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# VGG16 layer indices for feature extraction (after ReLU)
_VGG_SLICES = {
    "relu1_2": 4,
    "relu2_2": 9,
    "relu3_3": 16,
    "relu4_3": 23,
    "relu5_3": 30,
}


class LPIPSMetric(nn.Module):
    """LPIPS perceptual distance using VGG16 features.

    Extracts multi-scale features from VGG16 and computes normalized L2
    distances. Lower values indicate more perceptually similar images.

    All VGG weights are frozen – this module has no trainable parameters,
    but gradients flow through the input images for use as a loss function.

    Args:
        device: Device for the VGG model.
    """

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()

        # Split VGG into slices at each extraction point
        slices = list(_VGG_SLICES.values())
        self.slices = nn.ModuleList()
        prev = 0
        for end in slices:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:end]))
            prev = end

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", _IMAGENET_MEAN)
        self.register_buffer("std", _IMAGENET_STD)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization."""
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS distance between predicted and target images.

        Args:
            pred: Predicted images (B, 3, H, W) in [0, 1].
            target: Target images (B, 3, H, W) in [0, 1].

        Returns:
            Mean LPIPS distance (scalar, differentiable w.r.t. pred).
        """
        pred_norm = self._normalize(pred)
        target_norm = self._normalize(target)

        total_dist = torch.tensor(0.0, device=pred.device)

        feat_pred = pred_norm
        feat_target = target_norm

        for layer_slice in self.slices:
            feat_pred = layer_slice(feat_pred)
            feat_target = layer_slice(feat_target)

            # Normalize features along channel dimension
            pred_n = feat_pred / (feat_pred.norm(dim=1, keepdim=True) + 1e-8)
            target_n = feat_target / (feat_target.norm(dim=1, keepdim=True) + 1e-8)

            # Mean squared difference over spatial dims, averaged over batch
            dist = (pred_n - target_n) ** 2
            total_dist = total_dist + dist.mean()

        return total_dist / len(self.slices)


def lpips(pred: torch.Tensor, target: torch.Tensor, model: LPIPSMetric | None = None) -> float:
    """Compute LPIPS distance (convenience function).

    Args:
        pred: Predicted tensor (B, C, H, W) in [0, 1].
        target: Ground truth tensor (B, C, H, W) in [0, 1].
        model: Pre-instantiated LPIPSMetric. Created on-the-fly if None.

    Returns:
        Mean LPIPS distance (lower is better).
    """
    if model is None:
        model = LPIPSMetric(device=pred.device.type).to(pred.device)
    with torch.no_grad():
        return model(pred, target).item()
