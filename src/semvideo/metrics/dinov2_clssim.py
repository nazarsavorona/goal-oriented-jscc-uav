"""DINOv2 class-token similarity and image PSNR: the metric path of the jamming tables.

These are the definitions that ``scripts/eval_benchmark_sweep.py`` has always used and that
the other evaluators import from it. They live here so that a script can use the metric
without importing ``eval_benchmark_sweep``, which loads the upstream SwinJSCC code and
changes the working directory at import time. ``eval_benchmark_sweep`` re-exports them, so
both import paths give the same objects.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINO_RES = 518


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB between two 8-bit images of the same shape (peak 255)."""
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


class DINOv2Extractor:
    """Default backbone: DINOv2 ViT-S/14. Swap via ``backbone`` argument:
    ``dinov2_vits14`` (default, 21 M params, 384 feat dim, 518×518 input),
    ``dinov2_vitb14`` (86 M, 768 dim), ``dinov2_vitl14`` (300 M, 1024 dim).
    """

    def __init__(self, backbone: str = "dinov2_vits14"):
        self.backbone = backbone
        self.model = (
            torch.hub.load(
                "facebookresearch/dinov2",
                backbone,
                verbose=False,
            )
            .to(DEVICE)
            .eval()
        )
        self.tx = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((DINO_RES, DINO_RES)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    @torch.no_grad()
    def features(self, frame_bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the class token ``(D,)`` and the patch tokens ``(P, D)`` of a BGR frame."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = self.tx(rgb).unsqueeze(0).to(DEVICE)
        out = self.model.forward_features(x)
        return out["x_norm_clstoken"].squeeze(0), out["x_norm_patchtokens"].squeeze(0)


def cossim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two tokens, or the mean over matching rows of two token sets."""
    if a.dim() == 1:
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    return float(F.cosine_similarity(a, b, dim=-1).mean().item())
