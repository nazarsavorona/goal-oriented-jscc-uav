"""Detector-feature-distillation loss for goal-oriented JSCC training.

Recipe blends SR4IR (CVPR 2024, arxiv 2404.01692) and Adapt-ICMH
(ECCV 2024, arxiv 2407.09853): replace VGG-LPIPS with an L1 distance
between feature maps of a frozen YOLOv8m backbone applied to the
reconstruction vs. the original frame. Encourages the codec to
preserve the high-frequency content YOLO needs to fire on small
vehicles – the PSNR-good, detection-bad failure mode.

Stages used (YOLOv8m at 256x256 input):
  - stage 4 (C2f, 192x32x32) ≈ P3 – fine-grained, dominant for
    small-object detection in aerial imagery
  - stage 6 (C2f, 384x16x16) ≈ P4 – mid-scale objects
  - stage 9 (SPPF, 576x8x8)  ≈ P5 – global context

Layer weights favour shallow high-res features (P3 ≫ P5) per
Adapt-ICMH's bias toward small-object preservation.

Reference:
  - Kim et al., SR4IR (CVPR 2024): https://arxiv.org/abs/2404.01692
  - Liu et al., Adapt-ICMH (ECCV 2024): https://arxiv.org/abs/2407.09853
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

# ImageNet stats – YOLOv8 pretraining uses [0,1] inputs (no per-channel norm),
# but extracting features from {arbitrary} models tends to be more stable when
# normalised. YOLOv8 specifically expects [0,1] BGR/RGB without normalisation,
# so we leave inputs untouched. If misaligned domain hurts, set NORMALISE=True.
NORMALISE = False
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DetectorFeatureLoss(nn.Module):
    """Frozen YOLOv8m backbone feature distillation (L1).

    Args:
        weights_path: path to YOLOv8 weights (default: yolov8m.pt in cwd).
        stages: tuple of stage indices to extract features at. Defaults to
            ``(4, 6, 9)`` = P3/P4/P5.
        stage_weights: per-stage weight in the loss sum. Defaults to
            ``(1.0, 0.5, 0.25)`` favouring shallow high-res features.
        criterion: ``"l1"`` (recommended, SR4IR-style) or ``"mse"`` (Adapt-ICMH).
        device: cuda or cpu.
    """

    def __init__(
        self,
        weights_path: str = "yolov8m.pt",
        stages: tuple[int, ...] = (4, 6, 9),
        stage_weights: tuple[float, ...] = (1.0, 0.5, 0.25),
        criterion: str = "l1",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        # Lazy import – ultralytics is a heavy dep and not always present
        from ultralytics import YOLO

        if not Path(weights_path).exists():
            # Fall back to the cwd default; ultralytics auto-downloads if absent
            weights_path = "yolov8m.pt"
        yolo = YOLO(weights_path)
        # Assert this is YOLOv8m: stages 0..9 are straight-sequential (no
        # skip-from for f != -1). If a user swaps in v8x/v10/v11, FPN topology
        # changes (some f attributes become non-(-1)) and the sequential
        # extraction below would silently return wrong features.
        # This is the only supported configuration.
        for layer in yolo.model.model[: max(stages) + 1]:
            f_attr = getattr(layer, "f", -1)
            if isinstance(f_attr, list):
                assert all(x == -1 for x in f_attr), (
                    f"Detector backbone layer expects skip-from {f_attr}; only "
                    "straight-sequential v8m backbones (all f=-1) are supported."
                )
            else:
                assert f_attr == -1, (
                    f"Detector backbone layer expects skip-from {f_attr}; only "
                    "straight-sequential v8m backbones (all f=-1) are supported."
                )
        # The detection model lives at yolo.model (a `DetectionModel` nn.Module)
        # whose `.model` attribute is the sequential backbone+neck+head ModuleList
        backbone = yolo.model.model[: max(stages) + 1]
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        # Move to device – CompressAI's amp interaction prefers explicit FP32
        backbone = backbone.to(device).float()
        self.backbone = backbone
        self.stages = tuple(stages)
        self.stage_weights = tuple(stage_weights)
        assert len(self.stages) == len(self.stage_weights)
        self.crit = nn.L1Loss() if criterion == "l1" else nn.MSELoss()
        # YOLOv8 expects raw [0,1] RGB; inputs in our codebase are also [0,1] RGB
        # (FrameDataset returns torch.from_numpy(img).float().permute(2,0,1)/255.).
        if NORMALISE:
            self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std if NORMALISE else x

    def _extract(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run sequential stages, capture features at the requested indices."""
        h = self._normalise(x)
        feats: list[torch.Tensor] = []
        max_stage = max(self.stages)
        # YOLOv8 stages may take from non-previous (`f != -1`) for skip-conn,
        # but for stages 0..9 it's straight sequential. Asserted by inspection.
        for i, layer in enumerate(self.backbone):
            h = layer(h)
            if i in self.stages:
                feats.append(h)
            if i >= max_stage:
                break
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute scalar loss.

        Args:
            pred: (B, 3, H, W) reconstruction in [0, 1] RGB.
            target: (B, 3, H, W) original in [0, 1] RGB.

        Returns: scalar loss = Σ_l w_l · L1(F_l(pred), F_l(target)).
        """
        # Target features: fully detached (frozen backbone) – but graph still
        # propagates through pred features for the codec.
        with torch.no_grad():
            feats_t = self._extract(target)
        feats_p = self._extract(pred)
        total = pred.new_zeros(())
        for w, fp, ft in zip(self.stage_weights, feats_p, feats_t, strict=False):
            total = total + w * self.crit(fp, ft)
        return total
