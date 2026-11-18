"""DINOv2-feature L1 distillation loss for goal-oriented JSCC training.

Sister class to DetectorFeatureLoss but uses frozen DINOv2 ViT-S/14 patch
tokens instead of a YOLO backbone. Motivation:
detector-feature distillation gave +0.086 DINOv2 cls_sim @ SNR=20 over
the no-feat-loss baseline. Using DINOv2 itself as the feature extractor
is the natural next step – directly optimize what we measure.

Reference:
  - Oquab et al., DINOv2 (Meta AI 2023, arxiv 2304.07193)
  - Same recipe as Adapt-ICMH (ECCV 2024) but with self-supervised
    foundation model features instead of detector features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2FeatureLoss(nn.Module):
    """Frozen DINOv2 ViT-S/14 patch-token L1 distillation.

    Input is expected to be (B, 3, H, W) in [0, 1] RGB.
    Internally normalizes with ImageNet mean/std, then bilinearly resizes
    to 518×518 (DINOv2's patch-grid-friendly resolution: 518 = 14 × 37).

    Default (``layers=None``): use only the final patch-token output –
    Loss = L1(F_dino(pred), F_dino(target)) on x_norm_patchtokens (B × 1369 × 384).

    Multi-layer mode (``layers=(5, 8, 11)`` etc., 0-indexed of 12 ViT blocks):
    Following the standard DINOv2-segmentation recipe and ViTKD (Yang CVPRW 2024),
    distil multiple intermediate block outputs simultaneously. Loss is the
    layer-weighted sum of per-layer L1, where ``layer_weights`` favours deeper
    blocks (default (0.25, 0.25, 0.50) for shallow/mid/deep).
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        device: str = "cuda",
        layers: tuple[int, ...] | None = None,
        layer_weights: tuple[float, ...] | None = None,
        model_name: str = "dinov2_vits14",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        if model_name.startswith("dinov3"):
            # DINOv3 via HuggingFace transformers
            from transformers import AutoModel

            hf_id = {
                "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            }.get(model_name, model_name)
            backbone = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
            self._is_dinov3 = True
            self._input_size = 224  # DINOv3 ViT-S/16 natively 224 (14×14=196 patches)
            self._n_register = 4  # DINOv3 has 4 register tokens + 1 CLS
        else:
            backbone = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                verbose=False,
            )
            self._is_dinov3 = False
            self._input_size = 518
            self._n_register = 0
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        backbone = backbone.to(device).float()
        self.backbone = backbone
        if layers is None:
            self.layers: tuple[int, ...] | None = None
            self.layer_weights: tuple[float, ...] | None = None
        else:
            self.layers = tuple(int(l) for l in layers)
            if layer_weights is None:
                # Default: favour deep blocks slightly per ViTKD
                base = (0.25, 0.25, 0.50)
                if len(self.layers) == len(base):
                    self.layer_weights = base
                else:
                    self.layer_weights = tuple([1.0 / len(self.layers)] * len(self.layers))
            else:
                assert len(layer_weights) == len(self.layers)
                self.layer_weights = tuple(float(w) for w in layer_weights)
        # Build buffers directly on `device` so they ride with the parent without
        # needing a separate self.to(device) call (which would also try to move
        # the frozen DINOv2 backbone, redundant).
        self.register_buffer(
            "mean", torch.tensor(self._IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(self._IMAGENET_STD, device=device).view(1, 3, 1, 1)
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """x in [0, 1] RGB → patch tokens. Gradient flows to x."""
        x_norm = (x - self.mean) / self.std
        x_in = F.interpolate(
            x_norm, size=(self._input_size, self._input_size), mode="bilinear", align_corners=False
        )
        if self._is_dinov3:
            # HF DINOv3: last_hidden_state shape (B, 1 + n_register + N_patches, D).
            # For multi-layer access we'd need output_hidden_states=True.
            if self.layers is None:
                out = self.backbone(x_in)
                # Drop CLS (idx 0) + n_register tokens (idx 1..n_register).
                return out.last_hidden_state[:, 1 + self._n_register :, :]
            out = self.backbone(x_in, output_hidden_states=True)
            hs = out.hidden_states  # tuple of (B, T, D), one per block
            feats = []
            for li in self.layers:
                feats.append(hs[li][:, 1 + self._n_register :, :])
            return feats
        # DINOv2 path (torch.hub)
        if self.layers is None:
            out = self.backbone.forward_features(x_in)
            return out["x_norm_patchtokens"]
        feats = self.backbone.get_intermediate_layers(
            x_in,
            n=list(self.layers),
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        return list(feats)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat_t = self._features(target)
        feat_p = self._features(pred)
        if self.layers is None:
            return F.l1_loss(feat_p, feat_t)
        # Multi-layer: weighted sum of per-layer L1
        assert isinstance(feat_p, list) and isinstance(feat_t, list)
        assert self.layer_weights is not None
        total = pred.new_zeros(())
        for fp, ft, w in zip(feat_p, feat_t, self.layer_weights, strict=False):
            total = total + w * F.l1_loss(fp, ft)
        return total
