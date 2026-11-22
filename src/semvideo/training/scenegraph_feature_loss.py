"""Vision-language patch-feature distillation loss (the SigLIP term of the paper).

Sister class to :class:`DINOv2FeatureLoss` and :class:`DetectorFeatureLoss`: an L1 distance
between the patch tokens that a frozen vision-language image encoder produces for the
reconstruction and for the original. The SwinJSCC-capacity chain of the paper uses it with
the SigLIP ViT-B/16 image encoder (``model_name="siglip"``, weight 0.05).

The module, the class and the command-line flags say "scenegraph" for historical reasons.
The names are kept because the training arguments stored with the paper's checkpoints use
them (``scenegraph_feat_weight``, ``scenegraph_model``). ``owlv2`` is the historical default
of ``model_name``; it appears in the stored arguments of the stages that ran with weight 0,
that is, without this term.

Supported backbones (open, frozen weights from the Hugging Face hub):

* **siglip** – Google SigLIP ViT-B/16 (``google/siglip-base-patch16-224``); the backbone
  used in the paper.
* **owlv2** – Google OWLv2 (``google/owlv2-base-patch16-ensemble``); not used in the paper.

All variants:

* are loaded frozen (``requires_grad=False``, ``eval()``);
* take ``(B, 3, H, W)`` in [0, 1] RGB;
* ImageNet-normalize internally and bilinearly resize to the model's
  native input size;
* return a single scalar L1 distillation loss between the patch features of
  ``pred`` and ``target``.

References:
  - Zhai et al., SigLIP (ICCV 2023, arxiv 2303.15343)
  - Minderer et al., OWLv2 (NeurIPS 2023, arxiv 2306.09683)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_SUPPORTED_MODELS: dict[str, str] = {
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "owlv2_base": "google/owlv2-base-patch16-ensemble",
    "owlv2_large": "google/owlv2-large-patch14-ensemble",
    "siglip": "google/siglip-base-patch16-224",
    "siglip_base": "google/siglip-base-patch16-224",
    "siglip_large": "google/siglip-large-patch16-256",
}


class SceneGraphFeatureLoss(nn.Module):
    """Frozen vision-language-backbone L1 distillation loss.

    Args:
        device: cuda/cpu device for the backbone.
        model_name: one of the keys in ``_SUPPORTED_MODELS`` (e.g.
            ``"owlv2"``, ``"siglip"``) or a fully-qualified HF model id.
        layers: optional 0-indexed hidden-state indices for multi-layer
            distillation. ``None`` (default) uses only the final patch
            token output, mirroring :class:`DINOv2FeatureLoss`.
        layer_weights: per-layer weights when ``layers`` is provided.
            Defaults to ``(0.25, 0.25, 0.50)`` for 3 layers (deep-bias)
            else uniform.
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "owlv2",
        layers: tuple[int, ...] | None = None,
        layer_weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name

        key = model_name.lower()
        hf_id = _SUPPORTED_MODELS.get(key, model_name)
        # Lazy import – transformers is only needed when this loss is used.
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(hf_id)
        # Pull the vision tower out of multi-modal wrappers so we only
        # carry the visual encoder weights on-device.
        vision_tower = self._extract_vision_tower(backbone)
        for p in vision_tower.parameters():
            p.requires_grad_(False)
        vision_tower.eval()
        vision_tower = vision_tower.to(device).float()
        self.vision_tower = vision_tower

        # Determine the native input resolution from the config when
        # available; otherwise fall back to 224.
        self._input_size = self._infer_input_size(backbone)
        # Number of leading non-patch tokens (CLS / register / pooled).
        # SigLIP & OWLv2 ViT towers expose hidden states without a CLS
        # token (they use mean-pooled outputs), so default 0.
        self._n_leading = 0

        if layers is None:
            self.layers: tuple[int, ...] | None = None
            self.layer_weights: tuple[float, ...] | None = None
        else:
            self.layers = tuple(int(layer) for layer in layers)
            if layer_weights is None:
                base = (0.25, 0.25, 0.50)
                if len(self.layers) == len(base):
                    self.layer_weights = base
                else:
                    self.layer_weights = tuple([1.0 / len(self.layers)] * len(self.layers))
            else:
                assert len(layer_weights) == len(self.layers)
                self.layer_weights = tuple(float(w) for w in layer_weights)

        self.register_buffer(
            "mean",
            torch.tensor(self._IMAGENET_MEAN, device=device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(self._IMAGENET_STD, device=device).view(1, 3, 1, 1),
        )

    @staticmethod
    def _extract_vision_tower(backbone: nn.Module) -> nn.Module:
        """Locate the visual encoder inside a HF multimodal wrapper.

        SigLIP exposes ``vision_model``; OWLv2 exposes
        ``owlv2.vision_model`` (via the Owlv2Model wrapper) or
        ``vision_model`` on Owlv2VisionModel. We probe the common
        attribute names and fall back to the whole model.
        """
        for attr in ("vision_model", "vision_tower"):
            tower = getattr(backbone, attr, None)
            if isinstance(tower, nn.Module):
                return tower
        # Owlv2Model wraps as `.owlv2.vision_model`
        inner = getattr(backbone, "owlv2", None)
        if inner is not None:
            tower = getattr(inner, "vision_model", None)
            if isinstance(tower, nn.Module):
                return tower
        return backbone

    @staticmethod
    def _infer_input_size(backbone: nn.Module) -> int:
        cfg = getattr(backbone, "config", None)
        # Multi-modal configs usually nest the vision config.
        for attr in ("vision_config",):
            sub = getattr(cfg, attr, None) if cfg is not None else None
            if sub is not None and hasattr(sub, "image_size"):
                return int(sub.image_size)
        if cfg is not None and hasattr(cfg, "image_size"):
            return int(cfg.image_size)
        return 224

    def _features(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """x in [0, 1] RGB -> patch tokens. Gradient flows to x."""
        x_norm = (x - self.mean) / self.std
        x_in = F.interpolate(
            x_norm,
            size=(self._input_size, self._input_size),
            mode="bilinear",
            align_corners=False,
        )
        if self.layers is None:
            out = self.vision_tower(pixel_values=x_in)
            tokens = out.last_hidden_state
            return tokens[:, self._n_leading :, :]
        out = self.vision_tower(pixel_values=x_in, output_hidden_states=True)
        hs = out.hidden_states  # tuple of (B, T, D), one per block
        feats: list[torch.Tensor] = []
        for li in self.layers:
            feats.append(hs[li][:, self._n_leading :, :])
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the scalar patch-feature L1.

        Args:
            pred: (B, 3, H, W) reconstruction in [0, 1] RGB.
            target: (B, 3, H, W) original in [0, 1] RGB.

        Returns: scalar loss (single-layer) or weighted sum
            (multi-layer) of L1 between the patch features of pred
            and target.
        """
        with torch.no_grad():
            feat_t = self._features(target)
        feat_p = self._features(pred)
        if self.layers is None:
            assert isinstance(feat_p, torch.Tensor)
            assert isinstance(feat_t, torch.Tensor)
            return F.l1_loss(feat_p, feat_t)
        assert isinstance(feat_p, list) and isinstance(feat_t, list)
        assert self.layer_weights is not None
        total = pred.new_zeros(())
        for fp, ft, w in zip(feat_p, feat_t, self.layer_weights, strict=False):
            total = total + w * F.l1_loss(fp, ft)
        return total
