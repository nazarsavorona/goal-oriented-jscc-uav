"""Tests for :class:`SceneGraphFeatureLoss`.

Verifies the frozen vision-language supervisor (the SigLIP term of the paper)
produces a scalar loss and that gradients flow back to the reconstruction
tensor. Uses SigLIP-base on CPU. Skipped automatically when the HuggingFace
weights are not cacheable (offline CI).
"""

from __future__ import annotations

import pytest
import torch


def _try_load() -> object | None:
    try:
        from semvideo.training.scenegraph_feature_loss import (
            SceneGraphFeatureLoss,
        )

        return SceneGraphFeatureLoss(device="cpu", model_name="siglip")
    except Exception as exc:  # network/auth/HF failures -> skip, not fail.
        pytest.skip(f"SigLIP backbone unavailable: {exc!r}")
        return None


class TestSceneGraphFeatureLoss:
    """Single-layer L1 distillation over a frozen SigLIP backbone."""

    def test_forward_returns_scalar(self) -> None:
        loss_fn = _try_load()
        pred = torch.rand(2, 3, 224, 224)
        target = torch.rand(2, 3, 224, 224)
        value = loss_fn(pred, target)
        assert value.dim() == 0, "the loss must be a 0-D scalar"
        assert torch.isfinite(value), "loss must be finite"

    def test_gradient_flows_to_pred(self) -> None:
        loss_fn = _try_load()
        pred = torch.rand(2, 3, 224, 224, requires_grad=True)
        target = torch.rand(2, 3, 224, 224)
        value = loss_fn(pred, target)
        value.backward()
        assert pred.grad is not None
        assert float(pred.grad.abs().sum()) > 0.0

    def test_backbone_frozen(self) -> None:
        loss_fn = _try_load()
        n_trainable = sum(
            p.numel()
            for p in loss_fn.vision_tower.parameters()  # type: ignore[attr-defined]
            if p.requires_grad
        )
        assert n_trainable == 0, "backbone params must be frozen"
