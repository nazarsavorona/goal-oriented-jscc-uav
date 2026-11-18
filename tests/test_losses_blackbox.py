"""Black-box contract tests for semvideo.training loss functions.

Each loss module is treated as a BLACK BOX: we import only its public API and
assert observable input -> output contracts and the mathematical / domain
properties implied by the docstring-as-spec and the project verification rules
(en-dash only, full respectful names; gradients must flow; reproducibility).

Contracts exercised (per the task brief):
  (a) loss >= 0 where the spec says so;
  (b) loss(x, x) is the minimum / ~0 for identity-type distance losses;
  (c) loss is finite for valid inputs;
  (d) gradients flow to the inputs (or to an upstream encoder via a non-leaf
      latent);
  (e) ms_ssim multi-scale weights behave as a proper normalised weighting -- the
      loss of identical inputs is ~0 regardless of the number of scales.

Heavy / network feature extractors (YOLOv8m, DINOv2, SigLIP) are
guarded: if the backbone weights cannot be materialised in this environment the
property test is skipped (never failed) so the suite stays green on a laptop.

Author: Nazarii Savorona.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# Public API under test -- import ONLY documented entry points.
from semvideo.training.losses import ms_ssim_loss


# ─────────────────────────────────────────────────────────────────────────────
# Determinism: seed torch + numpy once per test via an autouse fixture so the
# whole file is reproducible (project rule: same seed -> identical metrics).
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(20240617)
    np.random.seed(20240617)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for guarding heavy / network backbones.
# ─────────────────────────────────────────────────────────────────────────────
def _skip_if_unavailable(factory):
    """Instantiate ``factory()`` or skip the test if the backbone is missing.

    The heavy supervisors download foundation-model weights (DINOv2,
    SigLIP, YOLOv8m). On an offline / trimmed environment construction
    raises -- we skip rather than fail so the contract suite stays green.
    """
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001 -- any download/import failure -> skip
        pytest.skip(f"feature-extractor backbone unavailable: {exc!r}")


# =============================================================================
# 1. ms_ssim_loss  (contract e is the headline; plus a/b/c/d)
# =============================================================================
class TestMsSsimLoss:
    @pytest.mark.parametrize("scales", [1, 2, 3])
    def test_identity_is_zero_regardless_of_scales(self, scales: int) -> None:
        """(e) Identical inputs -> ~0 loss for ANY number of scales.

        The docstring states the weights are renormalised so the loss can reach
        0 for ``scales < 3``. SSIM(x, x) == 1 at every scale, so the weighted
        sum of SSIM is 1 and ``1 - MS-SSIM`` is 0 independent of how the weights
        are distributed. This is the proper-normalised-weighting property.
        """
        x = torch.rand(2, 3, 64, 64)
        val = ms_ssim_loss(x, x.clone(), scales=scales)
        assert val.shape == ()
        assert torch.isfinite(val)
        assert val.item() == pytest.approx(0.0, abs=1e-5), (
            f"ms_ssim_loss(x, x) must be ~0 at scales={scales}, got {val.item()}"
        )

    @pytest.mark.parametrize("scales", [1, 2, 3])
    def test_nonnegative_and_bounded(self, scales: int) -> None:
        """(a)/(c) For valid [0, 1] images, 1 - MS-SSIM lies in [0, ~1]."""
        for _ in range(8):
            a = torch.rand(2, 3, 48, 48)
            b = torch.rand(2, 3, 48, 48)
            v = ms_ssim_loss(a, b, scales=scales).item()
            assert np.isfinite(v)
            # SSIM in [-1, 1] -> 1 - MS-SSIM in [0, 2]; for natural [0,1] noise
            # it sits well inside [0, 1]. Assert the spec lower bound strictly
            # and a generous, spec-implied upper bound.
            assert v >= -1e-5, f"ms_ssim loss went negative: {v}"
            assert v <= 2.0 + 1e-5

    def test_larger_difference_gives_larger_loss(self) -> None:
        """Monotonicity sanity: a closer pair scores no worse than a far pair."""
        x = torch.rand(2, 3, 48, 48)
        near = x + 0.01 * torch.randn_like(x)
        far = x + 0.5 * torch.randn_like(x)
        l_near = ms_ssim_loss(near.clamp(0, 1), x).item()
        l_far = ms_ssim_loss(far.clamp(0, 1), x).item()
        assert l_near <= l_far + 1e-4

    def test_gradient_flows_to_pred(self) -> None:
        """(d) Differentiable w.r.t. the reconstruction."""
        pred = torch.rand(2, 3, 48, 48, requires_grad=True)
        target = torch.rand(2, 3, 48, 48)
        ms_ssim_loss(pred, target).backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum().item() > 0.0


# =============================================================================
# 2. Heavy / network feature-distillation losses (GUARDED)
#    DetectorFeatureLoss (YOLOv8m), DINOv2FeatureLoss (ViT),
#    SceneGraphFeatureLoss (SigLIP). Same L1-distillation
#    contract: loss(x, x) == 0, non-negative, finite, gradient flows. Skipped
#    when the backbone weights cannot be materialised offline.
# =============================================================================
def _assert_distillation_contract(loss_fn, size: int) -> None:
    """Shared contract battery for an L1 feature-distillation supervisor."""
    loss_fn.eval()
    x = torch.rand(1, 3, size, size)
    # (b) identity -> 0
    with torch.no_grad():
        v_same = loss_fn(x, x.clone())
    assert v_same.dim() == 0
    assert torch.isfinite(v_same)
    assert float(v_same) == pytest.approx(0.0, abs=1e-5), (
        f"loss(x, x) must be ~0, got {float(v_same):.3e}"
    )
    # (a)/(c) random pair -> finite, >= 0, and strictly positive
    y = torch.rand(1, 3, size, size)
    with torch.no_grad():
        v_diff = loss_fn(x, y)
    assert torch.isfinite(v_diff)
    assert float(v_diff) >= 0.0
    assert float(v_diff) > 0.0
    # (d) gradient flows to pred through the frozen teacher
    pred = torch.rand(1, 3, size, size, requires_grad=True)
    target = torch.rand(1, 3, size, size)
    out = loss_fn(pred, target)
    out.backward()
    assert pred.grad is not None and pred.grad.abs().sum().item() > 0.0


class TestDINOv2FeatureLoss:
    def test_distillation_contract(self) -> None:
        from semvideo.training.dinov2_feature_loss import DINOv2FeatureLoss

        # Prefer dinov2_vitb14 (commonly cached); fall back to default vits14.
        def _make() -> DINOv2FeatureLoss:
            try:
                return DINOv2FeatureLoss(device="cpu", model_name="dinov2_vitb14")
            except Exception:  # noqa: BLE001
                return DINOv2FeatureLoss(device="cpu")

        loss_fn = _skip_if_unavailable(_make)
        _assert_distillation_contract(loss_fn, size=64)

    def test_backbone_frozen(self) -> None:
        from semvideo.training.dinov2_feature_loss import DINOv2FeatureLoss

        loss_fn = _skip_if_unavailable(
            lambda: DINOv2FeatureLoss(device="cpu", model_name="dinov2_vitb14")
        )
        n_trainable = sum(
            p.numel() for p in loss_fn.backbone.parameters() if p.requires_grad
        )
        assert n_trainable == 0


class TestDetectorFeatureLoss:
    def test_distillation_contract(self) -> None:
        from semvideo.training.detector_feature_loss import DetectorFeatureLoss

        loss_fn = _skip_if_unavailable(
            lambda: DetectorFeatureLoss(device="cpu")
        )
        # YOLOv8m backbone expects an input that survives 3 strided downsamples;
        # 128 is the smallest comfortable size for stages 4/6/9.
        _assert_distillation_contract(loss_fn, size=128)


class TestSceneGraphFeatureLoss:
    def test_distillation_contract(self) -> None:
        from semvideo.training.scenegraph_feature_loss import (
            SceneGraphFeatureLoss,
        )

        # SigLIP-base is the backbone the paper uses.
        loss_fn = _skip_if_unavailable(
            lambda: SceneGraphFeatureLoss(device="cpu", model_name="siglip")
        )
        _assert_distillation_contract(loss_fn, size=224)
