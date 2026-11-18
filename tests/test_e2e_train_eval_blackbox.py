"""End-to-end black-box tests of the codec with its channel and metric entry points.

These tests treat the public entry points as black boxes and assert observable
input -> output CONTRACTS listed in the project verification rules under
"Verification & Proof-Checking":

* gradient flow: gradients must pass through the channel layer, so the encoder of
  the compact codec receives a non-zero gradient through AWGN and Rayleigh fading;
* evaluation pass: the goal-oriented ``cls_sim`` metric (DINOv2 cosine similarity)
  is exercised through the ``extra_perceptual_metrics`` entry point, which
  fail-softs to ``None`` when the DINOv2 backbone cannot be loaded.

Only PUBLIC APIs are imported. No private attribute is asserted on, no golden
tensor produced by the current implementation is pasted back in, and every
randomised path is seeded so the suite is deterministic on a laptop CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from semvideo.channel.awgn import AWGNChannel
from semvideo.channel.fading import RayleighFadingChannel
from semvideo.models.deep_jscc import DeepJSCC

# Small, CPU-friendly geometry. 64x64 keeps DeepJSCC's 5 stride-2 stages well
# above the 11x11 SSIM-window floor while staying fast on a laptop.
_IMG = 64


def _seed_everything(seed: int = 1234) -> None:
    """Seed Python-less RNGs (torch + numpy) for deterministic CPU tests."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def _tiny_model() -> DeepJSCC:
    """A small DeepJSCC instance (public model API).

    ``base_channels`` is shrunk hard so the tests run in a couple
    of seconds on CPU. The architecture / contract is unchanged - this is just
    the smallest real JSCC autoencoder.
    """
    return DeepJSCC(channel_ratio=1 / 6, base_channels=8)


def _one_batch(n: int = 2, seed: int = 0) -> torch.Tensor:
    """Deterministic synthetic image batch in the documented [0, 1] range."""
    g = torch.Generator().manual_seed(seed)
    # rand -> already in [0, 1]; DeepJSCC's Sigmoid head targets this range.
    return torch.rand(n, 3, _IMG, _IMG, generator=g)


# ---------------------------------------------------------------------------
# Goal-oriented metric: cls_sim (DINOv2). Network / torch.hub guarded.
# ---------------------------------------------------------------------------
class TestClsSimMetricPass:
    """The project's documented goal-oriented metric pass produces ``cls_sim``.

    ``extra_perceptual_metrics`` (semvideo.baselines.codec_under_channel) is the
    public entry point that returns ``{"lpips": ..., "cls_sim": ...}``. The
    ``cls_sim`` branch downloads DINOv2 via torch.hub; on a laptop without that
    cache it fail-softs to ``None`` by design. We therefore assert the CONTRACT
    that holds either way (keys present, types valid) and, when the backbone IS
    available, the documented range property of a cosine similarity.
    """

    def test_extra_perceptual_metrics_contract(self) -> None:
        codec = pytest.importorskip(
            "semvideo.baselines.codec_under_channel",
            reason="baselines.codec_under_channel not importable",
        )
        # Two tiny identical RGB frames in HxWxC uint8 (the documented input).
        rng = np.random.default_rng(0)
        frame = (rng.random((48, 48, 3)) * 255).astype(np.uint8)
        orig = [frame.copy(), frame.copy(), frame.copy()]
        rec = [frame.copy(), frame.copy(), frame.copy()]

        out = codec.extra_perceptual_metrics(orig, rec, device="cpu")

        # Schema-stable contract: both documented keys are always present.
        assert set(out.keys()) == {"lpips", "cls_sim"}
        for k, v in out.items():
            assert v is None or isinstance(v, float), f"{k} must be float | None, got {type(v)}"

        # Range checks only when a backbone actually computed the value.
        if out["lpips"] is not None:
            assert out["lpips"] >= -1e-4, f"LPIPS must be non-negative: {out['lpips']}"
        if out["cls_sim"] is not None:
            # Cosine similarity of two identical frames must be ~1, and always
            # within the closed interval [-1, 1].
            assert -1.0 - 1e-4 <= out["cls_sim"] <= 1.0 + 1e-4, (
                f"cls_sim (cosine similarity) out of [-1, 1]: {out['cls_sim']}"
            )
            assert out["cls_sim"] == pytest.approx(1.0, abs=1e-2), (
                "cls_sim of identical frames must be ~1 "
                f"(got {out['cls_sim']})"
            )


# ---------------------------------------------------------------------------
# Gradient flow through the channel layer (differentiability contract).
# ---------------------------------------------------------------------------
class TestGradientFlowThroughChannel:
    """Gradients must flow through the channel layer (project verification rule).

    The channel injects noise but must remain differentiable so the encoder
    receives gradient. We back-propagate a reconstruction loss through the full
    public forward (model + AWGN channel) and require encoder parameters to
    receive a non-zero gradient.
    """

    @pytest.mark.parametrize("channel_cls", [AWGNChannel, RayleighFadingChannel])
    def test_encoder_receives_gradient(self, channel_cls) -> None:
        _seed_everything(0)
        model = _tiny_model()
        ch = channel_cls(snr_db=10.0)
        x = _one_batch(n=2, seed=6)
        recon = model(x, channel=ch, snr_db=10.0)
        loss = F.mse_loss(recon, x)
        loss.backward()
        got_grad = False
        for _, p in model.encoder.named_parameters():
            if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0:
                got_grad = True
                break
        assert got_grad, "Encoder must receive non-zero gradient through the channel layer"
