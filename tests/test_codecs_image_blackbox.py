"""Black-box contract tests for the image JSCC codec.

Treats the codec in :mod:`semvideo.models` as a black box: only the public
API (constructor + ``forward`` + the documented ``encode`` / ``decode``) is
touched, and only observable input -> output contracts and domain properties
are asserted. No private attribute, no internal call order, and no golden
tensor copied out of the current implementation is used as an oracle.

Codec under test (an end-to-end image JSCC reconstructor whose
``forward`` maps an image batch through an optional differentiable channel
back to an image of the same shape):

  - :class:`semvideo.models.deep_jscc.DeepJSCC`

Contracts asserted (derived from each docstring-as-spec, the JSCC domain
math, and the project verification rules):

  (a) round-trip output shape == input image shape;
  (b) outputs are finite and -- because the decoder ends in ``Sigmoid`` and
      the docstring promises reconstructions in [0, 1] -- lie within [0, 1];
  (c) the model is differentiable end-to-end, with gradient flowing through
      the channel layer (project rule: "gradients must flow through channel
      layers");
  (d) determinism under a fixed seed (project rule: "same seed must produce
      identical metrics");
  (e) per-sample independence: feeding a batch yields, per image, the same
      result as feeding that image alone (no cross-sample leakage). This is
      checked on the noiseless path so that channel-noise random sampling does
      not confound the comparison.

All tests run on CPU with tiny configs, small random images, and seeded RNG so
the suite stays green on a laptop with no network access.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from semvideo.channel.awgn import AWGNChannel
from semvideo.models.deep_jscc import DeepJSCC

# --------------------------------------------------------------------------- #
# Tiny-config black-box builders.
#
# Each builder returns (model, image_size). Images are small and channel
# widths are minimal so everything runs fast on CPU. The image size
# is encoded alongside the model because the codec has a divisibility
# constraint (16x for the CNN codec).
# We feed only the documented public constructor arguments.
# --------------------------------------------------------------------------- #
IMG_HW = 32  # divisible by 16 (CNN codec)


def _build_deep_jscc() -> tuple[nn.Module, int]:
    # channel_ratio=1/16 -> bottleneck width max(1, int(256/16)) = 16.
    return DeepJSCC(channel_ratio=1 / 16, base_channels=4), IMG_HW


# name -> builder. The name is also the pytest test id.
CODEC_BUILDERS = {
    "DeepJSCC": _build_deep_jscc,
}


def _seed(seed: int = 0) -> None:
    """Seed both RNGs the codec / channel may draw from."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def _rand_images(batch: int, hw: int, *, seed: int = 0) -> torch.Tensor:
    """Small CPU image batch in [0, 1), the documented input domain."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 3, hw, hw, generator=g)


def _build(name: str) -> tuple[nn.Module, int]:
    _seed(0)  # deterministic parameter init
    model, hw = CODEC_BUILDERS[name]()
    model.eval()  # disable dropout / training-only randomness for contract checks
    return model, hw


# --------------------------------------------------------------------------- #
# (a) Round-trip output shape == input image shape.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(CODEC_BUILDERS))
def test_roundtrip_shape_matches_input(name: str) -> None:
    """forward(image) must return an image of identical shape (B, 3, H, W)."""
    model, hw = _build(name)
    x = _rand_images(batch=2, hw=hw, seed=1)
    channel = AWGNChannel(snr_db=10.0)
    _seed(7)
    with torch.no_grad():
        y = model(x, channel=channel)
    assert y.shape == x.shape, f"{name}: {tuple(y.shape)} != {tuple(x.shape)}"


@pytest.mark.parametrize("name", list(CODEC_BUILDERS))
def test_roundtrip_shape_noiseless(name: str) -> None:
    """Shape contract must hold on the noiseless path (channel=None) too."""
    model, hw = _build(name)
    x = _rand_images(batch=2, hw=hw, seed=2)
    with torch.no_grad():
        y = model(x, channel=None)
    assert y.shape == x.shape


# --------------------------------------------------------------------------- #
# (b) Outputs finite and in [0, 1] (the decoder ends in Sigmoid; the docstring
#     promises reconstructions in [0, 1]).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(CODEC_BUILDERS))
def test_output_finite_and_in_unit_range(name: str) -> None:
    """Reconstruction is finite and within [0, 1] under a realistic channel."""
    model, hw = _build(name)
    x = _rand_images(batch=2, hw=hw, seed=3)
    channel = AWGNChannel(snr_db=10.0)
    _seed(11)
    with torch.no_grad():
        y = model(x, channel=channel)
    assert torch.isfinite(y).all(), f"{name}: non-finite reconstruction"
    # Sigmoid output is mathematically in (0, 1); allow a tiny numerical slack.
    assert y.min() >= -1e-5, f"{name}: min {y.min().item()} below 0"
    assert y.max() <= 1.0 + 1e-5, f"{name}: max {y.max().item()} above 1"


@pytest.mark.parametrize("name", list(CODEC_BUILDERS))
def test_output_finite_low_snr(name: str) -> None:
    """Even at a harsh SNR the codec must not emit NaN / Inf."""
    model, hw = _build(name)
    x = _rand_images(batch=2, hw=hw, seed=4)
    channel = AWGNChannel(snr_db=-5.0)
    _seed(13)
    with torch.no_grad():
        y = model(x, channel=channel)
    assert torch.isfinite(y).all(), f"{name}: non-finite at low SNR"
    assert y.min() >= -1e-5 and y.max() <= 1.0 + 1e-5


# --------------------------------------------------------------------------- #
# (c) Differentiable end-to-end, including through the channel layer.
#     Project rule: "gradients must flow through channel layers".
# --------------------------------------------------------------------------- #
class _GradProbeChannel(nn.Module):
    """AWGN channel wrapped with a learnable scalar gain on the symbols.

    Black-box probe: the channel keeps the AWGN noise behaviour but multiplies
    the symbols by a trainable parameter. If, after a backward pass, that
    parameter has a finite non-None gradient, then the autograd path provably
    runs *through* the channel module -- the project's "gradients must flow
    through channel layers" requirement.
    """

    def __init__(self, snr_db: float = 10.0) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.awgn = AWGNChannel(snr_db=snr_db)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        # Accept and ignore any extra kwargs (e.g. snr_db) a codec may pass.
        return self.awgn(x * self.gain)


@pytest.mark.parametrize("name", list(CODEC_BUILDERS))
def test_differentiable_through_channel(name: str) -> None:
    """Gradient flows to the input and through a parameter inside the channel."""
    model, hw = _build(name)
    model.train()  # gradients are a training-time property
    x = _rand_images(batch=2, hw=hw, seed=5).requires_grad_(True)
    channel = _GradProbeChannel(snr_db=10.0)
    _seed(17)
    y = model(x, channel=channel)
    loss = y.mean()
    loss.backward()

    # Gradient reaches the input image.
    assert x.grad is not None, f"{name}: no gradient on input"
    assert torch.isfinite(x.grad).all(), f"{name}: non-finite input grad"
    assert x.grad.abs().sum() > 0, f"{name}: zero input gradient"

    # Gradient reaches a parameter that lives *inside* the channel module,
    # proving the autograd graph passes through the channel.
    assert channel.gain.grad is not None, f"{name}: no grad through channel"
    assert torch.isfinite(channel.gain.grad).all()

    # At least one model parameter also receives gradient (encoder/decoder
    # are trained jointly -- "joint source-channel coding").
    model_grads = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert model_grads, f"{name}: no model parameter received gradient"
    assert any(g.abs().sum() > 0 for g in model_grads)


# --------------------------------------------------------------------------- #
# (d) Determinism under a fixed seed.
#     Project rule: "same seed must produce identical metrics".
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(CODEC_BUILDERS))
def test_determinism_fixed_seed_with_channel(name: str) -> None:
    """Re-seeding before forward reproduces the (stochastic-channel) output."""
    model, hw = _build(name)
    x = _rand_images(batch=2, hw=hw, seed=6)
    channel = AWGNChannel(snr_db=5.0)
    with torch.no_grad():
        _seed(99)
        y1 = model(x, channel=channel)
        _seed(99)
        y2 = model(x, channel=channel)
    assert torch.equal(y1, y2), (
        f"{name}: identical seed gave different outputs "
        f"(max diff {(y1 - y2).abs().max().item()})"
    )


@pytest.mark.parametrize("name", list(CODEC_BUILDERS))
def test_channel_noise_is_actually_random(name: str) -> None:
    """Sanity dual of (d): without re-seeding, channel noise must vary.

    This guards against a vacuous determinism test -- if the channel were a
    no-op the determinism check above would pass trivially. The AWGN spec adds
    a fresh ``randn_like`` draw per call, so two un-reseeded passes must differ.
    """
    model, hw = _build(name)
    x = _rand_images(batch=2, hw=hw, seed=6)
    channel = AWGNChannel(snr_db=5.0)
    with torch.no_grad():
        _seed(99)
        y1 = model(x, channel=channel)
        # no re-seed: the next noise draw is independent
        y2 = model(x, channel=channel)
    assert not torch.equal(y1, y2), f"{name}: channel appears deterministic / no-op"


# --------------------------------------------------------------------------- #
# (e) Per-sample independence (no cross-sample leakage).
#     Checked on the noiseless path so per-call channel-noise sampling does not
#     confound the per-image comparison.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(CODEC_BUILDERS))
def test_per_sample_independence_noiseless(name: str) -> None:
    """Processing a batch == processing each image alone (no leakage)."""
    model, hw = _build(name)
    x = _rand_images(batch=3, hw=hw, seed=8)
    with torch.no_grad():
        y_batch = model(x, channel=None)
        y_each = torch.cat(
            [model(x[i : i + 1], channel=None) for i in range(x.shape[0])],
            dim=0,
        )
    max_diff = (y_batch - y_each).abs().max().item()
    # Only floating-point reduction-order differences are tolerated.
    assert max_diff < 1e-4, (
        f"{name}: per-sample result differs from batched by {max_diff} "
        "-> cross-sample leakage"
    )


# --------------------------------------------------------------------------- #
# Cross-checking the encode/decode public sub-API against forward.
#
# The codec documents encode()/decode() as the building blocks of forward().
# On the noiseless path, forward(x, channel=None) must equal
# decode(encode(x)). This asserts the documented decomposition contract
# without peeking at internals.
# --------------------------------------------------------------------------- #
def _encode_decode_roundtrip(name: str, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Drive the codec's public encode/decode with its documented signature."""
    assert name == "DeepJSCC"
    # Documented: decode needs the encoded spatial (C, H/16, W/16) shape.
    c = max(1, int(256 * model.channel_ratio))
    spatial = (c, x.shape[-2] // 16, x.shape[-1] // 16)
    return model.decode(model.encode(x), spatial)


@pytest.mark.parametrize("name", list(CODEC_BUILDERS))
def test_forward_equals_encode_then_decode_noiseless(name: str) -> None:
    """forward(x, channel=None) == decode(encode(x)) on the noiseless path."""
    model, hw = _build(name)
    x = _rand_images(batch=2, hw=hw, seed=9)
    with torch.no_grad():
        y_forward = model(x, channel=None)
        y_split = _encode_decode_roundtrip(name, model, x)
    assert y_split.shape == y_forward.shape
    max_diff = (y_forward - y_split).abs().max().item()
    assert max_diff < 1e-4, (
        f"{name}: forward != decode(encode(...)) by {max_diff}; the documented "
        "encode/decode decomposition does not match forward"
    )
