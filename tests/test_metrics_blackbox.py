"""Black-box contract tests for ``semvideo.metrics``.

These tests treat each metric as a black box: they import ONLY the public
API (``psnr``, ``ssim``, ``lpips``/``LPIPSMetric``) and
assert input -> output CONTRACTS and mathematical / domain PROPERTIES that
follow from the docstrings-as-spec and the underlying image-quality math
(Wang et al., 2004 for SSIM; the standard PSNR definition; Zhang et al.,
2018 for LPIPS).  No private attributes, no internal call order, and no
"golden tensor copied from the current implementation" are used.

They also encode the project's verification rules, in particular
"PSNR must increase monotonically with SNR -- flag if
violated" and the reproducibility rule (same seed -> identical metrics).

All tests are deterministic (seeded torch + numpy, small CPU tensors,
tolerances).  The LPIPS path downloads VGG16 weights via torch.hub, so it
is guarded with ``importorskip`` + a network/download skip so the suite
stays green on a laptop without internet.
"""

import math

import numpy as np
import pytest
import torch

from semvideo.metrics.psnr_ssim import psnr, ssim


# en-dash used throughout (never em-dash), per project style.


@pytest.fixture(autouse=True)
def _determinism() -> None:
    """Seed every RNG before each test for reproducibility (project rule)."""
    torch.manual_seed(20260616)
    np.random.seed(20260616)


# ---------------------------------------------------------------------------
# PSNR -- contracts (a) identity, (b) range/monotonicity, (c) symmetry,
# (d) format-scaling invariance.
# ---------------------------------------------------------------------------
class TestPSNRContracts:
    """Contract and property tests for ``psnr``."""

    def test_identity_is_infinite(self) -> None:
        """(a) PSNR of a tensor against itself has zero MSE -> +inf."""
        x = torch.rand(2, 3, 16, 16)
        out = psnr(x, x)
        assert math.isinf(out) and out > 0.0

    def test_finite_for_distinct_inputs(self) -> None:
        """(b) For non-identical inputs PSNR is a finite real number."""
        x = torch.rand(2, 3, 16, 16)
        y = x + 0.05 * torch.randn_like(x)
        out = psnr(y, x)
        assert math.isfinite(out)

    def test_monotone_decreasing_in_noise(self) -> None:
        """(b) PSNR strictly decreases as additive noise grows.

        This is the metric-side analogue of the project rule "PSNR must
        increase monotonically with SNR": more noise (lower SNR) -> lower
        PSNR across a whole sweep, not just two points.
        """
        x = torch.rand(2, 3, 32, 32)
        # Fixed noise direction, scaled by an increasing sequence of stds.
        base_noise = torch.randn_like(x)
        levels = [0.01, 0.05, 0.1, 0.2, 0.4]
        values = [psnr(x + s * base_noise, x) for s in levels]
        # Strictly decreasing along the sweep.
        for lo, hi in zip(values[1:], values[:-1]):
            assert lo < hi

    def test_symmetry(self) -> None:
        """(c) PSNR is symmetric: it depends only on (pred - target) ** 2."""
        x = torch.rand(1, 3, 16, 16)
        y = torch.rand(1, 3, 16, 16)
        assert psnr(x, y) == pytest.approx(psnr(y, x), abs=1e-6)

    def test_known_value_from_definition(self) -> None:
        """(b) Derived from the PSNR definition, not from this code's output.

        Constant offset of 0.1 on [0, 1] data gives MSE = 0.01, so
        PSNR = 10 * log10(1 / 0.01) = 20 dB by definition.
        """
        x = torch.zeros(1, 1, 4, 4)
        y = torch.full((1, 1, 4, 4), 0.1)
        assert psnr(y, x) == pytest.approx(20.0, abs=1e-3)

    def test_format_scaling_invariance(self) -> None:
        """(d) PSNR is invariant to a consistent value-range rescaling.

        The same content expressed in [0, 1] with ``max_val=1`` and in
        [0, 255] with ``max_val=255`` must yield the same dB value: PSNR
        depends only on the relative error max_val**2 / MSE.
        """
        x01 = torch.rand(1, 3, 16, 16)
        y01 = (x01 + 0.05 * torch.randn_like(x01)).clamp(0.0, 1.0)
        p01 = psnr(y01, x01, max_val=1.0)
        p255 = psnr(y01 * 255.0, x01 * 255.0, max_val=255.0)
        assert p255 == pytest.approx(p01, abs=1e-3)

    def test_reproducible_same_seed(self) -> None:
        """Same seed -> identical metric (project reproducibility rule)."""
        torch.manual_seed(7)
        a = torch.rand(1, 3, 16, 16)
        b = a + 0.1 * torch.randn_like(a)
        v1 = psnr(b, a)
        torch.manual_seed(7)
        a2 = torch.rand(1, 3, 16, 16)
        b2 = a2 + 0.1 * torch.randn_like(a2)
        v2 = psnr(b2, a2)
        assert v1 == pytest.approx(v2, abs=1e-9)


# ---------------------------------------------------------------------------
# SSIM -- contracts (a) identity, (b) range, (c) symmetry, plus the
# monotone-degradation property.
# ---------------------------------------------------------------------------
class TestSSIMContracts:
    """Contract and property tests for ``ssim``."""

    def test_identity_near_one(self) -> None:
        """(a) SSIM of an image against itself is ~= 1.0."""
        x = torch.rand(1, 3, 32, 32)
        assert ssim(x, x) == pytest.approx(1.0, abs=1e-4)

    def test_constant_image_identity_exactly_one(self) -> None:
        """(a) For a flat image (zero variance) self-SSIM is essentially 1."""
        x = torch.full((1, 1, 16, 16), 0.5)
        assert ssim(x, x) == pytest.approx(1.0, abs=1e-5)

    def test_range_within_bounds(self) -> None:
        """(b) SSIM lies in [-1, 1] for arbitrary in-range inputs."""
        x = torch.rand(1, 3, 32, 32)
        y = torch.rand(1, 3, 32, 32)  # unrelated content
        val = ssim(x, y)
        assert -1.0 - 1e-5 <= val <= 1.0 + 1e-5

    def test_self_geq_cross(self) -> None:
        """(b) Self-similarity is at least as large as cross-similarity.

        SSIM is maximised (= 1) for identical inputs, so ssim(x, x) must
        dominate ssim(x, y) for unrelated y.
        """
        x = torch.rand(1, 3, 32, 32)
        y = torch.rand(1, 3, 32, 32)
        assert ssim(x, x) >= ssim(x, y)

    def test_symmetry(self) -> None:
        """(c) SSIM is symmetric under swapping pred and target."""
        x = torch.rand(1, 3, 32, 32)
        y = torch.rand(1, 3, 32, 32)
        assert ssim(x, y) == pytest.approx(ssim(y, x), abs=1e-6)

    def test_monotone_degradation(self) -> None:
        """(b) SSIM decreases as additive noise grows.

        A widely separated two-level comparison (small vs large noise) is a
        robust degradation contract that follows from the SSIM construction.
        """
        x = torch.rand(1, 3, 48, 48)
        noise = torch.randn_like(x)
        s_small = ssim((x + 0.02 * noise).clamp(0, 1), x)
        s_large = ssim((x + 0.25 * noise).clamp(0, 1), x)
        assert s_small > s_large

    def test_reproducible_same_seed(self) -> None:
        """Same seed -> identical SSIM (reproducibility rule)."""
        torch.manual_seed(11)
        a = torch.rand(1, 3, 32, 32)
        b = (a + 0.1 * torch.randn_like(a)).clamp(0, 1)
        v1 = ssim(b, a)
        torch.manual_seed(11)
        a2 = torch.rand(1, 3, 32, 32)
        b2 = (a2 + 0.1 * torch.randn_like(a2)).clamp(0, 1)
        v2 = ssim(b2, a2)
        assert v1 == pytest.approx(v2, abs=1e-9)


# ---------------------------------------------------------------------------
# LPIPS -- guarded (torchvision + VGG16 weight download).  Contracts:
# identity ~= 0, non-negativity, symmetry.
# ---------------------------------------------------------------------------
def _make_lpips_model():
    """Build an LPIPSMetric on CPU, skipping if torchvision/weights absent.

    The VGG16 backbone is fetched via torch.hub on first use; on a laptop
    without network access this raises.  We turn any such failure into a
    skip so the black-box suite stays green offline (per the task's
    "guard network-download paths" rule).
    """
    pytest.importorskip("torchvision", reason="LPIPS needs torchvision")
    from semvideo.metrics.lpips_fid import LPIPSMetric

    try:
        model = LPIPSMetric(device="cpu").to("cpu").eval()
    except Exception as exc:  # noqa: BLE001 -- download/availability guard
        pytest.skip(f"VGG16 weights unavailable (offline?): {exc}")
    return model


class TestLPIPSContracts:
    """Contract and property tests for the perceptual LPIPS distance."""

    def test_identity_near_zero(self) -> None:
        """(a) LPIPS distance of an image to itself is ~= 0 (>= 0)."""
        model = _make_lpips_model()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            d = model(x, x).item()
        assert d == pytest.approx(0.0, abs=1e-4)
        assert d >= -1e-6

    def test_non_negative(self) -> None:
        """(b) LPIPS is a distance: it is non-negative for any inputs."""
        model = _make_lpips_model()
        x = torch.rand(1, 3, 64, 64)
        y = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            d = model(x, y).item()
        assert d >= -1e-6

    def test_symmetry(self) -> None:
        """(c) The feature-space L2 distance is symmetric in its arguments."""
        model = _make_lpips_model()
        x = torch.rand(1, 3, 64, 64)
        y = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            d_xy = model(x, y).item()
            d_yx = model(y, x).item()
        assert d_xy == pytest.approx(d_yx, abs=1e-6)

    def test_distinct_greater_than_identity(self) -> None:
        """(b) Distance to a different image exceeds the self-distance."""
        model = _make_lpips_model()
        x = torch.rand(1, 3, 64, 64)
        y = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            d_self = model(x, x).item()
            d_cross = model(x, y).item()
        assert d_cross > d_self

    def test_convenience_function_matches_module(self) -> None:
        """The ``lpips`` helper reproduces the module's forward value."""
        model = _make_lpips_model()
        from semvideo.metrics.lpips_fid import lpips

        x = torch.rand(1, 3, 64, 64)
        y = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            d_module = model(x, y).item()
        d_fn = lpips(x, y, model=model)
        assert d_fn == pytest.approx(d_module, abs=1e-6)
