"""Tests for quality metrics."""

import torch

from semvideo.metrics.psnr_ssim import psnr


class TestPSNR:
    """Tests for PSNR metric."""

    def test_identical_images_infinite_psnr(self) -> None:
        """PSNR of identical images should be infinity."""
        x = torch.rand(1, 3, 64, 64)
        assert psnr(x, x) == float("inf")

    def test_psnr_decreases_with_noise(self) -> None:
        """More noise should result in lower PSNR."""
        x = torch.rand(1, 3, 64, 64)
        noise_small = x + 0.01 * torch.randn_like(x)
        noise_large = x + 0.1 * torch.randn_like(x)
        assert psnr(noise_small, x) > psnr(noise_large, x)

    def test_known_value(self) -> None:
        """Test against a known PSNR value."""
        x = torch.zeros(1, 1, 2, 2)
        y = torch.ones(1, 1, 2, 2) * 0.1  # MSE = 0.01
        expected_psnr = 20.0  # 10 * log10(1/0.01) = 20
        assert abs(psnr(y, x) - expected_psnr) < 0.01


class TestJammingTableMetricPath:
    """The 8-bit PSNR and token cosine of the jamming tables (semvideo.metrics.dinov2_clssim)."""

    def test_uint8_psnr_known_value(self) -> None:
        """Two 8-bit images 10 levels apart: PSNR = 20 log10(255 / 10)."""
        import math

        import numpy as np

        from semvideo.metrics.dinov2_clssim import psnr as psnr_u8

        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.full((4, 4, 3), 10, dtype=np.uint8)
        assert abs(psnr_u8(a, b) - 20 * math.log10(25.5)) < 1e-4  # float32 arithmetic
        assert psnr_u8(a, a) == float("inf")

    def test_cossim_token_and_token_set(self) -> None:
        """A single token gives its cosine; a token set gives the mean over matching rows."""
        from semvideo.metrics.dinov2_clssim import cossim

        a = torch.tensor([1.0, 0.0])
        assert abs(cossim(a, torch.tensor([0.0, 2.0]))) < 1e-7
        assert abs(cossim(a, torch.tensor([3.0, 0.0])) - 1.0) < 1e-6
        rows_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        rows_b = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        assert abs(cossim(rows_a, rows_b) - 0.5) < 1e-6

    def test_benchmark_sweep_definitions_are_the_shared_ones(self) -> None:
        """eval_benchmark_sweep re-exports the shared definitions rather than its own copies."""
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "scripts" / "eval_benchmark_sweep.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        defined = {
            n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        assert not {"psnr", "cossim", "DINOv2Extractor"} & defined
        imported = {
            alias.name
            for n in tree.body
            if isinstance(n, ast.ImportFrom) and n.module == "semvideo.metrics.dinov2_clssim"
            for alias in n.names
        }
        assert {"psnr", "cossim", "DINOv2Extractor", "DINO_RES"} <= imported


class TestMambaMatchedPairCriterion:
    """The control arm of the MambaJSCC matched pair keeps MambaJSCC's per-sample-sum MSE."""

    def test_psnr_arm_is_per_sample_sum_mse(self) -> None:
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "scripts" / "mamba_matched_pair.py"
        spec = importlib.util.spec_from_file_location("mamba_matched_pair", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # no upstream import at module level

        crit = mod.MatchedCriterion("PSNR", "cpu", yolo_weights="unused.pt")
        torch.manual_seed(0)
        target = torch.rand(3, 3, 8, 8)  # (B, C, H, W)
        recon = target + 0.1 * torch.randn_like(target)
        out = crit(recon, target)
        expected = ((recon - target) ** 2).sum() / target.shape[0]
        assert torch.allclose(out["total"], expected)
        assert set(out) == {"total", "mse"}
