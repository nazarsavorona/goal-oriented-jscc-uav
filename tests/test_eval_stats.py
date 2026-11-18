"""Unit tests for semvideo.eval.stats paired inference."""

from __future__ import annotations

import numpy as np
import pytest

from semvideo.eval.stats import (
    compare_paired,
    mean_diff_ci,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)


def test_bootstrap_ci_brackets_known_shift() -> None:
    """A constant +0.05 shift must give a CI tightly around +0.05."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.9, 0.02, 500)
    b = a - 0.05  # deterministic paired shift
    mean_diff, lo, hi = paired_bootstrap_ci(a, b, n_boot=5000, seed=1)
    assert mean_diff == pytest.approx(0.05, abs=1e-9)
    assert lo == pytest.approx(0.05, abs=1e-9)
    assert hi == pytest.approx(0.05, abs=1e-9)  # zero variance in diff -> degenerate CI


def test_bootstrap_and_t_ci_agree() -> None:
    """Bootstrap and parametric paired-t CIs should be close on Gaussian data."""
    rng = np.random.default_rng(2)
    a = rng.normal(0.9, 0.02, 400)
    b = a - 0.03 + rng.normal(0.0, 0.01, 400)
    _, blo, bhi = paired_bootstrap_ci(a, b, n_boot=10000, seed=3)
    _, tlo, thi = mean_diff_ci(a, b)
    assert blo == pytest.approx(tlo, abs=2e-3)
    assert bhi == pytest.approx(thi, abs=2e-3)


def test_wilcoxon_identical_is_undefined() -> None:
    """Identical arrays -> p=1.0 and NaN statistic (test undefined)."""
    a = np.linspace(0.8, 0.95, 50)
    stat, p = wilcoxon_signed_rank(a, a)
    assert np.isnan(stat)
    assert p == 1.0


def test_wilcoxon_detects_shift() -> None:
    """A clear shift must be highly significant."""
    rng = np.random.default_rng(4)
    a = rng.normal(0.9, 0.02, 200)
    b = a - 0.02 + rng.normal(0.0, 0.003, 200)
    _, p = wilcoxon_signed_rank(a, b)
    assert p < 1e-6


def test_distinguishable_verdict_true_and_false() -> None:
    """compare_paired verdict: True for a real shift, False for noise-only."""
    rng = np.random.default_rng(5)
    a = rng.normal(0.9, 0.02, 300)
    b_real = a - 0.02 + rng.normal(0.0, 0.003, 300)
    b_noise = a + rng.normal(0.0, 0.02, 300)  # zero-mean perturbation
    assert compare_paired(a, b_real, n_boot=4000, seed=6).distinguishable is True
    assert compare_paired(a, b_noise, n_boot=4000, seed=6).distinguishable is False


def test_unequal_length_raises() -> None:
    """Mismatched lengths must raise (alignment is the caller's job)."""
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.zeros(5), np.zeros(4))


def test_single_sample_t_ci_degenerate() -> None:
    """n=1 paired-t CI collapses to the point estimate."""
    m, lo, hi = mean_diff_ci(np.array([0.9]), np.array([0.8]))
    assert (m, lo, hi) == pytest.approx((0.1, 0.1, 0.1))
