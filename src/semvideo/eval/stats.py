"""Paired statistical inference for model comparisons.

This module implements the three estimators used to decide whether a
difference in mean per-image ``cls_sim`` (or any per-image scalar metric)
between two models is statistically distinguishable:

* :func:`paired_bootstrap_ci` - non-parametric percentile bootstrap
  confidence interval of the paired mean difference. Makes no normality
  assumption and is the primary estimator reported.
* :func:`wilcoxon_signed_rank` - Wilcoxon signed-rank test on the paired
  differences (two-sided), a distribution-free significance test.
* :func:`mean_diff_ci` - parametric paired-t (Student) confidence
  interval, reported as a sanity cross-check against the bootstrap.

All functions take two equal-length 1-D arrays ``a`` and ``b`` whose
i-th entries are the metric for the *same* image under model A and model
B respectively. The reported difference is ``A - B`` throughout, so a
positive mean difference means model A scores higher.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PairedComparison:
    """Result of a paired statistical comparison of two models.

    Attributes:
        n: Number of paired samples (images) used.
        mean_a: Mean of model A's per-image metric.
        mean_b: Mean of model B's per-image metric.
        mean_diff: Mean paired difference ``mean(a - b)``.
        ci_low: Lower bound of the 95 percent (by default) CI on the mean
            difference, from the percentile bootstrap.
        ci_high: Upper bound of the bootstrap CI on the mean difference.
        ci_low_t: Lower bound of the parametric paired-t CI (cross-check).
        ci_high_t: Upper bound of the parametric paired-t CI.
        wilcoxon_stat: Wilcoxon signed-rank test statistic.
        wilcoxon_p: Two-sided p-value from the Wilcoxon signed-rank test.
        conf_level: Confidence level used for both CIs (e.g. 0.95).
        distinguishable: True iff the two models are distinguishable at
            ``alpha = 1 - conf_level``: the bootstrap CI excludes 0 AND
            the Wilcoxon p-value is below ``alpha``.
    """

    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    ci_low_t: float
    ci_high_t: float
    wilcoxon_stat: float
    wilcoxon_p: float
    conf_level: float
    distinguishable: bool


def _validate_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce a pair of metric arrays.

    Args:
        a: Per-image metric for model A.
        b: Per-image metric for model B (same images, same order).

    Returns:
        Tuple of float64 1-D arrays ``(a, b)``.

    Raises:
        ValueError: If shapes mismatch, arrays are empty, or any entry is
            non-finite.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have equal length, got {a.shape} and {b.shape}")
    if a.size == 0:
        raise ValueError("paired arrays must be non-empty")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("paired arrays contain non-finite values")
    return a, b


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    conf_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the paired mean difference ``A - B``.

    Resamples image indices with replacement ``n_boot`` times, recomputes
    the mean paired difference on each resample, and returns the empirical
    percentile interval. Pairing is preserved because the same resampled
    index is used for both arrays.

    Args:
        a: Per-image metric for model A.
        b: Per-image metric for model B (same images, same order).
        n_boot: Number of bootstrap resamples.
        conf_level: Two-sided confidence level (e.g. 0.95 -> 2.5/97.5 pct).
        seed: RNG seed for reproducibility.

    Returns:
        Tuple ``(mean_diff, ci_low, ci_high)``.
    """
    a, b = _validate_pair(a, b)
    diff = a - b
    n = diff.size
    rng = np.random.default_rng(seed)
    # (n_boot, n) resampled indices -> vectorised bootstrap means.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)  # (n_boot,)
    alpha = 1.0 - conf_level
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(diff.mean()), float(lo), float(hi)


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Distribution-free test of the null hypothesis that the median of the
    paired differences ``A - B`` is zero. Zero differences are dropped via
    the ``wilcox`` convention; ties are handled by SciPy. If every
    difference is zero (identical models) the test is undefined and a
    p-value of 1.0 is returned.

    Args:
        a: Per-image metric for model A.
        b: Per-image metric for model B (same images, same order).

    Returns:
        Tuple ``(statistic, p_value)``. ``statistic`` is NaN when the test
        is undefined (all-zero differences).
    """
    a, b = _validate_pair(a, b)
    diff = a - b
    if np.allclose(diff, 0.0):
        return float("nan"), 1.0
    # zero_method="wilcox" drops exact zeros; "auto" picks exact vs normal.
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", mode="auto")
    return float(res.statistic), float(res.pvalue)


def mean_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    conf_level: float = 0.95,
) -> tuple[float, float, float]:
    """Parametric paired-t CI of the mean difference ``A - B``.

    Reported as a cross-check against :func:`paired_bootstrap_ci`. Uses the
    Student-t critical value with ``n - 1`` degrees of freedom on the
    standard error of the paired differences. For a single sample
    (``n == 1``) the interval is degenerate and returned as
    ``(diff, diff, diff)``.

    Args:
        a: Per-image metric for model A.
        b: Per-image metric for model B (same images, same order).
        conf_level: Two-sided confidence level (e.g. 0.95).

    Returns:
        Tuple ``(mean_diff, ci_low, ci_high)``.
    """
    a, b = _validate_pair(a, b)
    diff = a - b
    n = diff.size
    mean = float(diff.mean())
    if n == 1:
        return mean, mean, mean
    se = float(diff.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(1.0 - (1.0 - conf_level) / 2.0, df=n - 1))
    half = tcrit * se
    return mean, mean - half, mean + half


def compare_paired(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    conf_level: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Run all three paired estimators and bundle the verdict.

    Args:
        a: Per-image metric for model A.
        b: Per-image metric for model B (same images, same order).
        n_boot: Bootstrap resamples for the percentile CI.
        conf_level: Two-sided confidence level for both CIs.
        seed: RNG seed for the bootstrap.

    Returns:
        A populated :class:`PairedComparison`. ``distinguishable`` is True
        iff the bootstrap CI excludes 0 and the Wilcoxon p-value is below
        ``alpha = 1 - conf_level``.
    """
    a, b = _validate_pair(a, b)
    mean_diff, ci_low, ci_high = paired_bootstrap_ci(
        a, b, n_boot=n_boot, conf_level=conf_level, seed=seed
    )
    _, ci_low_t, ci_high_t = mean_diff_ci(a, b, conf_level=conf_level)
    w_stat, w_p = wilcoxon_signed_rank(a, b)
    alpha = 1.0 - conf_level
    ci_excludes_zero = (ci_low > 0.0) or (ci_high < 0.0)
    distinguishable = bool(ci_excludes_zero and (w_p < alpha))
    return PairedComparison(
        n=int(a.size),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_low_t=ci_low_t,
        ci_high_t=ci_high_t,
        wilcoxon_stat=w_stat,
        wilcoxon_p=w_p,
        conf_level=conf_level,
        distinguishable=distinguishable,
    )
