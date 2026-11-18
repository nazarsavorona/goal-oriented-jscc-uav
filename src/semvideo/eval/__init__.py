"""Statistical inference utilities for paired model evaluation.

This package provides paired statistical tests used to attach honest
uncertainty estimates (confidence intervals and p-values) to the
load-bearing ``cls_sim`` comparisons reported in the paper. All estimators
operate on *paired* per-image samples: the two
models being compared must be evaluated on the same images so that the
per-image difference is well defined.
"""

from semvideo.eval.stats import (
    PairedComparison,
    mean_diff_ci,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)

__all__ = [
    "PairedComparison",
    "mean_diff_ci",
    "paired_bootstrap_ci",
    "wilcoxon_signed_rank",
]
