"""Small, tested helpers for running and analysing an A/B test between two ML models.

Design first (sample size), then analyse once (effect sizes + confidence intervals,
not just p-values), and correct for multiple metrics.
"""

from .design import power_two_proportions, sample_size_two_proportions, cohens_h
from .analysis import (
    ProportionResult,
    ContinuousResult,
    compare_proportions,
    compare_continuous,
    holm,
    srm_check,
)
from .simulate import simulate_ab_test_data, peeking_false_positive_rate

__all__ = [
    "cohens_h",
    "power_two_proportions",
    "sample_size_two_proportions",
    "ProportionResult",
    "ContinuousResult",
    "compare_proportions",
    "compare_continuous",
    "holm",
    "srm_check",
    "simulate_ab_test_data",
    "peeking_false_positive_rate",
]
