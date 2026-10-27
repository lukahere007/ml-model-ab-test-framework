"""Analysis of a finished A/B test: effect sizes with confidence intervals, then p-values."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .design import cohens_h


@dataclass(frozen=True)
class ProportionResult:
    rate_a: float
    rate_b: float
    diff: float  # rate_b - rate_a
    ci_low: float
    ci_high: float
    cohens_h: float  # h(rate_b, rate_a)
    z: float
    p_value: float


@dataclass(frozen=True)
class ContinuousResult:
    mean_a: float
    mean_b: float
    mean_diff: float  # mean_b - mean_a
    mean_ci_low: float
    mean_ci_high: float
    median_a: float
    median_b: float
    median_diff: float  # median_b - median_a
    median_ci_low: float
    median_ci_high: float
    prob_b_less: float  # P(random B value < random A value), ties count 1/2
    welch_t: float
    welch_p: float
    mannwhitney_p: float


def _wilson(successes: int, n: int, z: float) -> tuple[float, float]:
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def compare_proportions(
    successes_a: int, n_a: int, successes_b: int, n_b: int, alpha: float = 0.05
) -> ProportionResult:
    """Two-proportion comparison (e.g. accuracy of model B vs model A).

    * Difference ``B - A`` with a Newcombe (hybrid Wilson score) confidence interval,
      which has better coverage than the textbook Wald interval near 0 or 1.
    * Two-sided pooled z-test. Its z**2 equals the Pearson chi-square statistic of the
      2x2 table without continuity correction, so this *is* the chi-square test.
    * Cohen's h as a scale-free effect size.
    """
    if min(n_a, n_b) <= 0:
        raise ValueError("group sizes must be positive")
    if not (0 <= successes_a <= n_a and 0 <= successes_b <= n_b):
        raise ValueError("successes must be between 0 and n")
    z_crit = stats.norm.ppf(1 - alpha / 2)
    pa, pb = successes_a / n_a, successes_b / n_b
    diff = pb - pa

    la, ua = _wilson(successes_a, n_a, z_crit)
    lb, ub = _wilson(successes_b, n_b, z_crit)
    ci_low = diff - math.sqrt((pb - lb) ** 2 + (ua - pa) ** 2)
    ci_high = diff + math.sqrt((ub - pb) ** 2 + (pa - la) ** 2)

    pooled = (successes_a + successes_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = diff / se if se > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))
    return ProportionResult(pa, pb, diff, ci_low, ci_high, cohens_h(pb, pa), z, p_value)


def compare_continuous(
    a, b, alpha: float = 0.05, n_boot: int = 5000, seed: int | None = 0
) -> ContinuousResult:
    """Compare a continuous, possibly skewed metric such as latency.

    * Mean difference ``B - A`` with a Welch confidence interval and Welch t-test
      (does not assume equal variances).
    * Median difference with a percentile bootstrap CI. For skewed latency the
      median (or a high percentile) is usually what users feel.
    * Mann-Whitney U p-value and the matching effect size P(B < A), the
      "probability of superiority" (0.5 = no difference).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("need at least two observations per group")

    welch = stats.ttest_ind(b, a, equal_var=False)
    ci = welch.confidence_interval(confidence_level=1 - alpha)

    rng = np.random.default_rng(seed)
    boot_a = np.median(rng.choice(a, size=(n_boot, len(a)), replace=True), axis=1)
    boot_b = np.median(rng.choice(b, size=(n_boot, len(b)), replace=True), axis=1)
    med_lo, med_hi = np.quantile(boot_b - boot_a, [alpha / 2, 1 - alpha / 2])

    mw = stats.mannwhitneyu(b, a, alternative="two-sided")
    # U counts pairs with b > a (+ half ties), so P(b < a) = 1 - U / (n_a * n_b).
    prob_b_less = 1 - mw.statistic / (len(a) * len(b))

    return ContinuousResult(
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=float(b.mean() - a.mean()),
        mean_ci_low=float(ci.low),
        mean_ci_high=float(ci.high),
        median_a=float(np.median(a)),
        median_b=float(np.median(b)),
        median_diff=float(np.median(b) - np.median(a)),
        median_ci_low=float(med_lo),
        median_ci_high=float(med_hi),
        prob_b_less=float(prob_b_less),
        welch_t=float(welch.statistic),
        welch_p=float(welch.pvalue),
        mannwhitney_p=float(mw.pvalue),
    )


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def srm_check(n_a: int, n_b: int, expected_share_a: float = 0.5) -> float:
    """Sample-ratio-mismatch check: chi-square goodness-of-fit p-value for the split.

    A tiny p-value (teams often use < 0.001) means the randomisation or logging is
    broken and the test results should not be trusted.
    """
    total = n_a + n_b
    expected = [total * expected_share_a, total * (1 - expected_share_a)]
    return float(stats.chisquare([n_a, n_b], f_exp=expected).pvalue)
