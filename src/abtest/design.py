"""Experiment design: effect size and sample size, computed *before* any data is collected.

The formulas follow Cohen (1988) and match R's ``pwr::pwr.2p.test`` (see the companion
repo https://github.com/lukahere007/a-b-test-power).
"""

from __future__ import annotations

import math

from scipy.stats import norm


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions: 2*asin(sqrt(p1)) - 2*asin(sqrt(p2))."""
    for p in (p1, p2):
        if not 0 <= p <= 1:
            raise ValueError("proportions must be in [0, 1]")
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def power_two_proportions(h: float, n_per_group: float, alpha: float = 0.05) -> float:
    """Power of a two-sided two-proportion test for effect size ``h`` and equal groups.

    ``h`` is the effect you *assume* when planning. Plugging in the effect observed in
    your own data ("observed" or "post-hoc" power) is uninformative: it is a monotone
    function of the p-value and says nothing new.
    """
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive")
    z_crit = norm.ppf(1 - alpha / 2)
    shift = abs(h) * math.sqrt(n_per_group / 2)
    return float(norm.cdf(shift - z_crit) + norm.cdf(-shift - z_crit))


def sample_size_two_proportions(
    p_control: float, p_treatment: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Observations needed *per group* to detect ``p_control -> p_treatment``.

    Uses the arcsine (Cohen's h) approximation, rounded up to a whole observation.
    """
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    h = cohens_h(p_treatment, p_control)
    if h == 0:
        raise ValueError("p_control and p_treatment must differ")
    z_sum = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    n = 2 * (z_sum / abs(h)) ** 2
    # The closed form ignores the (negligible) opposite tail; nudge up until it holds.
    n_int = math.ceil(n)
    while power_two_proportions(h, n_int, alpha) < power:
        n_int += 1
    return n_int
