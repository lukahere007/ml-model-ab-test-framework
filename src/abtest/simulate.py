"""Synthetic prediction logs and a peeking simulation.

Everything here is simulated. The "true" model accuracies and latencies are inputs,
so an analysis of this data can only show that the method recovers what was put in.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy.stats import norm


def simulate_ab_test_data(
    n: int = 1000,
    accuracy_a: float = 0.85,
    accuracy_b: float = 0.92,
    median_latency_a: float = 110.0,
    median_latency_b: float = 100.0,
    latency_sigma: float = 0.30,
    seed: int = 42,
    start: dt.datetime = dt.datetime(2025, 5, 7, 17, 0, 0),
) -> pd.DataFrame:
    """Simulate ``n`` requests randomly routed 50/50 to model A or B.

    Latency is log-normal (right-skewed, like real service latency) with the given
    medians in milliseconds. The start time is fixed so output is reproducible.
    """
    rng = np.random.default_rng(seed)
    model = rng.choice(np.array(["A", "B"]), size=n)
    is_b = model == "B"
    true_label = rng.integers(0, 2, size=n)
    correct = rng.random(n) < np.where(is_b, accuracy_b, accuracy_a)
    prediction = np.where(correct, true_label, 1 - true_label)
    median = np.where(is_b, median_latency_b, median_latency_a)
    latency = np.round(median * np.exp(rng.normal(0, latency_sigma, size=n)), 1)
    return pd.DataFrame(
        {
            "request_id": np.arange(n),
            "user_id": rng.integers(1000, 10000, size=n),
            "model_version": model,
            "prediction": prediction,
            "true_label": true_label,
            "success": correct,
            "latency_ms": latency,
            "timestamp": pd.date_range(start, periods=n, freq="s"),
        }
    )


def peeking_false_positive_rate(
    n_per_group: int,
    n_looks: int,
    p: float = 0.85,
    alpha: float = 0.05,
    n_sims: int = 4000,
    seed: int = 0,
) -> float:
    """Type I error when an A/A test is checked ``n_looks`` times and stopped at the
    first p < alpha.

    Both arms have the same true rate ``p``, so every "significant" result is a false
    positive. With one look the rate is ~alpha; with repeated looks it inflates.
    """
    if n_looks < 1 or n_looks > n_per_group:
        raise ValueError("n_looks must be between 1 and n_per_group")
    rng = np.random.default_rng(seed)
    a = rng.random((n_sims, n_per_group)) < p
    b = rng.random((n_sims, n_per_group)) < p
    looks = np.linspace(n_per_group / n_looks, n_per_group, n_looks).round().astype(int)
    cum_a = np.cumsum(a, axis=1)[:, looks - 1]
    cum_b = np.cumsum(b, axis=1)[:, looks - 1]
    pooled = (cum_a + cum_b) / (2 * looks)
    se = np.sqrt(pooled * (1 - pooled) * 2 / looks)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (cum_b - cum_a) / looks / se, 0.0)
    rejected = 2 * norm.sf(np.abs(z)) < alpha
    return float(rejected.any(axis=1).mean())
