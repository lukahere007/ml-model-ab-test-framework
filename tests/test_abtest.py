import numpy as np
import pytest
from scipy import stats

from abtest import (
    cohens_h,
    compare_continuous,
    compare_proportions,
    holm,
    peeking_false_positive_rate,
    power_two_proportions,
    sample_size_two_proportions,
    simulate_ab_test_data,
    srm_check,
)


# --- design -----------------------------------------------------------------

def test_sample_size_matches_r_pwr():
    # R: pwr.2p.test(h = ES.h(0.65, 0.5), power = 0.8)$n == 169.088 -> 170 per group
    assert cohens_h(0.65, 0.5) == pytest.approx(0.3046927, abs=1e-6)
    assert sample_size_two_proportions(0.5, 0.65) == 170


def test_power_matches_r_pwr():
    # R: pwr.2p.test(h = ES.h(0.65, 0.5), n = 25)$power == 0.1898896
    assert power_two_proportions(cohens_h(0.65, 0.5), 25) == pytest.approx(0.1898896, abs=1e-6)


def test_sample_size_reaches_target_power():
    n = sample_size_two_proportions(0.85, 0.90, power=0.9)
    h = cohens_h(0.90, 0.85)
    assert power_two_proportions(h, n) >= 0.9 > power_two_proportions(h, n - 1)


# --- proportions ------------------------------------------------------------

def test_newcombe_interval_matches_published_example():
    # Newcombe (1998), Stat Med 17:873, example (a): 56/70 vs 48/80, method 10.
    r = compare_proportions(48, 80, 56, 70)
    assert r.diff == pytest.approx(0.2)
    assert r.ci_low == pytest.approx(0.0524, abs=1e-4)
    assert r.ci_high == pytest.approx(0.3339, abs=1e-4)


def test_z_test_equals_uncorrected_chi_square():
    table = np.array([[400, 84], [470, 46]])
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    r = compare_proportions(400, 484, 470, 516)
    assert r.z**2 == pytest.approx(chi2)
    assert r.p_value == pytest.approx(p)


# --- continuous -------------------------------------------------------------

def test_continuous_matches_scipy_and_has_sensible_effect_size():
    rng = np.random.default_rng(1)
    a = rng.lognormal(np.log(110), 0.3, 400)
    b = rng.lognormal(np.log(100), 0.3, 400)
    r = compare_continuous(a, b, n_boot=500)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert r.welch_p == pytest.approx(ref.pvalue)
    assert r.mean_ci_low < r.mean_diff < r.mean_ci_high
    assert r.median_ci_low < r.median_diff < r.median_ci_high
    assert r.prob_b_less > 0.5  # B is faster


def test_identical_groups_give_no_effect():
    x = np.arange(1.0, 51.0)
    r = compare_continuous(x, x.copy(), n_boot=200)
    assert r.mean_diff == 0
    assert r.prob_b_less == pytest.approx(0.5)
    assert r.mannwhitney_p == pytest.approx(1.0)


# --- multiple testing, SRM, peeking ----------------------------------------

def test_holm():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_srm_check():
    assert srm_check(500, 500) == pytest.approx(1.0)
    assert srm_check(600, 400) < 0.001


def test_single_look_holds_alpha_and_peeking_inflates_it():
    one = peeking_false_positive_rate(500, n_looks=1, n_sims=4000)
    many = peeking_false_positive_rate(500, n_looks=10, n_sims=4000)
    assert 0.035 < one < 0.065
    assert many > 2 * one


# --- simulation -------------------------------------------------------------

def test_simulation_is_reproducible():
    a = simulate_ab_test_data(200, seed=7)
    b = simulate_ab_test_data(200, seed=7)
    assert a.equals(b)
    assert set(a["model_version"]) == {"A", "B"}
    assert (a["success"] == (a["prediction"] == a["true_label"])).all()
