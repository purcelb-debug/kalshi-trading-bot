import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from predlab.evaluation.metrics import (
    bootstrap_ci,
    brier,
    ece,
    log_loss,
    murphy_decomposition,
    paired_bootstrap_diff,
    reliability_table,
    skill_score,
)


def test_brier_hand_calculation():
    # ((0.8-1)^2 + (0.3-0)^2) / 2 = (0.04 + 0.09) / 2
    assert brier([0.8, 0.3], [1, 0]) == pytest.approx(0.065)


def test_log_loss_hand_calculation():
    expected = -(math.log(0.8) + math.log(0.7)) / 2
    assert log_loss([0.8, 0.3], [1, 0]) == pytest.approx(expected)


def test_log_loss_is_finite_for_certain_misses():
    assert math.isfinite(log_loss([1.0, 0.0], [0, 1]))


def test_ece_hand_calculation():
    # bin [0.2,0.3): p=0.2,0.2 obs=0.5 -> |0.2-0.5|*2 ; bin [0.9,1.0]: p=0.9,0.9 obs=1 -> 0.1*2
    assert ece([0.2, 0.2, 0.9, 0.9], [1, 0, 1, 1]) == pytest.approx((0.3 * 2 + 0.1 * 2) / 4)


def test_reliability_table_counts():
    rows = reliability_table([0.05, 0.15, 0.95], [0, 1, 1], n_bins=10)
    assert [(r["bin"], r["n"]) for r in rows] == [(0, 1), (1, 1), (9, 1)]


def test_murphy_identity_exact_when_forecasts_constant_within_bins():
    p = np.array([0.15] * 4 + [0.75] * 4)
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    d = murphy_decomposition(p, y)
    assert d["within_bin"] == pytest.approx(0.0, abs=1e-12)
    assert d["reliability"] - d["resolution"] + d["uncertainty"] == pytest.approx(brier(p, y))


@settings(max_examples=50, deadline=None)
@given(st.lists(st.tuples(st.floats(0, 1), st.integers(0, 1)), min_size=2, max_size=200))
def test_murphy_parts_always_sum_to_brier(pairs):
    p, y = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs], dtype=float)
    d = murphy_decomposition(p, y)
    assert d["reliability"] - d["resolution"] + d["uncertainty"] + d["within_bin"] == \
        pytest.approx(d["brier"], abs=1e-9)
    assert 0 <= brier(p, y) <= 1 and 0 <= ece(p, y) <= 1


def test_skill_score():
    assert skill_score(0.18, 0.24) == pytest.approx(0.25)


@pytest.mark.parametrize("p,y", [([0.5], [2]), ([1.2], [1]), ([0.5, 0.5], [1])])
def test_input_validation(p, y):
    with pytest.raises(ValueError):
        brier(p, y)


def test_bootstrap_ci_brackets_estimate_and_cluster_widens():
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(100), 10)
    shock = rng.normal(size=100)[day]                       # strongly correlated within day
    p = np.full(1000, 0.5)
    y = (rng.random(1000) < 1 / (1 + np.exp(-2 * shock))).astype(float)
    iid = bootstrap_ci(brier, p, y, n_boot=400, seed=1)
    clu = bootstrap_ci(brier, p, y, clusters=day, n_boot=400, seed=1)
    assert iid.low <= iid.estimate <= iid.high
    ll = bootstrap_ci(log_loss, rng.uniform(0.2, 0.8, 1000), y, clusters=day, n_boot=400)
    assert ll.low <= ll.estimate <= ll.high
    assert (clu.high - clu.low) >= (iid.high - iid.low) * 0.9  # not narrower than iid
    assert clu.estimate == iid.estimate


def test_paired_diff_detects_a_clearly_better_forecast():
    rng = np.random.default_rng(3)
    true = rng.uniform(0.05, 0.95, 3000)
    y = (rng.random(3000) < true).astype(float)
    d = paired_bootstrap_diff(brier, true, np.full(3000, 0.5), y, n_boot=300)
    assert d.high < 0                     # truth beats a coin flip, CI excludes 0
    same = paired_bootstrap_diff(brier, true, true, y, n_boot=100)
    assert same.estimate == 0 and same.low == 0 and same.high == 0
