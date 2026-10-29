"""Phase 4: trajectory features/model, calibration, stacking, category ranking, stress, probes."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from predlab.backtest.engine import StrategyConfig, run_backtest
from predlab.backtest.execution import ExecConfig, MarketData, Quote
from predlab.backtest.stress import StressedMarket, run_stress_suite
from predlab.data.collectors.synthetic import generate
from predlab.db.repo import init_market_db, make_engine
from predlab.db.schema import mkt_candles, synthetic_truth
from predlab.evaluation.category_rank import DISCLAIMER, compare_categories
from predlab.evaluation.metrics import brier, ece
from predlab.evaluation.overfit import multiple_testing_check, shuffled_label_probe
from predlab.evaluation.walkforward import run_walk_forward
from predlab.features.pointintime import build_dataset
from predlab.models.advanced import Calibrated, MarketTrajectory, Stacked
from predlab.models.base import Forecaster, Predictions, clip_p
from predlab.models.baselines import Logistic, MarketImplied
from predlab.models.registry import ALL_MODELS


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    eng = make_engine(f"sqlite:///{tmp_path_factory.mktemp('p4') / 'm.db'}")
    init_market_db(eng)
    generate(eng, seed=21, days=260)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def sports(syn):
    return build_dataset(syn, source="synthetic", category="sports")


def truth(eng):
    with eng.connect() as c:
        return dict(c.execute(select(synthetic_truth.c.ticker, synthetic_truth.c.true_p)).all())


class Oracle(Forecaster):
    """Reads the generator's true p (test-only)."""
    name = "oracle"

    def __init__(self, tp):
        self.tp = tp

    def fit(self, ds):
        return self

    def predict(self, ds):
        return Predictions(clip_p(ds.meta["ticker"].map(self.tp).to_numpy()))


class Squashed(MarketImplied):
    """Deliberately miscalibrated: pushes the market mid toward the extremes."""
    name = "squashed"

    def predict(self, ds):
        pr = super().predict(ds)
        z = np.log(pr.p / (1 - pr.p)) * 2.5
        return Predictions(clip_p(1 / (1 + np.exp(-z))))


# ---------------------------------------------------------------- features / models

def test_momentum_uses_only_older_quotes(syn, sports):
    i = 40
    t, d = sports.meta.loc[i, "ticker"], sports.meta.loc[i, "decision_ts"]
    with syn.connect() as c:
        cd = pd.DataFrame(c.execute(select(mkt_candles.c.end_ts, mkt_candles.c.yes_bid_close,
                                           mkt_candles.c.yes_ask_close)
                                    .where(mkt_candles.c.ticker == t)).mappings().all())
    cd["end_ts"] = pd.to_datetime(cd["end_ts"], utc=True)
    cd["mid"] = (cd["yes_bid_close"] + cd["yes_ask_close"]) / 2
    now = cd[cd["end_ts"] <= d].sort_values("end_ts")["mid"].iloc[-1]
    old = cd[cd["end_ts"] <= d - timedelta(hours=6)].sort_values("end_ts")["mid"].iloc[-1]

    def lg(p):
        return np.log(p / (1 - p))

    assert sports.X.loc[i, "mkt_mom6"] == pytest.approx(lg(now) - lg(old), abs=1e-9)


def test_market_trajectory_requires_market_features(syn):
    ds = build_dataset(syn, source="synthetic", category="weather", use_market_features=False)
    with pytest.raises(ValueError):
        MarketTrajectory().fit(ds)


def test_calibration_fixes_a_miscalibrated_model(sports):
    n = len(sports)
    tr, va = sports.subset(np.arange(n) < n * 0.7), sports.subset(np.arange(n) >= n * 0.7)
    raw = Squashed().fit(tr).predict(va).p
    cal = Calibrated(Squashed, "platt").fit(tr)
    fixed = cal.predict(va).p
    assert ece(fixed, va.y) < ece(raw, va.y)
    assert brier(fixed, va.y) < brier(raw, va.y)
    # Platt should roughly undo the 2.5x logit stretch: slope near 1/2.5 = 0.4
    assert 0.25 < cal.cal_.coef_[0][0] < 0.6
    # base model was fit only on the earlier 70% of the training window
    assert cal.base_.base_rate_ == pytest.approx(tr.y[: round(len(tr) * 0.7)].mean())


def test_stacked_weights_are_a_simplex_and_favor_the_oracle(syn, sports):
    tp = truth(syn)
    n = len(sports)
    tr = sports.subset(np.arange(n) < n * 0.7)
    m = Stacked({"market_implied": MarketImplied, "oracle": lambda: Oracle(tp)}).fit(tr)
    w = m.weights_
    assert sum(w.values()) == pytest.approx(1.0) and all(v >= 0 for v in w.values())
    assert w["oracle"] > 0.6


def test_registry_models_all_run(sports):
    res = run_walk_forward(sports, ALL_MODELS, n_folds=2, n_boot=20)
    assert res.errors == {} and set(res.summary) == set(ALL_MODELS)


# ---------------------------------------------------------------- category ranking

def test_category_ranking_covers_all_categories(syn):
    facs = {"market_implied": MarketImplied, "logistic_l2": lambda: Logistic("l2", n_boot=0)}
    table, skipped = compare_categories(syn, facs, source="synthetic", n_folds=3, n_boot=50,
                                        min_rows=40)
    assert set(table["category"]) | set(skipped) == {"weather", "economics", "financial",
                                                      "sports", "public_events"}
    assert list(table["rank"]) == list(range(1, len(table) + 1))
    assert table["composite_rank_score"].is_monotonic_increasing
    assert {"n", "brier", "log_loss", "ece", "fold_brier_sd", "freshness_days",
            "independent_info", "market_coverage", "est_cost_per_contract",
            "oos_win_share"} <= set(table.columns)
    assert "Not a recommendation" in DISCLAIMER


# ---------------------------------------------------------------- stress

def _oracle_run(syn, cat="public_events"):
    ds = build_dataset(syn, source="synthetic", category=cat)
    tp = truth(syn)
    sig = ds.meta[["ticker", "decision_ts", "cluster", "category", "series_ticker"]].assign(
        p=ds.meta["ticker"].map(tp).to_numpy())
    res = pd.DataFrame({"ticker": ds.meta["ticker"], "resolve_ts": ds.meta["label_ts"],
                        "result": np.where(ds.y == 1, "yes", "no")})
    return sig, res, MarketData(syn, ds.meta["ticker"])


def test_stress_suite_degrades_under_adverse_assumptions(syn):
    sig, res, md = _oracle_run(syn)
    t = run_stress_suite(sig, res, md).set_index("scenario")
    base = t.loc["base", "net_pnl"]
    assert base > 0
    for s in ("adverse_price_move_2c", "wrong_resolution_5pct", "fees_x3", "forecast_noise_0.5"):
        assert t.loc[s, "net_pnl"] < base, s
    assert t.loc["fees_x3", "fees"] / max(t.loc["fees_x3", "n_fills"], 1) > \
        t.loc["base", "fees"] / t.loc["base", "n_fills"]
    assert len(t) == 13


class _Two:
    """Old view cheap, current book expensive: the classic stale-quote trap."""
    T0 = pd.Timestamp("2026-01-05 12:00", tz="UTC")

    def quote(self, ticker, ts, cfg):
        px = 0.30 if ts < self.T0 else 0.70
        return Quote(ts - timedelta(minutes=5), "book", [(px, 500.0)], [(round(1.3 - px, 2), 500.0)],
                     px - 0.02, 0.3), ""


def test_stale_view_never_fills_at_the_old_price():
    """Regression (2026-10-29): stale quotes once both priced and filled the order at the old
    price, inventing profit. The view may be stale; the fill must use the current book."""
    md = StressedMarket(_Two(), stale_lag=timedelta(hours=3))
    sig = pd.DataFrame({"ticker": ["A"], "decision_ts": [_Two.T0], "p": [0.60], "cluster": ["E"],
                        "category": ["x"], "series_ticker": ["S"]})
    res = pd.DataFrame({"ticker": ["A"], "resolve_ts": [_Two.T0 + timedelta(days=1)],
                        "result": ["yes"]})
    out = run_backtest(sig, res, md, strategy=StrategyConfig(min_edge=0.02),
                       execution=ExecConfig(max_quote_age=timedelta(hours=5)))
    assert out.summary["n_orders"] == 1                    # the stale view saw an edge...
    assert out.summary["n_fills"] == 0                     # ...but the real ask (0.70) > limit
    assert out.orders["status"].iloc[0] == "unfilled"


# ---------------------------------------------------------------- probes

class Peeker(Forecaster):
    """Simulates a pipeline leak: its predictions are derived from the labels."""
    name = "peeker"

    def fit(self, ds):
        return self

    def predict(self, ds):
        return Predictions(clip_p(0.2 + 0.6 * ds.y))


def test_shuffled_label_probe_flags_leaks_only(sports):
    out = shuffled_label_probe(sports, {"market_implied": MarketImplied,
                                        "logistic_l2": lambda: Logistic("l2", n_boot=0),
                                        "peeker": Peeker}, n_perm=2, n_folds=2)
    assert out["peeker"]["suspicious"]
    assert not out["logistic_l2"]["suspicious"]
    assert out["market_implied"]["mean_gain_vs_constant"] < 0  # real prices vs shuffled outcomes


def test_multiple_testing_uses_bonferroni_level(sports):
    res = run_walk_forward(sports, {"market_implied": MarketImplied,
                                    "logistic_l2": lambda: Logistic("l2", n_boot=0),
                                    "logistic": lambda: Logistic(None, n_boot=0)},
                           n_folds=2, n_boot=20)
    out = multiple_testing_check(res, n_boot=200)
    assert set(out) == {"logistic_l2", "logistic"}
    assert all(v["level"] == pytest.approx(1 - 0.05 / 2) for v in out.values())
    assert all(v["low"] <= v["diff"] <= v["high"] for v in out.values())
