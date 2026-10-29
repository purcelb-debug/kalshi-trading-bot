"""Point-in-time features, splits, models, holdout guard and the forecast chain."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select, update

from predlab.data.collectors.synthetic import generate
from predlab.db.repo import init_market_db, make_engine, upsert
from predlab.db.schema import ext_observations, forecasts, mkt_candles, synthetic_truth
from predlab.evaluation.metrics import brier
from predlab.evaluation.splits import (
    HoldoutAlreadyUsed,
    HoldoutGuard,
    assert_no_leak,
    holdout_split,
    walk_forward,
)
from predlab.evaluation.walkforward import evaluate_holdout, run_walk_forward
from predlab.features.pointintime import LookAheadError, build_dataset
from predlab.forecasts import record_forecasts, verify_chain
from predlab.models.baselines import (
    MODEL_FACTORIES,
    FrequencyBayes,
    LightGBM,
    Logistic,
    MarketImplied,
)
from predlab.timeutil import utcnow


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    eng = make_engine(f"sqlite:///{tmp_path_factory.mktemp('p2') / 'm.db'}")
    init_market_db(eng)
    generate(eng, seed=5, days=300)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def sports_ds(syn):
    return build_dataset(syn, source="synthetic", category="sports")


def truth(eng, tickers):
    with eng.connect() as c:
        t = dict(c.execute(select(synthetic_truth.c.ticker, synthetic_truth.c.true_p)).all())
    return np.array([t[k] for k in tickers])


# ---------------------------------------------------------------- point-in-time features

def test_dataset_is_time_sorted_and_flags_present(sports_ds):
    m = sports_ds.meta
    assert m["decision_ts"].is_monotonic_increasing
    assert (m["input_max_ts"] <= m["decision_ts"]).all()
    assert {"mkt_mid", "obs_x1", "obs_x2", "obs_x3"} <= set(sports_ds.feature_names)
    assert m["dq_flags"].map(type).eq(list).all()


def test_candles_after_decision_time_are_ignored(syn, sports_ds):
    """Leakage canary: a candle that ends after the decision must not change features."""
    row = sports_ds.meta.iloc[10]
    before = sports_ds.X.loc[10, "mkt_mid"]
    upsert(syn, mkt_candles, [{
        "ticker": row["ticker"], "period_minutes": 60,
        "end_ts": row["decision_ts"] + timedelta(minutes=30),
        "price_open": None, "price_high": None, "price_low": None, "price_close": None,
        "price_mean": None, "yes_bid_open": 0.97, "yes_bid_close": 0.97, "yes_ask_open": 0.99,
        "yes_ask_close": 0.99, "volume": 1e6, "open_interest": 0, "fetched_at": utcnow()}],
        ["ticker", "period_minutes", "end_ts"])
    ds2 = build_dataset(syn, source="synthetic", category="sports")
    i = ds2.meta.index[ds2.meta["ticker"] == row["ticker"]][0]
    assert ds2.X.loc[i, "mkt_mid"] == before


def test_late_vintage_observation_is_ignored(syn):
    ds = build_dataset(syn, source="synthetic", category="economics")
    row = ds.meta.iloc[3]
    old = ds.X.loc[3, "obs_x1"]
    upsert(syn, ext_observations, [{
        "source": "synthetic", "series_id": f"x1:{row['ticker']}", "obs_ts": row["decision_ts"],
        "vintage_ts": row["decision_ts"] + timedelta(seconds=1), "value": 999.0, "units": None,
        "fetched_at": utcnow()}], ["source", "series_id", "obs_ts", "vintage_ts"])
    ds2 = build_dataset(syn, source="synthetic", category="economics")
    i = ds2.meta.index[ds2.meta["ticker"] == row["ticker"]][0]
    assert ds2.X.loc[i, "obs_x1"] == old != 999.0


def test_lookahead_error_raised_if_inputs_postdate_decision(syn, monkeypatch):
    import predlab.features.pointintime as pit

    real = pit._market_state

    def leaky(engine, base):
        out = real(engine, base)
        out["mkt_ts"] = out["mkt_ts"] + pd.Timedelta(hours=12)   # pretend we used later data
        return out

    monkeypatch.setitem(pit.FEATURE_SETS, "market", leaky)
    with pytest.raises(LookAheadError):
        build_dataset(syn, source="synthetic", category="weather")


def test_market_features_can_be_excluded(syn):
    ds = build_dataset(syn, source="synthetic", category="weather", use_market_features=False)
    assert not any(f.startswith("mkt_") for f in ds.feature_names)
    assert ds.meta["market_mid"].notna().all()  # still available as the reference price


# ---------------------------------------------------------------- splits

def test_holdout_is_latest_and_disjoint(sports_ds):
    dev, hold, h = holdout_split(sports_ds, 0.2)
    ts = sports_ds.meta["decision_ts"]
    assert set(dev).isdisjoint(hold) and len(dev) + len(hold) == len(sports_ds)
    assert ts.iloc[dev].max() < ts.iloc[hold].min()
    assert 0.17 < len(hold) / len(sports_ds) < 0.23
    assert holdout_split(sports_ds, 0.2)[2] == h and len(h) == 64


def test_walk_forward_folds_respect_time_and_label_availability(sports_ds):
    dev, hold, _ = holdout_split(sports_ds, 0.2)
    emb = timedelta(hours=12)
    folds = walk_forward(sports_ds, dev, n_folds=4, embargo=emb)
    assert len(folds) == 4
    m = sports_ds.meta
    for f in folds:
        assert set(f.train).isdisjoint(hold) and set(f.valid).isdisjoint(hold)
        assert m["decision_ts"].iloc[f.train].max() < m["decision_ts"].iloc[f.valid].min()
        assert m["label_ts"].iloc[f.train].max() <= m["decision_ts"].iloc[f.valid].min() - emb
        assert_no_leak(sports_ds, f, emb)
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes)  # expanding window


def test_assert_no_leak_catches_overlap(sports_ds):
    dev, _, _ = holdout_split(sports_ds, 0.2)
    f = walk_forward(sports_ds, dev, n_folds=3)[1]
    from predlab.evaluation.splits import Fold
    bad = Fold(9, np.concatenate([f.train, f.valid[:5]]), f.valid, f.valid_start, f.valid_end)
    with pytest.raises(LookAheadError):
        assert_no_leak(sports_ds, bad)


# ---------------------------------------------------------------- models

def test_logistic_recovers_true_probabilities(syn):
    """Without market features the model sees x only, so it can at best learn
    sigmoid(b.x) attenuated by hidden noise; it must clearly beat a constant and land
    near the oracle that knows the true p."""
    ds = build_dataset(syn, source="synthetic", category="sports", use_market_features=False)
    n = len(ds)
    tr, va = ds.subset(np.arange(n) < n * 0.7), ds.subset(np.arange(n) >= n * 0.7)
    m = Logistic("l2", n_boot=0).fit(tr)
    p = m.predict(va).p
    tp = truth(syn, va.meta["ticker"])
    oracle, model, const = brier(tp, va.y), brier(p, va.y), brier(np.full(len(va), tr.y.mean()), va.y)
    assert model < const - 0.02
    assert model - oracle < 0.04
    assert np.corrcoef(p, tp)[0, 1] > 0.75


def test_logistic_bootstrap_ci_and_explanations(syn):
    ds = build_dataset(syn, source="synthetic", category="weather", use_market_features=False)
    n = len(ds)
    tr, va = ds.subset(np.arange(n) < n * 0.7), ds.subset(np.arange(n) >= n * 0.7)
    fc = Logistic(None, n_boot=20).fit(tr).forecasts(va)
    assert all(f.ci_low <= f.p <= f.ci_high for f in fc[:50])
    top = fc[0].explanation["top_features"]
    assert len(top) == 3 and all("feature" in t for t in top)
    # x3 is pure noise in the generator; it should rarely be the strongest driver
    first = [f.explanation["top_features"][0]["feature"] for f in fc]
    assert first.count("obs_x3") / len(first) < 0.2


def test_frequency_bayes_credible_interval_covers_true_rate(syn):
    ds = build_dataset(syn, source="synthetic", category="economics")
    m = FrequencyBayes(level=0.9).fit(ds)
    pr = m.predict(ds.subset(np.arange(len(ds)) < 1))
    assert pr.ci_low[0] <= ds.y.mean() <= pr.ci_high[0]
    assert pr.explanations[0]["n_train"] == len(ds)
    other = build_dataset(syn, source="synthetic", category="sports")
    assert m.predict(other.subset(np.arange(len(other)) < 1)).flags[0] == ["unseen_group_pooled"]


def test_market_implied_uses_mid_and_flags_fallback(sports_ds):
    m = MarketImplied().fit(sports_ds)
    ds = sports_ds.subset(np.arange(len(sports_ds)) < 5)
    ds.meta.loc[0, "market_mid"] = np.nan
    pr = m.predict(ds)
    assert pr.p[1] == pytest.approx(ds.meta["market_mid"].iloc[1])
    assert pr.flags[0] == ["fallback_base_rate"] and pr.p[0] == pytest.approx(sports_ds.y.mean())


def test_lightgbm_shap_explanations(sports_ds):
    n = len(sports_ds)
    m = LightGBM().fit(sports_ds.subset(np.arange(n) < n * 0.7))
    fc = m.forecasts(sports_ds.subset(np.arange(n) >= n * 0.9))
    assert "no_ci" in fc[0].dq_flags and fc[0].ci_low is None
    assert len(fc[0].explanation["top_features"]) == 3


def test_model_versions_are_stable_and_param_sensitive():
    assert Logistic("l2").version == Logistic("l2").version
    assert Logistic("l2").version != Logistic("l1").version
    assert LightGBM().version != LightGBM(num_leaves=31).version


# ---------------------------------------------------------------- walk-forward + holdout

def test_walk_forward_runs_all_models_on_identical_rows(sports_ds):
    res = run_walk_forward(sports_ds, MODEL_FACTORIES, n_folds=3, n_boot=50)
    assert res.errors == {}
    counts = res.predictions.groupby("model")["ticker"].apply(frozenset)
    assert counts.nunique() == 1  # paired comparison: same validation markets for every model
    assert set(res.predictions["ticker"]).isdisjoint(set(sports_ds.meta["ticker"].iloc[res.holdout_rows]))
    t = res.table()
    assert set(t["model"]) == set(MODEL_FACTORIES)
    assert t.loc[t["model"] == "market_implied", "vs_market_brier_diff"].isna().all()


def test_failing_model_is_reported_not_fatal(sports_ds):
    class Broken(MarketImplied):
        name = "broken"

        def fit(self, ds):
            raise RuntimeError("boom")

    res = run_walk_forward(sports_ds, {"market_implied": MarketImplied, "broken": Broken},
                           n_folds=2, n_boot=20)
    assert "broken" in res.errors and set(res.predictions["model"]) == {"market_implied"}


def test_holdout_scored_once_per_version(sports_ds, sim_engine):
    res = run_walk_forward(sports_ds, {"market_implied": MarketImplied}, n_folds=2, n_boot=20)
    guard = HoldoutGuard(sim_engine)
    m = evaluate_holdout(sports_ds, res, "frequency_bayes", FrequencyBayes, guard, n_boot=50)
    assert m["n"] == len(res.holdout_rows)
    with pytest.raises(HoldoutAlreadyUsed):
        evaluate_holdout(sports_ds, res, "frequency_bayes", FrequencyBayes, guard, n_boot=50)
    # a different version is a different model and may be scored
    evaluate_holdout(sports_ds, res, "fb2", lambda: FrequencyBayes(prior=(2, 2)), guard, n_boot=50)


# ---------------------------------------------------------------- forecast chain

def test_forecast_chain_detects_tampering(sports_ds, sim_engine):
    ds = sports_ds.subset(np.arange(len(sports_ds)) < 20)
    fc = MarketImplied().fit(ds).forecasts(ds)
    assert record_forecasts(sim_engine, fc[:10]) == 10
    record_forecasts(sim_engine, fc[10:])
    assert verify_chain(sim_engine) == []
    with sim_engine.begin() as c:
        c.execute(update(forecasts).where(forecasts.c.id == 5).values(p=0.99))
    assert verify_chain(sim_engine) == [5]


def test_forecast_with_future_inputs_rejected(sports_ds, sim_engine):
    from dataclasses import replace
    ds = sports_ds.subset(np.arange(len(sports_ds)) < 1)
    f = MarketImplied().fit(ds).forecasts(ds)[0]
    with pytest.raises(ValueError, match="future"):
        record_forecasts(sim_engine, [replace(f, input_max_ts=utcnow() + timedelta(days=1))])


def test_market_coverage_reported(sports_ds):
    res = run_walk_forward(sports_ds, {"market_implied": MarketImplied}, n_folds=2, n_boot=20)
    assert res.market_coverage == 1.0
    ds = sports_ds.subset(np.ones(len(sports_ds), bool))
    ds.meta["market_mid"] = np.nan
    assert run_walk_forward(ds, {"market_implied": MarketImplied}, n_folds=2,
                            n_boot=20).market_coverage == 0.0
