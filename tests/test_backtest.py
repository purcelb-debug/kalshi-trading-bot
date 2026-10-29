"""Phase 3: fees, fills, accounting and the event-driven engine, against hand-built cases."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from predlab.backtest.engine import (
    StrategyConfig,
    _max_drawdown,
    run_backtest,
    save_run,
    sensitivity_grid,
    signals_from_walk_forward,
)
from predlab.backtest.execution import ExecConfig, MarketData, Quote, simulate_buy
from predlab.backtest.fees import FeeSchedule, round_up_cent
from predlab.backtest.portfolio import Portfolio
from predlab.data.collectors.synthetic import generate
from predlab.db.repo import init_market_db, make_engine, upsert
from predlab.db.schema import (
    mkt_candles,
    mkt_orderbook_snapshots,
    sim_fills,
    sim_runs,
    synthetic_truth,
)
from predlab.features.pointintime import LookAheadError, build_dataset
from predlab.timeutil import to_utc, utcnow

T0 = pd.Timestamp("2026-01-05 12:00", tz="UTC")


# ---------------------------------------------------------------- fees

@pytest.mark.parametrize("c,p,expected", [
    (100, 0.50, 1.75),   # published top of range for 100 contracts
    (100, 0.01, 0.07),   # published bottom of range
    (1, 0.50, 0.02),     # 0.0175 rounds up to 2 cents
    (10, 0.40, 0.17),    # 0.07*10*0.4*0.6 = 0.168
    (0, 0.50, 0.0),
])
def test_taker_fee_matches_published_formula(c, p, expected):
    assert FeeSchedule().fee(c, p) == pytest.approx(expected)


def test_maker_multiplier_stress_and_no_fee_series():
    assert FeeSchedule(maker_fees=True).fee(100, 0.5, "maker") == pytest.approx(0.44)
    assert FeeSchedule(maker_fees=False).fee(100, 0.5, "maker") == 0.0
    assert FeeSchedule(multiplier=2).fee(100, 0.5) == pytest.approx(3.50)
    assert FeeSchedule(stress=3).fee(100, 0.5) == pytest.approx(5.25)
    assert FeeSchedule.from_series("none", None).fee(100, 0.5) == 0.0
    assert FeeSchedule.from_series("quadratic_with_maker_fees", 1).maker_fees
    assert FeeSchedule.from_series("something_new", None).fee(100, 0.5) == pytest.approx(1.75)
    with pytest.raises(ValueError):
        FeeSchedule().fee(1, 1.0)


def test_round_up_cent_ignores_float_noise():
    assert round_up_cent(1.7500000000001) == 1.75 and round_up_cent(1.7501) == 1.76


# ---------------------------------------------------------------- fills

def book_quote(no_bids, yes_bids, ts=T0):
    return Quote(ts, "book",
                 sorted((round(1 - p, 4), s) for p, s in no_bids),
                 sorted((round(1 - p, 4), s) for p, s in yes_bids),
                 max((p for p, _ in yes_bids), default=None),
                 max((p for p, _ in no_bids), default=None))


def test_book_fill_walks_depth_with_participation_cap():
    q = book_quote(no_bids=[(0.60, 100), (0.58, 50)], yes_bids=[(0.35, 80)])
    cfg = ExecConfig(participation=0.5)
    fr = simulate_buy(q, "yes", 100, limit=0.45, cfg=cfg, decision_ts=T0)
    assert fr.levels == [(0.40, 50.0), (0.42, 25.0)]
    assert fr.qty == 75 and fr.reason == "partial_fill" and not fr.liquidity_uncertain
    assert fr.avg_price == pytest.approx((50 * 0.40 + 25 * 0.42) / 75)
    assert simulate_buy(q, "yes", 100, 0.41, cfg, T0).qty == 50       # limit stops at level 1
    no = simulate_buy(q, "no", 10, 0.70, cfg, T0)                     # NO ask = 1 - 0.35
    assert no.levels == [(0.65, 10.0)]


def test_never_fills_at_midpoint():
    q = book_quote(no_bids=[(0.55, 1000)], yes_bids=[(0.40, 1000)])   # bid .40 / ask .45
    fr = simulate_buy(q, "yes", 10, 0.99, ExecConfig(), T0)
    assert fr.avg_price == pytest.approx(0.45) != 0.425


def test_candle_quote_is_uncertain_capped_and_slipped():
    q = Quote(T0, "candle", [(0.40, np.inf)], [(0.65, np.inf)], 0.35, 0.60)
    fr = simulate_buy(q, "yes", 100, 0.99, ExecConfig(unknown_depth_cap=10, slippage=0.01), T0)
    assert fr.qty == 10 and fr.avg_price == pytest.approx(0.41) and fr.liquidity_uncertain


def test_quote_after_decision_is_rejected():
    q = book_quote([(0.6, 10)], [], ts=T0 + timedelta(minutes=1))
    with pytest.raises(LookAheadError):
        simulate_buy(q, "yes", 1, 0.9, ExecConfig(), T0)


@pytest.fixture
def md_engine(tmp_path):
    eng = make_engine(f"sqlite:///{tmp_path / 'md.db'}")
    init_market_db(eng)
    now = utcnow()
    upsert(eng, mkt_candles, [{
        "ticker": "C", "period_minutes": 60, "end_ts": T0 - timedelta(hours=1),
        "price_open": None, "price_high": None, "price_low": None, "price_close": None,
        "price_mean": None, "yes_bid_open": .3, "yes_bid_close": .30, "yes_ask_open": .34,
        "yes_ask_close": .34, "volume": 5, "open_interest": 5, "fetched_at": now}],
        ["ticker", "period_minutes", "end_ts"])
    upsert(eng, mkt_orderbook_snapshots, [
        {"ticker": "C", "snapshot_ts": T0 - timedelta(hours=3), "side": "no", "price": .7,
         "size": 100},
        {"ticker": "B", "snapshot_ts": T0 - timedelta(minutes=10), "side": "no", "price": .55,
         "size": 40}], ["ticker", "snapshot_ts", "side", "price"])
    yield eng
    eng.dispose()


def test_quote_source_priority_and_staleness(md_engine):
    md = MarketData(md_engine, ["B", "C", "X"])
    cfg = ExecConfig(max_book_age=timedelta(hours=1), max_quote_age=timedelta(hours=2))
    qb, _ = md.quote("B", T0, cfg)
    assert qb.source == "book" and qb.yes_asks == [(0.45, 40.0)]
    qc, _ = md.quote("C", T0, cfg)               # book is 3h old -> falls back to candle
    assert qc.source == "candle" and qc.yes_asks[0][0] == pytest.approx(0.34)
    assert qc.no_asks[0][0] == pytest.approx(0.70)
    assert md.quote("C", T0 + timedelta(hours=2), cfg) == (None, "stale_quote")
    assert md.quote("X", T0, cfg) == (None, "no_liquidity_data")
    assert md.quote("B", T0 - timedelta(hours=1), cfg) == (None, "no_liquidity_data")


# ---------------------------------------------------------------- accounting

def test_portfolio_gross_net_and_settlement():
    pf = Portfolio(100.0)
    fee = FeeSchedule().fee(10, 0.40)
    pf.buy("A", "yes", 10, 0.40, fee, mark=0.35)
    assert pf.cash == pytest.approx(100 - 4.0 - 0.17)
    assert pf.equity == pytest.approx(pf.cash + 10 * 0.35)   # marked at bid, not cost
    assert pf.exposure == pytest.approx(4.0)
    assert pf.settle("A", "yes") == pytest.approx(6.0)
    assert pf.cash == pytest.approx(100 - 4.17 + 10) and pf.exposure == 0
    assert pf.equity - pf.initial == pytest.approx(6.0 - 0.17)  # net = gross - fees
    with pytest.raises(ValueError):
        pf.buy("B", "yes", 1000, 0.5, 0, 0.4)


def test_max_drawdown_hand_calculation():
    assert _max_drawdown(pd.Series([100, 120, 90, 130, 104])) == pytest.approx(0.25)


# ---------------------------------------------------------------- engine

def tiny_world(md_engine):
    """Two markets with book quotes; signals with an edge on each."""
    upsert(md_engine, mkt_orderbook_snapshots, [
        {"ticker": "M1", "snapshot_ts": T0, "side": "no", "price": .60, "size": 400},
        {"ticker": "M1", "snapshot_ts": T0, "side": "yes", "price": .36, "size": 400},
        {"ticker": "M2", "snapshot_ts": T0 + timedelta(days=1), "side": "yes", "price": .50,
         "size": 400},
        {"ticker": "M2", "snapshot_ts": T0 + timedelta(days=1), "side": "no", "price": .45,
         "size": 400}], ["ticker", "snapshot_ts", "side", "price"])
    sig = pd.DataFrame({"ticker": ["M1", "M2"], "decision_ts": [T0, T0 + timedelta(days=1)],
                        "p": [0.60, 0.30], "cluster": ["E1", "E2"], "category": ["x", "x"],
                        "series_ticker": ["S", "S"]})
    return sig, MarketData(md_engine, ["M1", "M2"])


def test_engine_decisions_do_not_depend_on_outcomes(md_engine):
    sig, md = tiny_world(md_engine)
    strat = StrategyConfig(initial_cash=1000, min_edge=0.02, max_market_frac=0.05)
    # Every market resolves after every decision, so no decision may differ when outcomes
    # flip. (Earlier resolutions legitimately change later sizing via realized equity.)
    res_a = pd.DataFrame({"ticker": ["M1", "M2"], "resolve_ts": [T0 + timedelta(days=3)] * 2,
                          "result": ["yes", "no"]})
    res_b = res_a.assign(result=["no", "yes"])
    a = run_backtest(sig, res_a, md, strategy=strat)
    b = run_backtest(sig, res_b, md, strategy=strat)
    cols = ["ticker", "side", "qty_requested", "qty_filled", "limit_price"]
    assert a.orders[cols].equals(b.orders[cols])            # identical decisions
    assert list(a.orders["side"]) == ["yes", "no"]          # M1 cheap YES, M2 cheap NO
    assert a.summary["net_pnl"] > 0 > b.summary["net_pnl"]
    assert a.summary["fees"] == b.summary["fees"] > 0
    s = a.summary
    assert s["net_pnl"] == pytest.approx(s["gross_pnl_settled"] - s["fees"])
    assert s["open_positions_at_end"] == 0 and s["uncertain_fill_share"] == 0


def test_settlement_cash_is_available_to_same_time_decision(md_engine):
    sig, md = tiny_world(md_engine)
    # M1 spends nearly all cash (23 @ 0.40 + 0.39 fee leaves 0.41 < one M2 contract).
    # M1 resolves at the exact timestamp of the M2 decision.
    strat = StrategyConfig(initial_cash=10, min_edge=0.02, max_market_frac=1.0,
                           max_cluster_frac=1.0, max_total_exposure_frac=1.0, kelly_fraction=3.0)
    when = [T0 + timedelta(days=1), T0 + timedelta(days=5)]
    win = run_backtest(sig, pd.DataFrame({"ticker": ["M1", "M2"], "resolve_ts": when,
                                          "result": ["yes", "no"]}), md, strategy=strat)
    lose = run_backtest(sig, pd.DataFrame({"ticker": ["M1", "M2"], "resolve_ts": when,
                                           "result": ["no", "no"]}), md, strategy=strat)
    assert win.fills[win.fills["ticker"] == "M1"]["qty"].sum() == 23
    assert win.fills[win.fills["ticker"] == "M2"]["qty"].sum() > 0   # funded by the payout
    assert "M2" not in set(lose.fills["ticker"])
    assert lose.summary["rejections"] == {"size_below_one_contract": 1}


def test_no_trade_without_edge_and_rejections_counted(md_engine):
    sig, md = tiny_world(md_engine)
    sig = sig.assign(p=[0.40, 0.52])                       # at or inside the spread
    res = pd.DataFrame({"ticker": ["M1", "M2"], "resolve_ts": [T0 + timedelta(days=3)] * 2,
                        "result": ["yes", "no"]})
    out = run_backtest(sig, res, md)
    assert out.summary["n_fills"] == 0
    assert out.summary["rejections"] == {"edge_below_threshold": 2}


# ---------------------------------------------------------------- synthetic end-to-end

@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    eng = make_engine(f"sqlite:///{tmp_path_factory.mktemp('bt') / 'm.db'}")
    init_market_db(eng)
    generate(eng, seed=9, days=240)
    yield eng
    eng.dispose()


def _signals(syn, ds, p):
    sig = ds.meta[["ticker", "decision_ts", "cluster", "category", "series_ticker"]].assign(p=p)
    res = pd.DataFrame({"ticker": ds.meta["ticker"], "resolve_ts": ds.meta["label_ts"],
                        "result": np.where(ds.y == 1, "yes", "no")})
    return sig, res, MarketData(syn, ds.meta["ticker"])


def test_oracle_profits_and_market_mid_does_not_trade(syn):
    """Sanity bounds: knowing the true p beats an inefficient market net of costs; trading
    on the market's own mid finds no edge after spread and fees."""
    ds = build_dataset(syn, source="synthetic", category="public_events")
    with syn.connect() as c:
        tp = dict(c.execute(select(synthetic_truth.c.ticker, synthetic_truth.c.true_p)).all())
    sig, res, md = _signals(syn, ds, ds.meta["ticker"].map(tp).to_numpy())
    oracle = run_backtest(sig, res, md)
    assert oracle.summary["n_fills"] > 10 and oracle.summary["net_pnl"] > 0
    sig_m = sig.assign(p=ds.meta["market_mid"].fillna(0.5).to_numpy())
    mid = run_backtest(sig_m, res, md)
    assert mid.summary["n_fills"] == 0


def test_fee_and_slippage_sensitivity_is_monotone(syn):
    ds = build_dataset(syn, source="synthetic", category="public_events")
    with syn.connect() as c:
        tp = dict(c.execute(select(synthetic_truth.c.ticker, synthetic_truth.c.true_p)).all())
    sig, res, md = _signals(syn, ds, ds.meta["ticker"].map(tp).to_numpy())

    def run(fs, sl):
        return run_backtest(sig, res, md, execution=ExecConfig(slippage=sl),
                            default_fees=FeeSchedule(stress=fs))

    g = sensitivity_grid(run, fee_stress=(1.0, 3.0), slippage=(0.0, 0.03))
    base = g[(g.fee_stress == 1) & (g.slippage == 0)]["net_pnl"].item()
    worst = g[(g.fee_stress == 3) & (g.slippage == 0.03)]["net_pnl"].item()
    assert worst < base and len(g) == 4


def test_walk_forward_signals_and_persistence(syn, sim_engine):
    from predlab.evaluation.walkforward import run_walk_forward
    from predlab.models.baselines import Logistic, MarketImplied
    ds = build_dataset(syn, source="synthetic", category="sports")
    wf = run_walk_forward(ds, {"market_implied": MarketImplied,
                               "logistic_l2": lambda: Logistic("l2", n_boot=0)},
                          n_folds=3, n_boot=20)
    sig, res = signals_from_walk_forward(wf, "logistic_l2", ds)
    assert set(sig["ticker"]) == set(wf.predictions["ticker"])            # OOS rows only
    assert set(sig["ticker"]).isdisjoint(ds.meta["ticker"].iloc[wf.holdout_rows])
    out = run_backtest(sig, res, MarketData(syn, sig["ticker"]))
    run_id = save_run(sim_engine, out, model_version=wf.versions["logistic_l2"])
    with sim_engine.connect() as c:
        assert c.execute(select(func.count()).select_from(sim_runs)).scalar() == 1
        assert c.execute(select(func.count()).select_from(sim_fills)
                         .where(sim_fills.c.run_id == run_id)).scalar() == len(out.fills)
    assert out.summary["banner"] == "PAPER TRADING — NOT REAL MONEY"
    assert to_utc(out.equity["ts"].max()) <= to_utc(res["resolve_ts"].max())
