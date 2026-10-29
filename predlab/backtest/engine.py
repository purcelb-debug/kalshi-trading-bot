"""Event-driven historical paper-trading simulator. PAPER TRADING, NOT REAL MONEY.

Events are processed in time order. At equal timestamps, settlements happen before
decisions (cash from a resolved market is available, but a decision never sees an outcome
that resolves at or after it). The strategy receives only the forecast and the quote; the
outcome is applied by the engine at the settlement event.

Signals must be out-of-sample forecasts (e.g. walk-forward validation predictions);
``signals_from_walk_forward`` builds them from an ``EvalResult``.
"""

import math
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import Engine, select

from predlab import BANNER
from predlab.backtest.execution import ExecConfig, MarketData, simulate_buy
from predlab.backtest.fees import FeeSchedule
from predlab.backtest.portfolio import Portfolio
from predlab.db.schema import (
    mkt_series,
    sim_equity,
    sim_fills,
    sim_orders,
    sim_positions,
    sim_runs,
)
from predlab.evaluation.metrics import brier
from predlab.timeutil import utcnow


@dataclass(frozen=True)
class StrategyConfig:
    initial_cash: float = 1000.0
    min_edge: float = 0.03             # required expected value per contract after fees ($)
    kelly_fraction: float = 0.25       # fraction of full Kelly
    max_market_frac: float = 0.02      # max cost in one market, share of equity
    max_cluster_frac: float = 0.05     # max cost in one event (correlated markets)
    max_total_exposure_frac: float = 0.5
    allow_no: bool = True


@dataclass
class BacktestResult:
    run_id: str
    summary: dict
    orders: pd.DataFrame
    fills: pd.DataFrame
    positions: pd.DataFrame
    equity: pd.DataFrame
    config: dict = field(default_factory=dict)


def _ev(p_side: float, price: float, fees: FeeSchedule) -> float:
    return p_side - price - fees.per_contract(price)


def _kelly(q: float, a: float) -> float:
    return max(0.0, (q - a) / (1 - a)) if a < 1 else 0.0


def run_backtest(signals: pd.DataFrame, resolutions: pd.DataFrame, market: MarketData, *,
                 strategy: StrategyConfig | None = None, execution: ExecConfig | None = None,
                 fee_for: dict[str, FeeSchedule] | None = None,
                 default_fees: FeeSchedule | None = None) -> BacktestResult:
    """signals: ticker, decision_ts, p, cluster, category, series_ticker
    resolutions: ticker, resolve_ts, result ("yes"/"no")."""
    strategy = strategy or StrategyConfig()
    execution = execution or ExecConfig()
    default_fees = default_fees or FeeSchedule()
    fee_for = fee_for or {}
    signals = signals.assign(decision_ts=pd.to_datetime(signals["decision_ts"], utc=True))
    resolutions = resolutions.assign(resolve_ts=pd.to_datetime(resolutions["resolve_ts"], utc=True))
    pf = Portfolio(strategy.initial_cash)
    events = [(r.decision_ts, 1, "decide", r) for r in signals.itertuples()]
    events += [(r.resolve_ts, 0, "settle", r) for r in resolutions.itertuples()
               if r.ticker in set(signals["ticker"])]
    events.sort(key=lambda e: (e[0], e[1]))
    orders, fills = [], []
    cluster_cost: Counter = Counter()
    rejects: Counter = Counter()
    order_id = 0
    last_ts = None
    for ts, _, kind, r in events:
        if last_ts is not None and ts != last_ts:
            pf.snapshot(last_ts)
        last_ts = ts
        if kind == "settle":
            pf.settle(r.ticker, r.result)
            for (t, _side), pos in pf.positions.items():
                if t == r.ticker:
                    cluster_cost[pos.cluster] -= pos.cost
            continue
        fees = fee_for.get(r.series_ticker, default_fees)
        quote, why = market.quote(r.ticker, ts, execution)
        if quote is None:
            rejects[why] += 1
            continue
        slip = execution.slippage if quote.source != "book" else 0.0
        best = None
        for side, q_side, asks in (("yes", r.p, quote.yes_asks), ("no", 1 - r.p, quote.no_asks)):
            if side == "no" and not strategy.allow_no or not asks:
                continue
            a = asks[0][0] + slip
            if not 0 < a < 1:
                continue
            ev = _ev(q_side, a, fees)
            if best is None or ev > best[2]:
                best = (side, q_side, ev, a)
        if best is None or best[2] < strategy.min_edge:
            rejects["edge_below_threshold"] += 1
            continue
        side, q_side, ev, a = best
        eq = pf.equity
        budget = min(
            strategy.kelly_fraction * _kelly(q_side, a) * eq,
            strategy.max_market_frac * eq - pf.market_exposure(r.ticker),
            strategy.max_cluster_frac * eq - cluster_cost[r.cluster],
            strategy.max_total_exposure_frac * eq - pf.exposure,
            pf.cash,
        )
        unit = a + fees.per_contract(a) + 0.01  # +1c headroom for per-fill fee rounding
        qty = math.floor(max(budget, 0) / unit)
        if qty < 1:
            rejects["size_below_one_contract"] += 1
            continue
        limit = q_side - strategy.min_edge - fees.per_contract(a)
        fr = simulate_buy(quote, side, qty, limit, execution, ts)
        order_id += 1
        orders.append({"order_id": order_id, "ts": ts, "ticker": r.ticker, "side": side,
                       "limit_price": round(limit, 4), "qty_requested": qty,
                       "qty_filled": fr.qty, "p_model": r.p, "ev_per_contract": round(ev, 4),
                       "status": "filled" if fr.qty >= qty else ("partial" if fr.qty else "unfilled"),
                       "reason": fr.reason, "quote_source": quote.source})
        mark = quote.yes_bid if side == "yes" else quote.no_bid
        for px, q in fr.levels:
            fee = fees.fee(q, px)
            if q * px + fee > pf.cash:
                q = math.floor(pf.cash / (px + fees.per_contract(px) + 0.01))
                if q < 1:
                    break
                fee = fees.fee(q, px)
            pf.buy(r.ticker, side, q, px, fee, mark, cluster=r.cluster, category=r.category)
            cluster_cost[r.cluster] += q * px
            fills.append({"order_id": order_id, "ts": ts, "ticker": r.ticker, "side": side,
                          "price": px, "qty": q, "fee": fee,
                          "liquidity_uncertain": fr.liquidity_uncertain})
    if last_ts is not None:
        pf.snapshot(last_ts)
    return _result(pf, signals, resolutions, orders, fills, rejects, strategy, execution)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float(((peak - equity) / peak).max())


def _result(pf, signals, resolutions, orders, fills, rejects, strategy, execution):
    run_id = uuid.uuid4().hex[:12]
    pos = pd.DataFrame([asdict(p) for p in pf.positions.values()])
    eqf = pf.equity_frame()
    orders_df, fills_df = pd.DataFrame(orders), pd.DataFrame(fills)
    settled = pos[pos["settled"]] if len(pos) else pos
    if len(settled):
        settled = settled.assign(gross=settled["payout"] - settled["cost"],
                                 net=settled["payout"] - settled["cost"] - settled["fees"])
    res = resolutions.set_index("ticker")["result"]
    sig = signals[signals["ticker"].isin(res.index)]
    y = (res.loc[sig["ticker"]] == "yes").to_numpy(dtype=float)
    traded = sig["ticker"].isin(set(fills_df["ticker"])) if len(fills_df) else sig["ticker"] == ""
    summary = {
        "banner": BANNER,
        "initial_cash": pf.initial,
        "final_equity": round(pf.equity, 4),
        "net_pnl": round(pf.equity - pf.initial, 4),
        "return": (pf.equity - pf.initial) / pf.initial,
        "gross_pnl_settled": round(float(settled["gross"].sum()) if len(settled) else 0.0, 4),
        "net_pnl_settled": round(float(settled["net"].sum()) if len(settled) else 0.0, 4),
        "fees": round(pf.fees_paid, 4),
        "max_drawdown": _max_drawdown(eqf["equity"]) if len(eqf) else 0.0,
        "n_signals": len(signals),
        "n_orders": len(orders_df),
        "n_fills": len(fills_df),
        "n_positions": len(pos),
        "open_positions_at_end": int((~pos["settled"]).sum()) if len(pos) else 0,
        "contracts": float(fills_df["qty"].sum()) if len(fills_df) else 0.0,
        "uncertain_fill_share": float(fills_df["liquidity_uncertain"].mean()) if len(fills_df) else 0.0,
        "partial_or_unfilled_orders": int((orders_df["status"] != "filled").sum()) if len(orders_df) else 0,
        "rejections": dict(rejects),
        "peak_exposure": float(eqf["exposure"].max()) if len(eqf) else 0.0,
        "hit_rate": float((settled["payout"] > 0).mean()) if len(settled) else None,
        "brier_all_signals": brier(sig["p"], y) if len(sig) else None,
        "brier_traded": brier(sig["p"][traded.to_numpy()], y[traded.to_numpy()])
        if traded.any() else None,
        "net_pnl_ci95": _pnl_ci(settled),
    }
    return BacktestResult(run_id, summary, orders_df, fills_df, pos, eqf,
                          {"strategy": asdict(strategy), "execution": {
                              k: str(v) for k, v in asdict(execution).items()}})


def _pnl_ci(settled: pd.DataFrame, n_boot: int = 2000, seed: int = 0):
    """Cluster bootstrap (by event) of total settled net P&L. None if too few clusters."""
    if len(settled) == 0 or settled["cluster"].nunique() < 10:
        return None
    by = settled.groupby("cluster")["net"].sum().to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.array([by[rng.integers(0, len(by), len(by))].sum() for _ in range(n_boot)])
    return [round(float(np.quantile(boots, 0.025)), 4), round(float(np.quantile(boots, 0.975)), 4)]


# ---------------------------------------------------------------- inputs and persistence

def signals_from_walk_forward(result, model: str, ds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample signals for one model, plus resolutions for those markets."""
    pr = result.predictions[result.predictions["model"] == model][["ticker", "p"]]
    if pr.empty:
        raise ValueError(f"no walk-forward predictions for model {model!r}")
    meta = ds.meta.set_index("ticker")
    sig = pr.join(meta[["decision_ts", "cluster", "category", "series_ticker"]], on="ticker")
    y = pd.Series(ds.y, index=ds.meta["ticker"])
    res = pd.DataFrame({"ticker": sig["ticker"].to_numpy(),
                        "resolve_ts": meta.loc[sig["ticker"], "label_ts"].reset_index(drop=True),
                        "result": np.where(y.loc[sig["ticker"]].values == 1.0, "yes", "no")})
    return sig.sort_values("decision_ts").reset_index(drop=True), res


def fee_schedules(engine: Engine, stress: float = 1.0) -> dict[str, FeeSchedule]:
    with engine.connect() as c:
        rows = c.execute(select(mkt_series.c.series_ticker, mkt_series.c.fee_type,
                                mkt_series.c.fee_multiplier)).all()
    return {r.series_ticker: FeeSchedule.from_series(r.fee_type, r.fee_multiplier, stress)
            for r in rows}


def save_run(sim_engine: Engine, res: BacktestResult, kind: str = "backtest",
             model_version: str | None = None, holdout_hash: str | None = None) -> str:
    """Persist a run to the simulation database only."""
    with sim_engine.begin() as c:
        c.execute(sim_runs.insert().values(
            run_id=res.run_id, created_ts=utcnow(), kind=kind, config_json=res.config,
            model_version=model_version, holdout_hash=holdout_hash,
            data_cutoff_ts=res.equity["ts"].max() if len(res.equity) else None,
            summary_json=_jsonable(res.summary)))
        id_map = {}
        for o in res.orders.to_dict("records"):
            r = c.execute(sim_orders.insert().values(
                run_id=res.run_id, ts=o["ts"], ticker=o["ticker"], side=o["side"],
                limit_price=o["limit_price"], qty_requested=o["qty_requested"],
                qty_filled=o["qty_filled"], status=o["status"], reason=o["reason"]))
            id_map[o["order_id"]] = r.inserted_primary_key[0]
        if len(res.fills):
            c.execute(sim_fills.insert(), [{
                "order_id": id_map[f["order_id"]], "run_id": res.run_id, "ts": f["ts"],
                "ticker": f["ticker"], "side": f["side"], "price": f["price"], "qty": f["qty"],
                "fee": f["fee"], "liquidity_uncertain": bool(f["liquidity_uncertain"])}
                for f in res.fills.to_dict("records")])
        if len(res.positions):
            c.execute(sim_positions.insert(), [{
                "run_id": res.run_id, "ticker": p["ticker"], "side": p["side"], "qty": p["qty"],
                "cost_basis": p["cost"], "fees_paid": p["fees"], "settled": bool(p["settled"]),
                "payout": p["payout"]} for p in res.positions.to_dict("records")])
        if len(res.equity):
            c.execute(sim_equity.insert(), [{
                "run_id": res.run_id, "ts": e["ts"], "cash": e["cash"],
                "exposure": e["exposure"], "equity": e["equity"], "gross_pnl": e["gross_pnl"],
                "net_pnl": e["net_pnl"], "fees_cum": e["fees_cum"]}
                for e in res.equity.to_dict("records")])
    return res.run_id


def _jsonable(d):
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in d.items()}


def sensitivity_grid(run, fee_stress=(1.0, 1.5, 2.0, 3.0), slippage=(0.0, 0.01, 0.02, 0.03)):
    """``run(fee_stress, slippage) -> BacktestResult``. Returns one row per combination."""
    rows = []
    for fs in fee_stress:
        for sl in slippage:
            s = run(fs, sl).summary
            rows.append({"fee_stress": fs, "slippage": sl, "net_pnl": s["net_pnl"],
                         "return": s["return"], "fees": s["fees"], "n_fills": s["n_fills"],
                         "max_drawdown": s["max_drawdown"]})
    return pd.DataFrame(rows)
