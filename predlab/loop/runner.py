"""Autonomous research loop. PAPER TRADING — NOT REAL MONEY.

One cycle, for one certified deployment (model version x series x horizon):

  collect   open markets, their candles and order books, recent settlements, and the
            series' external inputs (weather / FRED)
  update    refit the model on every outcome known by now
  forecast  each open market once, when its time to close first falls inside
            [horizon - late_tolerance, horizon] (the same decision point the model was
            trained and certified on); recorded in the hash-chained forecast log together
            with the market-implied probability
  simulate  the paper record is a deterministic replay of the backtest engine over the
            immutable forward-forecast log, with quotes collected at decision time, so the
            paper strategy is exactly the certified one (kappa shrink, entry band, sizing)
  degrade   rolling skill vs the market and paper drawdown; alerts go to research_log
  evidence  paper days, alerts and P&L are written into the registry's gate evidence

Nothing here can place an order. The separate ``executor`` package reads the gate.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, func, select, update

from predlab import BANNER
from predlab.backtest.engine import (
    BacktestResult,
    StrategyConfig,
    fee_schedules,
    run_backtest,
    save_run,
)
from predlab.backtest.execution import MarketData
from predlab.db.schema import (
    forecasts,
    mkt_markets,
    model_registry,
    outcomes,
    research_log,
    sim_equity,
    sim_fills,
    sim_orders,
    sim_positions,
    sim_runs,
)
from predlab.evaluation.metrics import brier, paired_bootstrap_diff
from predlab.features.pointintime import build_dataset, build_live_dataset
from predlab.forecasts import record_forecasts
from predlab.models.registry import ALL_MODELS
from predlab.research_log import log_event
from predlab.timeutil import to_utc, utcnow

COMPONENT = "loop"


@dataclass(frozen=True)
class LoopConfig:
    deployment: str
    late_tolerance: timedelta = timedelta(hours=2)
    paper_cash: float = 1000.0
    degradation_window: int = 100     # most recent resolved forward forecasts
    degradation_min: int = 30         # no skill verdict below this many
    report_dir: str = "reports"


class DeploymentError(RuntimeError):
    pass


def load_deployment(sim: Engine, deployment: str) -> dict:
    with sim.connect() as c:
        row = c.execute(select(model_registry)
                        .where(model_registry.c.model_version == deployment)).mappings().first()
    if row is None:
        raise DeploymentError(f"{deployment} is not certified (run `predlab certify`)")
    if row["status"] == "retired":
        raise DeploymentError(f"{deployment} is retired")
    return dict(row)


def paper_run_id(deployment: str) -> str:
    return f"paper-{deployment}"


def _forward_forecasts(sim: Engine, deployment: str, now) -> pd.DataFrame:
    """First forecast per ticker for this deployment, made at or before ``now``."""
    first = (select(func.min(forecasts.c.id).label("id"))
             .where(forecasts.c.model_version == deployment,
                    forecasts.c.created_ts <= to_utc(now))
             .group_by(forecasts.c.ticker).subquery())
    with sim.connect() as c:
        rows = c.execute(select(forecasts.c.ticker, forecasts.c.created_ts, forecasts.c.p,
                                forecasts.c.market_ref_p)
                         .join(first, first.c.id == forecasts.c.id)).mappings().all()
    df = pd.DataFrame(rows, columns=["ticker", "created_ts", "p", "market_ref_p"])
    df["created_ts"] = pd.to_datetime(df["created_ts"], utc=True)
    return df


def _resolutions(market: Engine, tickers: list[str], now) -> pd.DataFrame:
    """Latest yes/no outcome known at ``now``; resolve time = when it was first recorded."""
    frames = []
    with market.connect() as c:
        for i in range(0, len(tickers), 900):
            q = (select(outcomes.c.ticker, outcomes.c.recorded_at, outcomes.c.result)
                 .where(outcomes.c.ticker.in_(tickers[i:i + 900]),
                        outcomes.c.recorded_at <= to_utc(now),
                        outcomes.c.result.in_(("yes", "no"))))
            frames.append(pd.DataFrame(c.execute(q).mappings().all(),
                                       columns=["ticker", "recorded_at", "result"]))
    df = pd.concat(frames) if frames else pd.DataFrame(columns=["ticker", "recorded_at", "result"])
    if df.empty:
        return pd.DataFrame(columns=["ticker", "resolve_ts", "result"])
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    df = df.sort_values("recorded_at")
    first = df.groupby("ticker")["recorded_at"].first().rename("resolve_ts")
    last = df.groupby("ticker")["result"].last()
    return pd.concat([first, last], axis=1).reset_index()


def _market_meta(market: Engine, tickers: list[str]) -> pd.DataFrame:
    frames = []
    with market.connect() as c:
        for i in range(0, len(tickers), 900):
            q = (select(mkt_markets.c.ticker, mkt_markets.c.event_ticker, mkt_markets.c.category,
                        mkt_markets.c.series_ticker)
                 .where(mkt_markets.c.ticker.in_(tickers[i:i + 900])))
            frames.append(pd.DataFrame(c.execute(q).mappings().all(),
                                       columns=["ticker", "event_ticker", "category",
                                                "series_ticker"]))
    return pd.concat(frames) if frames else pd.DataFrame(
        columns=["ticker", "event_ticker", "category", "series_ticker"])


def replay_paper(market: Engine, sim: Engine, deployment: str, *, now, strategy: StrategyConfig
                 ) -> tuple[BacktestResult | None, pd.DataFrame]:
    """Rebuild the paper record from the forward-forecast log and persist it under
    ``paper-<deployment>`` (replacing the previous rebuild). Returns (result, scored)."""
    fc = _forward_forecasts(sim, deployment, now)
    run_id = paper_run_id(deployment)
    with sim.begin() as c:
        for t in (sim_fills, sim_orders, sim_positions, sim_equity, sim_runs):
            c.execute(delete(t).where(t.c.run_id == run_id))
    if fc.empty:
        return None, fc
    tickers = fc["ticker"].tolist()
    meta = _market_meta(market, tickers).set_index("ticker")
    events = meta["event_ticker"].reindex(fc["ticker"]).to_numpy()
    sig = pd.DataFrame({
        "ticker": fc["ticker"], "decision_ts": fc["created_ts"], "p": fc["p"],
        "cluster": np.where(pd.isna(events), fc["ticker"], events),
        "category": meta["category"].reindex(fc["ticker"]).to_numpy(),
        "series_ticker": meta["series_ticker"].reindex(fc["ticker"]).to_numpy()})
    res = _resolutions(market, tickers, now)
    bt = run_backtest(sig, res, MarketData(market, tickers), strategy=strategy,
                      fee_for=fee_schedules(market))
    bt.run_id = run_id
    save_run(sim, bt, kind="loop", model_version=deployment)
    scored = fc.merge(res[["ticker", "result"]], on="ticker", how="inner")
    scored["y"] = (scored["result"] == "yes").astype(float)
    scored["cluster"] = sig.set_index("ticker").loc[scored["ticker"], "cluster"].to_numpy()
    return bt, scored


def check_degradation(scored: pd.DataFrame, bt: BacktestResult | None, cfg: LoopConfig,
                      strategy: StrategyConfig) -> tuple[list[dict], dict]:
    """Alerts (count against the gate) and diagnostics for the forward record."""
    alerts, diag = [], {"resolved": len(scored)}
    s = scored.dropna(subset=["market_ref_p"]).sort_values("created_ts").tail(cfg.degradation_window)
    diag["scored_vs_market"] = len(s)
    if len(s) >= cfg.degradation_min:
        d = paired_bootstrap_diff(brier, s["p"].to_numpy(), s["market_ref_p"].to_numpy(),
                                  s["y"].to_numpy(), clusters=s["cluster"].to_numpy(), n_boot=500)
        diag["brier_diff_vs_market"] = d.as_dict()
        if d.low > 0:
            alerts.append({"rule": "worse_than_market",
                           "detail": f"rolling Brier diff {d.estimate:.4f} [{d.low:.4f}, {d.high:.4f}]"})
        elif d.estimate >= 0:
            diag["edge_disappeared"] = True    # warning only: not significantly worse
    if bt is not None and strategy.drawdown_halt is not None:
        dd = bt.summary["max_drawdown"]
        diag["paper_max_drawdown"] = dd
        if dd > strategy.drawdown_halt:
            alerts.append({"rule": "paper_drawdown",
                           "detail": f"max drawdown {dd:.1%} > {strategy.drawdown_halt:.0%}"})
    return alerts, diag


def _log_alerts_once_per_day(sim: Engine, deployment: str, alerts: list[dict], now) -> None:
    day = pd.Timestamp(now).strftime("%Y-%m-%d")
    for a in alerts:
        msg = f"alert {deployment} {a['rule']} {day}"
        with sim.connect() as c:
            seen = c.execute(select(research_log.c.id).where(research_log.c.message == msg)).first()
        if seen is None:
            log_event(sim, "degradation", "alert", msg, {"deployment": deployment, **a}, ts=now)


def _paper_stats(sim: Engine, deployment: str) -> tuple[int, int, str | None]:
    """(distinct UTC days with a completed cycle, alerts logged, first cycle time)."""
    with sim.connect() as c:
        ok = [r.ts for r in c.execute(select(research_log.c.ts).where(
            research_log.c.component == COMPONENT,
            research_log.c.message == f"cycle_ok {deployment}"))]
        n_alerts = c.execute(select(func.count()).select_from(research_log).where(
            research_log.c.component == "degradation",
            research_log.c.message.like(f"alert {deployment} %"))).scalar()
    days = {pd.Timestamp(t).strftime("%Y-%m-%d") for t in ok}
    return len(days), int(n_alerts or 0), (str(min(ok)) if ok else None)


def run_cycle(market: Engine, sim: Engine, cfg: LoopConfig, *, now=None,
              collect: Callable[[dict, LoopConfig], Awaitable[None]] | None = None) -> dict:
    """Run one cycle. ``now`` defaults to the wall clock *after* collection, so quotes
    gathered during collection are never later than the decision time."""
    dep = load_deployment(sim, cfg.deployment)
    params = dep["params_json"]
    series, horizon = params["series"], timedelta(hours=params["horizon_h"])
    strategy = StrategyConfig(**{**(dep["validation_json"] or {}).get("strategy", {}),
                                 "initial_cash": cfg.paper_cash})
    if collect is not None:
        try:
            asyncio.run(collect(params, cfg))
        except Exception as e:  # noqa: BLE001 - a failed collection skips the cycle, recorded
            log_event(sim, COMPONENT, "error", f"collect_failed {cfg.deployment}",
                      {"error": f"{type(e).__name__}: {e}"})
            return {"status": "collect_failed", "error": str(e)}
    now = pd.Timestamp(now if now is not None else utcnow())

    train = build_dataset(market, series=series, horizon=horizon, as_of=now)
    model = ALL_MODELS[params["model"]]().fit(train)
    if model.version != params["model_version"]:
        raise DeploymentError(f"model code now yields {model.version}, certified "
                              f"{params['model_version']}; certify the new version")

    done = set(_forward_forecasts(sim, cfg.deployment, now)["ticker"])
    live = build_live_dataset(market, series=series, as_of=now,
                              feature_names=train.feature_names,
                              min_to_close=horizon - cfg.late_tolerance, max_to_close=horizon)
    new = 0
    if len(live):
        live = live.subset(~live.meta["ticker"].isin(done).to_numpy())
    if len(live):
        items = [replace(f, model_version=cfg.deployment) for f in model.forecasts(live)]
        ref = {t: float(m) for t, m in zip(live.meta["ticker"], live.meta["market_mid"])
               if pd.notna(m)}
        new = record_forecasts(sim, items, market_ref=ref, created_ts=now)

    return update_paper_record(market, sim, cfg, dep, now=now, strategy=strategy,
                               new_forecasts=new, prior_forecasts=len(done))


def update_paper_record(market: Engine, sim: Engine, cfg: LoopConfig, dep: dict, *, now,
                        strategy: StrategyConfig, new_forecasts: int,
                        prior_forecasts: int) -> dict:
    """Replay the forward log, check degradation, write gate evidence and the report.
    Shared by every deployment kind (fitted models and the LLM desk)."""
    new, gate = new_forecasts, (dep["validation_json"] or {}).get("gate", {})
    bt, scored = replay_paper(market, sim, cfg.deployment, now=now, strategy=strategy)
    alerts, diag = check_degradation(scored, bt, cfg, strategy)
    _log_alerts_once_per_day(sim, cfg.deployment, alerts, now)
    log_event(sim, COMPONENT, "info", f"cycle_ok {cfg.deployment}",
              {"new_forecasts": new, "alerts": alerts, **diag}, ts=now)
    days, n_alerts, start = _paper_stats(sim, cfg.deployment)
    summary = bt.summary if bt is not None else {}
    gate = {**gate, "paper_days": days, "paper_degradation_alerts": n_alerts,
            "paper_start": start, "paper_forecasts": int(prior_forecasts + new),
            "paper_resolved": diag["resolved"],
            "paper_settled_trades": summary.get("n_settled", 0),
            "paper_net_pnl": summary.get("net_pnl"),
            "paper_net_pnl_ci95": summary.get("net_pnl_ci95"),
            "paper_brier_diff_vs_market": (diag.get("brier_diff_vs_market") or {}).get("estimate"),
            "paper_updated": str(now)}
    validation = {**(dep["validation_json"] or {}), "gate": gate}
    with sim.begin() as c:
        c.execute(update(model_registry).where(model_registry.c.model_version == cfg.deployment)
                  .values(validation_json=json.loads(json.dumps(validation, default=str))))
    out = {"banner": BANNER, "status": "ok", "deployment": cfg.deployment,
           "registry_status": dep["status"], "now": str(now), "new_forecasts": new,
           "alerts": alerts, "diagnostics": diag, "gate": gate,
           "paper": {k: summary.get(k) for k in ("net_pnl", "net_pnl_ci95", "n_fills", "fees",
                                                 "max_drawdown", "open_positions_at_end")}}
    _write_report(cfg, out, now)
    return out


def _write_report(cfg: LoopConfig, out: dict, now) -> None:
    d = Path(cfg.report_dir)
    d.mkdir(parents=True, exist_ok=True)
    safe = cfg.deployment.replace("@", "_").replace("/", "_")
    (d / f"loop-{safe}-latest.json").write_text(json.dumps(out, indent=2, default=str))


def replay(market: Engine, sim: Engine, cfg: LoopConfig, *, start, end,
           step: timedelta = timedelta(hours=1)) -> list[dict]:
    """Run cycles over historical time with no collection (synthetic demo and tests)."""
    t, out = pd.Timestamp(start), []
    while t <= pd.Timestamp(end):
        out.append(run_cycle(market, sim, cfg, now=t))
        t += step
    return out


def run_forever(market: Engine, sim: Engine, cfg: LoopConfig, *, interval: timedelta,
                collect, max_cycles: int | None = None, sleep=None) -> None:
    import time
    sleep = sleep or time.sleep
    n = 0
    while max_cycles is None or n < max_cycles:
        started = datetime.now().timestamp()
        try:
            run_cycle(market, sim, cfg, collect=collect)
        except DeploymentError:
            raise
        except Exception as e:  # noqa: BLE001 - keep looping; the failure is logged
            log_event(sim, COMPONENT, "error", f"cycle_failed {cfg.deployment}",
                      {"error": f"{type(e).__name__}: {e}"})
        n += 1
        if max_cycles is None or n < max_cycles:
            sleep(max(0.0, interval.total_seconds() - (datetime.now().timestamp() - started)))

