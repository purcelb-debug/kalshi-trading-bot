"""Desk deployment: registration and one cycle (table -> meeting -> validated BUYs ->
forecast log -> the same deterministic paper replay and gate evidence as any model).

Only BUY decisions become forecasts. WAIT and SKIP record nothing, so a stale mid can never
turn a "no" into a trade. Each market is put in front of the desk once, at its decision
point, matching how fitted deployments forecast.
"""

from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import JSON, Column, Engine, Integer, MetaData, String, Table, Text, func, select

from desk.config import DeskSettings
from desk.meeting import Meeting, run_meeting
from desk.validate import parse_trades
from predlab.backtest.engine import StrategyConfig
from predlab.db.schema import UTCDateTime, forecasts, mkt_markets, model_registry
from predlab.features.pointintime import build_live_dataset
from predlab.forecasts import record_forecasts
from predlab.loop.runner import LoopConfig, load_deployment, update_paper_record
from predlab.models.base import Forecast
from predlab.research_log import log_event
from predlab.timeutil import to_utc, utcnow

DESK_MD = MetaData()
desk_meetings = Table(
    "desk_meetings", DESK_MD,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("deployment", String, nullable=False, index=True),
    Column("ts", UTCDateTime, nullable=False, index=True),
    Column("tickers_json", JSON),         # every market shown to the desk
    Column("decisions_json", JSON),       # validated BUYs recorded as forecasts
    Column("notes_json", JSON),           # what validation dropped, and why
    Column("usage_json", JSON),
    Column("transcript", Text),
)


def desk_deployment(series: str, horizon_h: float) -> str:
    return f"desk-claude@{series}-h{horizon_h:g}"


def init_desk_db(sim: Engine) -> None:
    DESK_MD.create_all(sim)


def register_desk(sim: Engine, cfg: DeskSettings, *, series: str, horizon_h: float = 30.0,
                  category: str | None = None) -> str:
    """Admit the desk to forward paper trading. It has no holdout evidence and never can
    (an LLM cannot be replayed point-in-time on history), so its go-live gate stays closed."""
    dv = desk_deployment(series, horizon_h)
    with sim.connect() as c:
        if c.execute(select(model_registry.c.model_version)
                     .where(model_registry.c.model_version == dv)).first():
            raise ValueError(f"{dv} is already registered")
    now = utcnow()
    strat = StrategyConfig(shrink=cfg.shrink)
    gate = {"candidate": True, "desk": True, "shrink_kappa": cfg.shrink,
            "holdout_brier_diff_vs_market_ci_high": None, "holdout_net_pnl_ci_low": None,
            "paper_days": 0, "paper_degradation_alerts": 0, "paper_start": None}
    with sim.begin() as c:
        c.execute(model_registry.insert().values(
            model_version=dv, model_name="desk", category=category, params_hash="desk",
            params_json={"model": "desk", "model_version": dv, "series": series,
                         "horizon_h": horizon_h, "params": {"model": cfg.model,
                                                            "skeptic_model": cfg.skeptic_model}},
            trained_ts=now, validation_json={
                "strategy": asdict(strat), "gate": gate,
                "reasons": ["LLM desk: no holdout certification is possible"]},
            status="candidate", status_ts=now))
    log_event(sim, "desk", "info", f"{dv} registered as candidate", {"gate": gate})
    return dv


def meetings_today(sim: Engine, deployment: str, now) -> int:
    day = pd.Timestamp(now).normalize()
    with sim.connect() as c:
        return int(c.execute(select(func.count()).select_from(desk_meetings).where(
            desk_meetings.c.deployment == deployment,
            desk_meetings.c.ts >= to_utc(day))).scalar() or 0)


def _shown(sim: Engine, deployment: str) -> set[str]:
    with sim.connect() as c:
        return {t for (ts,) in c.execute(select(desk_meetings.c.tickers_json).where(
            desk_meetings.c.deployment == deployment)) for t in (ts or [])}


def market_table(market: Engine, sim: Engine, cfg: DeskSettings, *, series: str,
                 horizon: timedelta, late: timedelta, now) -> pd.DataFrame:
    """Open markets at their decision point, with mid, spread and the quant model's p."""
    live = build_live_dataset(market, series=series, as_of=now, feature_names=[],
                              min_to_close=horizon - late, max_to_close=horizon)
    if not len(live):
        return pd.DataFrame()
    tickers = live.meta["ticker"].tolist()
    with market.connect() as c:
        m = pd.DataFrame(c.execute(select(
            mkt_markets.c.ticker, mkt_markets.c.event_ticker, mkt_markets.c.title,
            mkt_markets.c.strike_type, mkt_markets.c.floor_strike, mkt_markets.c.cap_strike,
            mkt_markets.c.close_time, mkt_markets.c.yes_bid, mkt_markets.c.yes_ask)
            .where(mkt_markets.c.ticker.in_(tickers))).mappings().all())
    t = live.meta[["ticker", "market_mid"]].merge(m, on="ticker", how="left")
    t["hours_to_close"] = (pd.to_datetime(t["close_time"], utc=True) - pd.Timestamp(now)
                           ).dt.total_seconds() / 3600
    t["spread"] = t["yes_ask"] - t["yes_bid"]
    t["p_model"] = float("nan")
    if cfg.quant_deployment:
        with sim.connect() as c:
            q = dict(c.execute(select(forecasts.c.ticker, forecasts.c.p).where(
                forecasts.c.model_version == cfg.quant_deployment,
                forecasts.c.ticker.in_(tickers), forecasts.c.created_ts <= to_utc(now))
                .order_by(forecasts.c.id)).all())
        t["p_model"] = t["ticker"].map(q)
    t["event_ticker"] = t["event_ticker"].fillna(t["ticker"])
    return t.sort_values("ticker").reset_index(drop=True)


def _fmt(v, spec=".2f") -> str:
    return "n/a" if pd.isna(v) else format(v, spec)


def opening_text(t: pd.DataFrame, now) -> str:
    rows = ["ticker | event | strike | hours to close | mid | spread | p_model"]
    for r in t.itertuples():
        strike = f"{r.strike_type or ''} {_fmt(r.floor_strike, 'g')}-{_fmt(r.cap_strike, 'g')}"
        rows.append(f"{r.ticker} | {r.event_ticker} | {strike} | {_fmt(r.hours_to_close, '.1f')}"
                    f" | {_fmt(r.market_mid)} | {_fmt(r.spread)} | {_fmt(r.p_model)}")
    titles = {e: tt for e, tt in zip(t["event_ticker"], t["title"]) if isinstance(tt, str)}
    return (f"UTC time: {pd.Timestamp(now):%Y-%m-%d %H:%M}. PAPER TRADING ONLY.\n"
            f"Events: " + "; ".join(f"{e}: {tt}" for e, tt in titles.items()) +
            "\n\nMARKETS\n" + "\n".join(rows))


def _save_transcript(cfg: DeskSettings, deployment: str, now, text: str) -> None:
    d = Path(cfg.report_dir) / "desk-meetings"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{pd.Timestamp(now):%Y%m%dT%H%M%S}.txt").write_text(
        f"[DESK MEETING] {deployment}  {pd.Timestamp(now):%Y-%m-%d %H:%M} UTC  PAPER\n\n{text}",
        encoding="utf-8")


def run_desk_cycle(market: Engine, sim: Engine, cfg: DeskSettings, loop: LoopConfig, *,
                   client=None, now=None) -> dict:
    """One cycle. The meeting runs only when markets newly reach their decision point and
    the daily meeting cap allows; the paper record is refreshed every cycle regardless."""
    init_desk_db(sim)
    dep = load_deployment(sim, loop.deployment)
    params = dep["params_json"]
    if params.get("model") != "desk":
        raise ValueError(f"{loop.deployment} is not a desk deployment")
    strategy = StrategyConfig(**{**(dep["validation_json"] or {}).get("strategy", {}),
                                 "initial_cash": loop.paper_cash})
    now = pd.Timestamp(now if now is not None else utcnow())
    t = market_table(market, sim, cfg, series=params["series"],
                     horizon=timedelta(hours=params["horizon_h"]), late=loop.late_tolerance,
                     now=now)
    if len(t):
        t = t[~t["ticker"].isin(_shown(sim, loop.deployment))]
    meeting, recorded, status = None, 0, "no_new_markets"
    if len(t) and meetings_today(sim, loop.deployment, now) >= cfg.max_meetings_per_day:
        status = "meeting_cap_reached"
        log_event(sim, "desk", "warning", f"meeting cap reached {loop.deployment}",
                  {"cap": cfg.max_meetings_per_day, "waiting": t["ticker"].tolist()}, ts=now)
    elif len(t):
        if client is None:
            from desk.llm import make_client
            client = make_client()
        meeting = run_meeting(client, cfg, opening_text(t, now))
        decisions, notes = parse_trades(meeting.lead_text,
                                        dict(zip(t["ticker"], t["event_ticker"])),
                                        max_buys=cfg.max_buys, p_floor=cfg.p_floor)
        mids = dict(zip(t["ticker"], t["market_mid"]))
        items = [Forecast(ticker=d.ticker, model_version=loop.deployment, p=d.p_yes,
                          ci_low=None, ci_high=None, input_max_ts=now, decision_ts=now,
                          dq_flags=(), explanation={"desk": d.reason, "side": d.side})
                 for d in decisions]
        ref = {k: float(v) for k, v in mids.items() if pd.notna(v)}
        recorded = record_forecasts(sim, items, market_ref=ref, created_ts=now) if items else 0
        text = meeting.text() + "\n\n[SYSTEM]\n" + "\n".join(
            [f"recorded BUY {d.ticker} p_yes={d.p_yes:.2f} (sized by code)" for d in decisions]
            + notes + ([] if decisions or notes else ["no trades"]))
        with sim.begin() as c:
            c.execute(desk_meetings.insert().values(
                deployment=loop.deployment, ts=now.to_pydatetime(),
                tickers_json=t["ticker"].tolist(),
                decisions_json=[asdict(d) for d in decisions], notes_json=notes,
                usage_json=meeting.usage, transcript=text))
        _save_transcript(cfg, loop.deployment, now, text)
        status = "meeting_held"
    with sim.connect() as c:
        prior = int(c.execute(select(func.count(func.distinct(forecasts.c.ticker))).where(
            forecasts.c.model_version == loop.deployment)).scalar() or 0) - recorded
    out = update_paper_record(market, sim, loop, dep, now=now, strategy=strategy,
                              new_forecasts=recorded, prior_forecasts=prior)
    out["desk"] = {"status": status, "recorded": recorded,
                   "usage": meeting.usage if meeting else []}
    return out


__all__ = ["Meeting", "desk_deployment", "register_desk", "run_desk_cycle"]
