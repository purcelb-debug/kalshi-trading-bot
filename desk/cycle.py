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
from desk.meeting import Meeting, providers, run_meeting
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
    return f"desk@{series}-h{horizon_h:g}"


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
                         "horizon_h": horizon_h, "params": {
                             "provider": cfg.provider, "model": cfg.model,
                             "search_provider": cfg.search_provider,
                             "search_model": cfg.search_model,
                             "skeptic_model": cfg.skeptic_model}},
            trained_ts=now, validation_json={
                "strategy": asdict(strat), "gate": gate,
                "reasons": ["LLM desk: no holdout certification is possible"]},
            status="candidate", status_ts=now))
    log_event(sim, "desk", "info", f"{dv} registered as candidate", {"gate": gate})
    return dv


def meetings_today(sim: Engine, now) -> int:
    """Meetings held today by every desk deployment (the cap is shared)."""
    day = pd.Timestamp(now).normalize()
    with sim.connect() as c:
        return int(c.execute(select(func.count()).select_from(desk_meetings).where(
            desk_meetings.c.ts >= to_utc(day))).scalar() or 0)


def meeting_cost(usage: list[dict], cfg: DeskSettings) -> float:
    """Estimated dollars for one meeting from its measured token use."""
    total = 0.0
    for u in usage or []:
        p_in, p_out = cfg.prices.get(u.get("model"), cfg.prices["claude-opus-5-5"])
        total += (u.get("input_tokens", 0) * p_in + u.get("output_tokens", 0) * p_out) / 1e6
        if u.get("search_backend") != "tavily":     # Tavily's free tier costs nothing
            total += u.get("web_searches", 0) * cfg.web_search_usd
    return total


def spend_today(sim: Engine, cfg: DeskSettings, now) -> float:
    """Estimated LLM spend today across every desk deployment (the budget is shared)."""
    day = pd.Timestamp(now).normalize()
    with sim.connect() as c:
        rows = c.execute(select(desk_meetings.c.usage_json).where(
            desk_meetings.c.ts >= to_utc(day))).scalars().all()
    return sum(meeting_cost(u, cfg) for u in rows)


def desk_deployments(sim: Engine) -> list[str]:
    """Registered desk deployments that are still live (not retired)."""
    with sim.connect() as c:
        return [r[0] for r in c.execute(select(model_registry.c.model_version).where(
            model_registry.c.model_name == "desk", model_registry.c.status != "retired")
            .order_by(model_registry.c.model_version))]


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


def is_broad(params: dict) -> bool:
    """``desk@ALL``: scans every open market and trades real money through the executor."""
    return params.get("series") == "ALL"


def _save_transcript(cfg: DeskSettings, deployment: str, now, text: str, *,
                     real_money: bool = False) -> None:
    d = Path(cfg.report_dir) / "desk-meetings"
    d.mkdir(parents=True, exist_ok=True)
    tag = "REAL MONEY" if real_money else "PAPER"
    safe = deployment.replace("@", "_")
    (d / f"{pd.Timestamp(now):%Y%m%dT%H%M%S}-{safe}.txt").write_text(
        f"[DESK MEETING] {deployment}  {pd.Timestamp(now):%Y-%m-%d %H:%M} UTC  {tag}\n\n{text}",
        encoding="utf-8")


def run_desk_cycle(market: Engine, sim: Engine, cfg: DeskSettings, loop: LoopConfig, *,
                   clients: dict | None = None, now=None, scan_fetch=None,
                   allow_meeting: bool = True) -> dict:
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
    broad = is_broad(params)
    if broad:
        from desk.scan import scan_table, scan_text
        t = scan_table(cfg, exclude=_shown(sim, loop.deployment), now=now, fetch=scan_fetch)
    else:
        t = market_table(market, sim, cfg, series=params["series"],
                         horizon=timedelta(hours=params["horizon_h"]),
                         late=loop.late_tolerance, now=now)
        if len(t):
            t = t[~t["ticker"].isin(_shown(sim, loop.deployment))]
    meeting, recorded, status = None, 0, "no_new_markets"
    spent = spend_today(sim, cfg, now) if len(t) else 0.0
    if len(t) and (meetings_today(sim, now) >= cfg.max_meetings_per_day
                   or spent >= cfg.daily_budget_usd):
        status = "meeting_cap_reached"
        log_event(sim, "desk", "warning", f"meeting cap reached {loop.deployment}",
                  {"cap": cfg.max_meetings_per_day, "budget_usd": cfg.daily_budget_usd,
                   "spent_usd": round(spent, 4), "waiting": t["ticker"].tolist()}, ts=now)
    elif len(t) and not allow_meeting:
        status = "deferred"          # this run already held its meetings; next run
    elif len(t):
        if clients is None:
            from desk.llm import make_client
            clients = {p: make_client(cfg, p) for p in providers(cfg)}
        meeting = run_meeting(clients, cfg, scan_text(t, now) if broad else opening_text(t, now))
        decisions, notes = parse_trades(meeting.lead_text,
                                        dict(zip(t["ticker"], t["event_ticker"])),
                                        max_buys=cfg.max_buys, p_floor=cfg.p_floor,
                                        exit_gap=cfg.exit_min_gap)
        mids = dict(zip(t["ticker"], t["market_mid"]))
        items = [Forecast(ticker=d.ticker, model_version=loop.deployment, p=d.p_yes,
                          ci_low=None, ci_high=None, input_max_ts=now, decision_ts=now,
                          dq_flags=(), explanation={"desk": d.reason, "side": d.side,
                                                    "take_profit": d.take_profit,
                                                    "stop": d.stop})
                 for d in decisions]
        ref = {k: float(v) for k, v in mids.items() if pd.notna(v)}
        recorded = record_forecasts(sim, items, market_ref=ref, created_ts=now) if items else 0
        text = meeting.text() + "\n\n[SYSTEM]\n" + "\n".join(
            [f"recorded BUY {d.ticker} p_yes={d.p_yes:.2f} take_profit={d.take_profit} "
             f"stop={d.stop} (sized and placed by code)" for d in decisions]
            + ([f"ABORTED: {meeting.aborted}; no trades"] if meeting.aborted else [])
            + notes + ([] if decisions or notes else ["no trades"])
            + [f"est. cost ${meeting_cost(meeting.usage, cfg):.3f}"])
        with sim.begin() as c:
            c.execute(desk_meetings.insert().values(
                deployment=loop.deployment, ts=now.to_pydatetime(),
                tickers_json=t["ticker"].tolist(),
                decisions_json=[asdict(d) for d in decisions], notes_json=notes,
                usage_json=meeting.usage, transcript=text))
        _save_transcript(cfg, loop.deployment, now, text, real_money=broad)
        status = "meeting_aborted" if meeting.aborted else "meeting_held"
    desk_out = {"status": status, "recorded": recorded, "usage": meeting.usage if meeting else []}
    if broad:
        # Scanned markets aren't in the research database, so there is no paper replay:
        # this deployment's record is the executor's ledger (real fills and settlements).
        return {"status": "ok", "deployment": loop.deployment, "gate": None, "desk": desk_out}
    with sim.connect() as c:
        prior = int(c.execute(select(func.count(func.distinct(forecasts.c.ticker))).where(
            forecasts.c.model_version == loop.deployment)).scalar() or 0) - recorded
    out = update_paper_record(market, sim, loop, dep, now=now, strategy=strategy,
                              new_forecasts=recorded, prior_forecasts=prior)
    out["desk"] = desk_out
    return out


__all__ = ["Meeting", "desk_deployment", "desk_deployments", "register_desk", "run_desk_cycle"]
