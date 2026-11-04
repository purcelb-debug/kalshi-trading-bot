"""One execution cycle: forecasts -> fresh order books -> sized IOC limit buys -> audit.

Decision rules are the backtest's (edge after fees, fractional Kelly, caps), so live
behaviour matches what was simulated. Every order passes ``pre_trade_check`` immediately
before submission. In ``live`` mode the go-live gate must be open for the model version.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, func, select

from executor.client import KalshiTradingClient, TradingError
from executor.config import BASE_URLS, ExecutorSettings
from executor.exits import exit_levels, plan_exit
from executor.gate import certified_shrink, load_gate
from executor.safety import AuditLog, State, pre_trade_check, startup_checks
from predlab.backtest.decision import Plan, plan_order
from predlab.backtest.execution import MarketData, Quote
from predlab.backtest.fees import FeeSchedule
from predlab.data.collectors.kalshi import norm_orderbook
from predlab.data.http import KALSHI_READ_PATHS, ReadOnlyClient
from predlab.db.schema import forecasts
from predlab.timeutil import utcnow


def latest_forecasts(sim_engine: Engine, model_version: str, max_age_hours: float) -> pd.DataFrame:
    """Most recent forecast per ticker for one model, no older than ``max_age_hours``."""
    cutoff = utcnow() - pd.Timedelta(hours=max_age_hours)
    sub = (select(forecasts.c.ticker, func.max(forecasts.c.id).label("id"))
           .where(forecasts.c.model_version == model_version, forecasts.c.created_ts >= cutoff)
           .group_by(forecasts.c.ticker).subquery())
    with sim_engine.connect() as c:
        rows = c.execute(select(forecasts.c.ticker, forecasts.c.p, forecasts.c.created_ts)
                         .join(sub, sub.c.id == forecasts.c.id)).mappings().all()
    return pd.DataFrame(rows, columns=["ticker", "p", "created_ts"])


async def fetch_quotes(base_url: str, tickers: list[str]) -> dict[str, Quote]:
    out = {}
    async with ReadOnlyClient(base_url, allowed_paths=KALSHI_READ_PATHS, reads_per_sec=5) as c:
        for t in tickers:
            ts = utcnow()
            rows = norm_orderbook(await c.get(f"/markets/{t}/orderbook"), t, ts)
            df = pd.DataFrame(rows, columns=["ticker", "snapshot_ts", "side", "price", "size"])
            out[t] = MarketData._book_quote(pd.Timestamp(ts), df)
    return out


def run_cycle(cfg: ExecutorSettings, sim_engine: Engine, model_version: str, *,
              client: KalshiTradingClient | None = None,
              quotes: dict[str, Quote] | None = None,
              confirm: Callable[[Plan], bool] | None = None,
              fees: FeeSchedule | None = None, forecast_max_age_h: float = 6.0) -> list[dict]:
    data = Path(cfg.data_dir)
    state_path = data / f"state_{cfg.mode}.json"
    state = State.load(state_path)
    audit = AuditLog(data / f"audit_{cfg.mode}.jsonl")
    fees = fees or FeeSchedule()
    results: list[dict] = []

    chk = startup_checks(cfg, state)
    if not chk.ok:
        audit.write("cycle_refused", reason=chk.reason, mode=cfg.mode)
        return [{"status": "refused", "reason": chk.reason}]
    if cfg.mode == "live":
        gate = load_gate(sim_engine, model_version, cfg.gate_paper_weeks,
                         cfg.gate_min_paper_trades)
        if not gate.open and cfg.gate_overridden:
            # Owner chose to trade before the evidence exists; limits still apply.
            audit.write("gate_overridden", model_version=model_version,
                        reasons=list(gate.reasons))
        elif not gate.open:
            audit.write("gate_closed", model_version=model_version, reasons=list(gate.reasons))
            return [{"status": "gate_closed", "reasons": list(gate.reasons)}]
    if cfg.mode != "dry_run" and client is None:
        raise ValueError("demo/live mode needs an authenticated client")

    # Reconcile with the exchange first: its balance and positions are the truth.
    if client is not None:
        cash, port_value = client.balance_usd()
        positions = client.positions()
    else:
        cash, port_value, positions = 1000.0, 0.0, []      # dry run: notional equity
    exposure = {p["ticker"]: p["exposure_usd"] for p in positions}
    total_value = cash + port_value
    state.roll_day(total_value)
    audit.write("cycle_start", mode=cfg.mode, model_version=model_version, cash=cash,
                portfolio_value=port_value, open_exposure=sum(exposure.values()))

    # Exits first: close positions this deployment opened whose take-profit or stop was hit.
    levels = exit_levels(sim_engine, model_version)
    held = [p for p in positions if p["ticker"] in levels]
    if held and client is not None:
        books = quotes if quotes is not None else asyncio.run(fetch_quotes(
            BASE_URLS["demo" if cfg.mode == "demo" else "live"], [p["ticker"] for p in held]))
        for p in held:
            q = books.get(p["ticker"])
            if q is not None and (utcnow() - q.ts).total_seconds() > cfg.max_quote_age_s:
                q = None
            ex = plan_exit(p, levels[p["ticker"]], q)
            if ex is None:
                continue
            audit.write("exit_submit", mode=cfg.mode, plan=ex.__dict__)
            try:
                ack = client.place_ioc_exit(ex.ticker, ex.held, ex.count, ex.min_price)
            except (TradingError, OSError) as e:
                state.consecutive_errors += 1
                audit.write("exit_error", ticker=ex.ticker, error=str(e))
                results.append({"ticker": ex.ticker, "status": "exit_error", "reason": str(e)})
                if state.consecutive_errors >= cfg.max_consecutive_errors:
                    state.halted = f"{state.consecutive_errors} consecutive order errors"
                    break
                continue
            state.consecutive_errors = 0
            state.orders_today += 1
            audit.write("exit_ack", reason=ex.reason, **ack.__dict__)
            if ack.fill_count:
                state.closed = sorted({*state.closed, ex.ticker})
                cash += ack.fill_count * ex.min_price
                if ack.fill_count >= ex.count:
                    exposure.pop(ex.ticker, None)
            results.append({"ticker": ex.ticker, "status": "exited" if ack.fill_count
                            else "exit_unfilled", "reason": ex.reason, "filled": ack.fill_count})
        if state.halted:
            state.save(state_path)
            audit.write("cycle_end", results=len(results), halted=state.halted)
            return results

    up = total_value - (state.day_start_value_usd or total_value)
    if cfg.daily_profit_target_usd is not None and up >= cfg.daily_profit_target_usd:
        # Goal met for the day: keep the gain, open nothing new until tomorrow.
        audit.write("profit_target_reached", day=state.day, up_usd=round(up, 2),
                    target_usd=cfg.daily_profit_target_usd)
        state.save(state_path)
        return results + [{"status": "target_reached", "up_usd": round(up, 2)}]

    fc = latest_forecasts(sim_engine, model_version, forecast_max_age_h)
    if fc.empty:
        audit.write("no_forecasts", model_version=model_version)
        state.save(state_path)
        return results + [{"status": "no_forecasts"}]
    if quotes is None:
        quotes = asyncio.run(fetch_quotes(BASE_URLS["demo" if cfg.mode == "demo" else "live"],
                                          fc["ticker"].tolist()))
    equity = min(total_value, cfg.bankroll_cap_usd or total_value)
    # Trade exactly as certified: the certified kappa, or the configured shrink if that is
    # more conservative (closer to the market).
    kappa = certified_shrink(sim_engine, model_version)
    shrink = cfg.shrink if kappa is None else min(cfg.shrink, kappa)
    for r in fc.itertuples():
        if r.ticker in state.closed:
            results.append({"ticker": r.ticker, "status": "skipped", "reason": "exited earlier"})
            continue
        if exposure.get(r.ticker, 0.0) > 0:
            # One decision per market, as in the backtest: never add to a held position.
            results.append({"ticker": r.ticker, "status": "skipped", "reason": "already held"})
            continue
        q = quotes.get(r.ticker)
        if q is None or (utcnow() - q.ts).total_seconds() > cfg.max_quote_age_s:
            results.append({"ticker": r.ticker, "status": "skipped", "reason": "no fresh book"})
            continue
        plan, why = plan_order(r.ticker, r.p, q, fees, equity=equity, cash=cash,
                               min_edge=cfg.min_edge, kelly_fraction=cfg.kelly_fraction,
                               max_order_usd=cfg.max_order_usd, shrink=shrink,
                               min_entry_price=cfg.min_entry_price,
                               max_entry_price=cfg.max_entry_price,
                               min_order_usd=cfg.min_order_usd)
        if plan is None:
            results.append({"ticker": r.ticker, "status": "skipped", "reason": why})
            continue
        chk = pre_trade_check(cfg, state, cost_usd=plan.cost_usd, cash_usd=cash,
                              open_exposure_usd=sum(exposure.values()),
                              market_exposure_usd=exposure.get(r.ticker, 0.0),
                              total_value_usd=total_value)
        if not chk.ok:
            if "daily loss stop" in chk.reason:
                state.halted = chk.reason
            audit.write("order_blocked", ticker=r.ticker, reason=chk.reason, plan=plan.__dict__)
            results.append({"ticker": r.ticker, "status": "blocked", "reason": chk.reason})
            if state.halted or "kill switch" in chk.reason:
                break
            continue
        if cfg.mode == "dry_run":
            audit.write("dry_run_order", plan=plan.__dict__)
            results.append({"ticker": r.ticker, "status": "dry_run", **plan.__dict__})
            continue
        if cfg.mode == "live" and cfg.attended and not (confirm and confirm(plan)):
            audit.write("order_declined_by_user", plan=plan.__dict__)
            results.append({"ticker": r.ticker, "status": "declined"})
            continue
        audit.write("order_submit", mode=cfg.mode, plan=plan.__dict__)
        try:
            ack = client.place_ioc_buy(r.ticker, plan.outcome, plan.count, plan.max_price)
        except (TradingError, OSError) as e:
            if isinstance(e, TradingError) and "insufficient_balance" in str(e):
                # Kalshi rejected it outright (nothing reached the book): the cash is in
                # open positions. Skip buying for the rest of this cycle, but don't count
                # it toward the error halt, or exits would stop too.
                audit.write("order_blocked", ticker=r.ticker,
                            reason="insufficient balance on Kalshi", plan=plan.__dict__)
                results.append({"ticker": r.ticker, "status": "blocked",
                                "reason": "insufficient balance on Kalshi"})
                break
            # Unknown whether it reached the exchange: never auto-resubmit; the next
            # cycle reconciles positions from Kalshi before doing anything else.
            state.consecutive_errors += 1
            audit.write("order_error", ticker=r.ticker, error=str(e))
            results.append({"ticker": r.ticker, "status": "error", "reason": str(e)})
            if state.consecutive_errors >= cfg.max_consecutive_errors:
                state.halted = f"{state.consecutive_errors} consecutive order errors"
                break
            continue
        state.consecutive_errors = 0
        state.orders_today += 1
        audit.write("order_ack", **ack.__dict__)
        spent = ack.fill_count * plan.max_price
        cash -= spent
        exposure[r.ticker] = exposure.get(r.ticker, 0.0) + spent
        results.append({"ticker": r.ticker, "status": "filled" if ack.fill_count else "unfilled",
                        "order_id": ack.order_id, "filled": ack.fill_count})
    state.save(state_path)
    audit.write("cycle_end", results=len(results), halted=state.halted)
    return results
