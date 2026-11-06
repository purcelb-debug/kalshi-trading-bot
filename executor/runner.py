"""One execution cycle: forecasts -> fresh order books -> sized limit buys -> audit.

Buys are IOC at the ask (``order_style=taker``) or resting post-only orders 1c above the
bid (``maker``), which are kept while their side and price still hold and cancelled when
the edge goes; Kalshi expires them after ``maker_ttl_min`` regardless.

Decision rules are the backtest's (edge after fees, fractional Kelly, caps), so live
behaviour matches what was simulated. Every order passes ``pre_trade_check`` immediately
before submission. In ``live`` mode the go-live gate must be open for the model version.
"""

import asyncio
import json
import math
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, func, select

from executor.client import KalshiTradingClient, TradingError
from executor.config import BASE_URLS, ExecutorSettings
from executor.exits import exit_levels, plan_exit
from executor.gate import certified_shrink, load_gate
from executor.safety import AuditLog, Busy, State, exclusive, pre_trade_check, startup_checks
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
        rows = c.execute(select(forecasts.c.ticker, forecasts.c.p, forecasts.c.created_ts,
                                forecasts.c.explanation)
                         .join(sub, sub.c.id == forecasts.c.id)).mappings().all()
    df = pd.DataFrame(rows, columns=["ticker", "p", "created_ts", "explanation"])
    # The desk withdrew its BUY on a re-review: nothing to trade.
    withdrawn = df["explanation"].map(
        lambda e: bool((json.loads(e) if isinstance(e, str) else (e or {})).get("withdrawn")))
    return df[~withdrawn.astype(bool)].drop(columns="explanation").reset_index(drop=True)


async def fetch_quotes(base_url: str, tickers: list[str]) -> dict[str, Quote]:
    out = {}
    async with ReadOnlyClient(base_url, allowed_paths=KALSHI_READ_PATHS, reads_per_sec=5) as c:
        for t in tickers:
            ts = utcnow()
            rows = norm_orderbook(await c.get(f"/markets/{t}/orderbook"), t, ts)
            df = pd.DataFrame(rows, columns=["ticker", "snapshot_ts", "side", "price", "size"])
            out[t] = MarketData._book_quote(pd.Timestamp(ts), df)
    return out


async def fetch_markets(base_url: str, tickers: list[str]) -> dict[str, dict]:
    """Market -> {exchange: the Kalshi exchange whose cash it trades against, close: when
    trading ends}. Either may be None if Kalshi doesn't say."""
    out = {}
    async with ReadOnlyClient(base_url, allowed_paths=KALSHI_READ_PATHS, reads_per_sec=5) as c:
        for t in tickers:
            m = (await c.get(f"/markets/{t}")).get("market") or {}
            ex, close = m.get("exchange_index"), m.get("close_time")
            out[t] = {"exchange": None if ex is None else int(ex),
                      "close": pd.Timestamp(close) if close else None}
    return out


def _sync_resting(client, state: State, audit: AuditLog
                  ) -> tuple[dict[str, float], dict[int, float]]:
    """Report fills on the bot's resting buys since last cycle, forget finished ones, and
    return the cash held by every resting order on the account (the bot's and any placed
    by hand), by ticker and by exchange. It counts as exposure, and Kalshi's balance still
    includes it ("balance only changes on fills, settlements and transfers"), so it isn't
    spendable."""
    tracked = state.resting
    since = min((t["ts"] for t in tracked.values()), default=None)
    resting = client.orders(status="resting")
    listed = client.orders(min_ts=int(since) - 60) if since is not None else []
    by_id = {o["order_id"]: o for o in listed + resting}
    for oid, t in list(tracked.items()):
        o = by_id.get(oid)
        if o is None:
            if time.time() - t["ts"] > 6 * 3600:      # long expired and out of Kalshi's list
                audit.write("maker_untracked", order_id=oid, ticker=t["ticker"])
                tracked.pop(oid)
            continue
        new = o["filled"] - t.get("filled", 0.0)
        if new > 0:
            audit.write("maker_fill", order_id=oid, ticker=t["ticker"], outcome=t["outcome"],
                        fill_count=new, price=t["price"], model_version=t["model_version"])
            t["filled"] = o["filled"]
        if o["status"] != "resting" or o["remaining"] <= 0:
            tracked.pop(oid)
    cost: dict[str, float] = {}
    by_ex: dict[int, float] = {}
    for o in resting:
        if o["remaining"] > 0:
            v = o["remaining"] * o["cost"]
            cost[o["ticker"]] = cost.get(o["ticker"], 0.0) + v
            if o.get("exchange") is not None:
                by_ex[o["exchange"]] = by_ex.get(o["exchange"], 0.0) + v
    return cost, by_ex


_DOWN_CODES = ("exchange_is_paused", "trading_is_paused", "service_unavailable")


def _kalshi_down(e: Exception) -> bool:
    """Kalshi itself is unavailable (maintenance, paused, 503): nothing reached the book."""
    return isinstance(e, TradingError) and (
        e.status == 503 or any(c in str(e) for c in _DOWN_CODES))


#: Series that bet on the same price at the same close ("BTC above X" and "BTC in range X"),
#: so the event cap counts them as one event.
SAME_UNDERLYING = {"KXBTCD": "KXBTC", "KXETHD": "KXETH", "KXSOLD": "KXSOL", "KXXRPD": "KXXRP"}


def event_of(ticker: str) -> str:
    """Kalshi market ticker -> its event (KXHIGHNY-26NOV06-B67.5 -> KXHIGHNY-26NOV06), with
    same-underlying series merged (KXBTCD-26NOV0617-T75999.99 -> KXBTC-26NOV0617)."""
    ev = ticker.rsplit("-", 1)[0]
    series, _, rest = ev.partition("-")
    return f"{SAME_UNDERLYING.get(series, series)}-{rest}" if rest else ev


def run_cycle(cfg: ExecutorSettings, sim_engine: Engine, model_version: str, **kw) -> list[dict]:
    """One cycle, never two at once: a second run waits up to two minutes, then refuses."""
    data = Path(cfg.data_dir)
    try:
        with exclusive(data / f"executor_{cfg.mode}.lock", wait_s=120):
            return _run_cycle(cfg, sim_engine, model_version, **kw)
    except Busy as e:
        AuditLog(data / f"audit_{cfg.mode}.jsonl").write("cycle_refused", reason=str(e),
                                                         mode=cfg.mode)
        return [{"status": "refused", "reason": str(e)}]


def _run_cycle(cfg: ExecutorSettings, sim_engine: Engine, model_version: str, *,
               client: KalshiTradingClient | None = None,
               quotes: dict[str, Quote] | None = None,
               exchanges: dict[str, int] | None = None,
               closes: dict[str, pd.Timestamp] | None = None,
               confirm: Callable[[Plan], bool] | None = None,
               fees: FeeSchedule | None = None, forecast_max_age_h: float = 6.0) -> list[dict]:
    cfg = cfg.for_strategy(model_version)
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
    base = BASE_URLS["demo" if cfg.mode == "demo" else "live"]

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

    # Resting orders hold cash like positions do. If they can't be read, nothing new is
    # bought this cycle (exits still run): the caps can't be checked without them.
    resting_cost: dict[str, float] = {}
    resting_by_ex: dict[int, float] = {}
    no_buys = ""
    if client is not None and hasattr(client, "orders"):
        try:
            resting_cost, resting_by_ex = _sync_resting(client, state, audit)
        except (TradingError, OSError) as e:
            audit.write("orders_lookup_failed", error=str(e)[:200])
            no_buys = "could not read resting orders"
    cash -= sum(resting_cost.values())     # held by resting orders: not spendable
    mine = {t["ticker"]: {**t, "order_id": oid} for oid, t in state.resting.items()
            if t["model_version"] == model_version}
    pools: dict[int, float] = {}
    ex_of: dict[str, int] = {}

    def cancel(ticker: str, why: str) -> bool:
        """Cancel this deployment's resting buy on ``ticker``; True once it's gone."""
        nonlocal cash
        o = mine.pop(ticker)
        try:
            n = client.cancel_order(o["order_id"], ticker)
        except (TradingError, OSError) as e:
            # Kept in state: the next cycle sees whether it filled, expired or still rests.
            audit.write("maker_cancel_error", order_id=o["order_id"], ticker=ticker,
                        error=str(e)[:200])
            return False
        state.resting.pop(o["order_id"], None)
        audit.write("maker_cancel", order_id=o["order_id"], ticker=ticker, reason=why,
                    cancelled=n)
        freed = n * o["price"]
        cash += freed
        if ex_of.get(ticker) in pools:
            pools[ex_of[ticker]] += freed
        left = resting_cost.get(ticker, 0.0) - freed
        if left > 1e-9:
            resting_cost[ticker] = left
        else:
            resting_cost.pop(ticker, None)
        return True

    def finish(extra: list[dict], why: str) -> list[dict]:
        """Cancel this deployment's resting buys nothing in this cycle kept, then save."""
        if not state.halted:
            for t in list(mine):
                cancel(t, why)
        state.save(state_path)
        audit.write("cycle_end", results=len(results) + len(extra), halted=state.halted)
        return results + extra

    # Exits first: close positions this deployment opened whose take-profit or stop was hit.
    levels = exit_levels(sim_engine, model_version)
    held = [p for p in positions if p["ticker"] in levels]
    if held and client is not None:
        books = quotes if quotes is not None else asyncio.run(fetch_quotes(
            base, [p["ticker"] for p in held]))
        if closes is None:
            try:
                closes = {t: m["close"] for t, m in asyncio.run(fetch_markets(
                    base, [p["ticker"] for p in held])).items() if m["close"] is not None}
            except Exception as e:  # noqa: BLE001 - close unknown: stops stay on
                audit.write("market_lookup_failed", error=str(e)[:200])
                closes = {}
        for p in held:
            q = books.get(p["ticker"])
            if q is not None and (utcnow() - q.ts).total_seconds() > cfg.max_quote_age_s:
                q = None
            close = closes.get(p["ticker"])
            soon = close is not None and \
                close - pd.Timestamp(utcnow()) <= pd.Timedelta(hours=cfg.no_stop_hours)
            ex = plan_exit(p, levels[p["ticker"]], q, allow_stop=not soon)
            if ex is None:
                continue
            audit.write("exit_submit", mode=cfg.mode, plan=ex.__dict__)
            try:
                ack = client.place_ioc_exit(ex.ticker, ex.held, ex.count, ex.min_price)
            except (TradingError, OSError) as e:
                if _kalshi_down(e):
                    # Not our error: stop trading this cycle, retry next run, no halt count.
                    audit.write("exchange_unavailable", ticker=ex.ticker, error=str(e)[:200])
                    results.append({"ticker": ex.ticker, "status": "exchange_unavailable",
                                    "reason": str(e)})
                    no_buys = "Kalshi unavailable"
                    break
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
            return finish([], "halted")

    if no_buys:
        mine.clear()                       # can't see them now: leave them to expire
        return finish([{"status": "no_buys", "reason": no_buys}], no_buys)
    up = total_value - (state.day_start_value_usd or total_value)
    if cfg.daily_profit_target_usd is not None and up >= cfg.daily_profit_target_usd:
        # Goal met for the day: keep the gain, open nothing new until tomorrow.
        audit.write("profit_target_reached", day=state.day, up_usd=round(up, 2),
                    target_usd=cfg.daily_profit_target_usd)
        return finish([{"status": "target_reached", "up_usd": round(up, 2)}],
                      "daily goal reached")

    fc = latest_forecasts(sim_engine, model_version, forecast_max_age_h)
    if fc.empty:
        audit.write("no_forecasts", model_version=model_version)
        return finish([{"status": "no_forecasts"}], "no current forecast")
    if quotes is None:
        quotes = asyncio.run(fetch_quotes(base, fc["ticker"].tolist()))
    # Kalshi keeps cash per exchange; an order may only use its market's exchange's cash.
    pools.update(client.cash_by_exchange() if hasattr(client, "cash_by_exchange") else {})
    for ex, held_back in resting_by_ex.items():
        if ex in pools:
            pools[ex] -= held_back
    if pools and exchanges is None:
        try:
            exchanges = {t: m["exchange"] for t, m in asyncio.run(fetch_markets(
                base, fc["ticker"].tolist())).items() if m["exchange"] is not None}
        except Exception as e:  # noqa: BLE001 - unknown exchange: size against all cash
            audit.write("exchange_lookup_failed", error=str(e)[:200])
            exchanges = {}
    ex_of.update(exchanges or {})
    # Kelly sizes against the bankroll: what's available now, or the account up to the cap.
    equity = max(cash, 0.0) if cfg.cap_to_available else \
        min(total_value, cfg.bankroll_cap_usd or total_value)
    # Trade exactly as certified: the certified kappa, or the configured shrink if that is
    # more conservative (closer to the market).
    kappa = certified_shrink(sim_engine, model_version)
    shrink = cfg.shrink if kappa is None else min(cfg.shrink, kappa)
    maker = cfg.order_style == "maker" and (client is None or hasattr(client, "place_maker_buy"))
    for r in fc.itertuples():
        if r.ticker in state.closed:
            if r.ticker in mine:
                cancel(r.ticker, "exited earlier")
            results.append({"ticker": r.ticker, "status": "skipped", "reason": "exited earlier"})
            continue
        if r.ticker in state.dropped:
            if r.ticker in mine:
                cancel(r.ticker, "dropped earlier")
            results.append({"ticker": r.ticker, "status": "skipped",
                            "reason": "dropped: price and forecast fell"})
            continue
        if exposure.get(r.ticker, 0.0) > 0:
            # One decision per market, as in the backtest: never add to a held position.
            if r.ticker in mine:
                cancel(r.ticker, "position opened")
            results.append({"ticker": r.ticker, "status": "skipped", "reason": "already held"})
            continue
        if r.ticker in resting_cost and r.ticker not in mine:
            results.append({"ticker": r.ticker, "status": "skipped", "reason": "order resting"})
            continue
        q = quotes.get(r.ticker)
        if q is None or (utcnow() - q.ts).total_seconds() > cfg.max_quote_age_s:
            mine.pop(r.ticker, None)      # can't re-price it: leave it to expire
            results.append({"ticker": r.ticker, "status": "skipped", "reason": "no fresh book"})
            continue
        ex = ex_of.get(r.ticker)
        own = mine.get(r.ticker)
        free = min(cash, pools[ex]) if ex in pools else cash
        # Our own resting order's cash comes back if it's cancelled and replaced.
        reserved = (own["count"] - own.get("filled", 0.0)) * own["price"] if own else 0.0
        plan, why = plan_order(r.ticker, r.p, q, fees, equity=equity, cash=free + reserved,
                               min_edge=cfg.min_edge, kelly_fraction=cfg.kelly_fraction,
                               max_order_usd=cfg.max_order_usd, shrink=shrink,
                               min_entry_price=cfg.min_entry_price,
                               max_entry_price=cfg.max_entry_price,
                               min_order_usd=cfg.min_order_usd, maker=maker)
        if plan is not None and cfg.max_forecast_gap is not None:
            gap = plan.p_side - (plan.max_price if maker else plan.best_ask)
            if gap > cfg.max_forecast_gap + 1e-9:
                plan, why = None, (f"forecast {gap:.2f} above price, more than "
                                   f"{cfg.max_forecast_gap:.2f}: likely a forecast error")
        if own and plan is not None and plan.outcome == own["outcome"] \
                and own.get("p") is not None and plan.max_price < own["price"] - 1e-9 \
                and plan.p_side < own["p"] - 1e-9:
            # Price and forecast both fell: re-pricing lower would chase a falling market.
            state.dropped = sorted({*state.dropped, r.ticker})
            cancel(r.ticker, "price and forecast both fell")
            results.append({"ticker": r.ticker, "status": "skipped",
                            "reason": "dropped: price and forecast fell"})
            continue
        if own:
            if plan is not None and plan.outcome == own["outcome"] \
                    and abs(plan.max_price - own["price"]) < 1e-9:
                mine.pop(r.ticker)        # same side and price: keep its place in the queue
                results.append({"ticker": r.ticker, "status": "resting", "kept": True,
                                "order_id": own["order_id"]})
                continue
            if not cancel(r.ticker, "re-priced" if plan else f"edge gone: {why}"):
                results.append({"ticker": r.ticker, "status": "skipped",
                                "reason": "resting order could not be cancelled"})
                continue
            free = min(cash, pools[ex]) if ex in pools else cash     # now includes the refund
        avail = free
        if plan is not None and plan.count * plan.max_price > avail:
            # The exchange may hold the limit price, not the ask, while the order is
            # checked: keep count x limit inside this exchange's cash.
            n = math.floor(avail / plan.max_price)
            plan = replace(plan, count=n, cost_usd=round(plan.cost_usd / plan.count * n, 2)) \
                if n >= 1 else None
            why = why or f"not enough cash on exchange {ex}"
        ev = event_of(r.ticker)
        in_event = sum(v for t, v in (*exposure.items(), *resting_cost.items())
                       if event_of(t) == ev)
        room = (cfg.max_event_usd - in_event) if cfg.max_event_usd is not None else math.inf
        if plan is not None and plan.count * plan.max_price > room + 1e-9:
            # Shrink to what's left of the event budget, but never below the minimum order.
            n = math.floor(max(room, 0.0) / plan.max_price + 1e-9)
            if n >= 1 and n * plan.max_price >= (cfg.min_order_usd or 0) - 1e-9:
                plan = replace(plan, count=n, cost_usd=round(plan.cost_usd / plan.count * n, 2))
            else:
                plan, why = None, f"event cap ${cfg.max_event_usd:.2f} reached on {ev}"
        if plan is None:
            results.append({"ticker": r.ticker, "status": "skipped", "reason": why})
            continue
        chk = pre_trade_check(cfg, state, cost_usd=plan.cost_usd, cash_usd=avail,
                              open_exposure_usd=sum(exposure.values())
                              + sum(resting_cost.values()),
                              market_exposure_usd=exposure.get(r.ticker, 0.0),
                              total_value_usd=total_value, event_exposure_usd=in_event)
        if not chk.ok:
            if "daily loss stop" in chk.reason:
                state.halted = chk.reason
            audit.write("order_blocked", ticker=r.ticker, reason=chk.reason, plan=plan.__dict__)
            results.append({"ticker": r.ticker, "status": "blocked", "reason": chk.reason})
            if state.halted or "kill switch" in chk.reason:
                break
            continue
        if cfg.mode == "dry_run":
            audit.write("dry_run_order", plan=plan.__dict__, style=cfg.order_style)
            results.append({"ticker": r.ticker, "status": "dry_run", **plan.__dict__})
            continue
        if cfg.mode == "live" and cfg.attended and not (confirm and confirm(plan)):
            audit.write("order_declined_by_user", plan=plan.__dict__)
            results.append({"ticker": r.ticker, "status": "declined"})
            continue
        audit.write("order_submit", mode=cfg.mode, plan=plan.__dict__,
                    style="maker" if maker else "taker")
        try:
            if maker:
                ack = client.place_maker_buy(
                    r.ticker, plan.outcome, plan.count, plan.max_price,
                    expires_ts=int(time.time() + cfg.maker_ttl_min * 60))
            else:
                ack = client.place_ioc_buy(r.ticker, plan.outcome, plan.count, plan.max_price)
        except (TradingError, OSError) as e:
            if isinstance(e, TradingError) and "insufficient_balance" in str(e):
                # Kalshi rejected it outright (nothing reached the book). Skip this market
                # and treat its exchange as empty for the rest of the cycle; markets on
                # other exchanges still trade. Not counted toward the error halt.
                audit.write("order_blocked", ticker=r.ticker, exchange=ex,
                            reason="insufficient balance on Kalshi", plan=plan.__dict__)
                results.append({"ticker": r.ticker, "status": "blocked",
                                "reason": "insufficient balance on Kalshi"})
                if ex in pools:
                    pools[ex] = 0.0
                else:
                    break                   # one shared pool, and it's empty
                continue
            if _kalshi_down(e):
                # Kalshi is down: no more orders this cycle, retry next run, no halt count.
                # Resting buys can't be cancelled now; they expire on Kalshi's side.
                audit.write("exchange_unavailable", ticker=r.ticker, error=str(e)[:200])
                results.append({"ticker": r.ticker, "status": "exchange_unavailable",
                                "reason": str(e)})
                mine.clear()
                break
            # Unknown whether it reached the exchange: never auto-resubmit; the next
            # cycle reconciles positions and resting orders from Kalshi first.
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
        rest = ack.remaining_count if maker else 0.0
        if rest > 0:
            state.resting[ack.order_id] = {
                "ticker": r.ticker, "model_version": model_version, "outcome": plan.outcome,
                "price": plan.max_price, "p": plan.p_side, "count": float(plan.count),
                "filled": float(ack.fill_count), "ts": time.time()}
            resting_cost[r.ticker] = resting_cost.get(r.ticker, 0.0) + rest * plan.max_price
        spent = (ack.fill_count + rest) * plan.max_price      # filled, or held by the order
        cash -= spent
        if ex in pools:
            pools[ex] -= spent
        if ack.fill_count:
            exposure[r.ticker] = exposure.get(r.ticker, 0.0) + ack.fill_count * plan.max_price
        results.append({"ticker": r.ticker, "order_id": ack.order_id, "filled": ack.fill_count,
                        "status": "filled" if ack.fill_count else
                        "resting" if rest > 0 else "unfilled"})
    return finish([], "no current forecast")
