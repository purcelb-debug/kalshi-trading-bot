"""Turn what changed since the last run into one short phone message.

One message per run, and only when something happened: trades, early sells, settlements,
desk meetings, problems, the bot going quiet, or the daily goal report. It starts with
today's profit and the account value split into cash and open positions.

Sources (all local, read-only): the executor ledger (``executor sync`` runs just before
the digest, so fills, settlements and the balance are fresh), the executor audit log
(problems only), desk transcripts (counted, not quoted), and the paper loop reports (only
to notice the job stopping). Progress lives in ``data/notify_state.json`` so each event is
reported once; nothing older than the first run is replayed.
"""

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from executor.config import ExecutorSettings
from notify.config import NotifySettings

ET = ZoneInfo("America/New_York")


def load_state(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def save_state(path: str, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(p)


def _money(v: float, sign: bool = False) -> str:
    return f"{'+' if sign and v >= 0 else '-' if v < 0 else ''}${abs(v):.2f}"


def _ledger(cfg: NotifySettings):
    from executor.ledger import load
    try:
        return load(cfg.executor_data_dir)
    except Exception:  # noqa: BLE001 - no ledger yet
        return None


def _header(cfg: NotifySettings, ledger, now: datetime) -> str:
    when = now.astimezone(ET)
    top = f"Kalshi bot · {when:%a} {when:%I:%M %p}".replace(" 0", " ")
    if ledger is None or ledger[2].empty:
        return top + "\nAccount: no balance recorded yet"
    b = ledger[2].iloc[-1]
    cash, pos = float(b["cash"]), float(b["positions_value"])
    total = cash + pos
    today = _today_pnl(cfg, ledger[2], total, now)
    goal = f" (goal {_money(cfg.daily_target_usd)})" if cfg.daily_target_usd else ""
    return (f"{top}\nToday: {_money(today, sign=True)}{goal}\n"
            f"Account {_money(total)} = cash {_money(cash)} + positions {_money(pos)}")


def _today_pnl(cfg: NotifySettings, b: pd.DataFrame, total: float, now: datetime) -> float:
    """Since the executor's start-of-day value (the same base as its +$8 lock), else since
    yesterday's last balance."""
    today = now.astimezone(ET).date().isoformat()
    sp = Path(cfg.executor_data_dir) / "state_live.json"
    if sp.is_file():
        st = json.loads(sp.read_text(encoding="utf-8"))
        if st.get("day") == today and st.get("day_start_value_usd") is not None:
            return total - float(st["day_start_value_usd"])
    days = b["ts"].dt.tz_convert(ET).dt.date.astype(str)
    before = b[days < today]
    return total - float(before.iloc[-1]["equity"]) if len(before) else 0.0


def _settlement_lines(ledger, state: dict) -> list[str]:
    """Markets that settled since the last run, with realized profit. Settlements of your
    own manual trades are labelled, so they aren't mistaken for the bot's."""
    if ledger is None:
        return []
    s = ledger[1]
    last = state.get("last_settle_ts")
    if last is None:                        # first run: start from now, don't replay
        state["last_settle_ts"] = (str(s["settled_ts"].max()) if len(s)
                                   else "1970-01-01T00:00:00+00:00")
        return []
    new = s[s["settled_ts"] > pd.Timestamp(last)]
    if new.empty:
        return []
    state["last_settle_ts"] = str(new["settled_ts"].max())
    return [f"Settled {r.ticker}: {_money(r.realized_pnl, sign=True)}"
            + ("" if r.source == "bot" else " (your manual trade)") for r in new.itertuples()]


_BLOCK_REASON = re.compile(r"^[^($\d]+")


def _fill_price(held: str, yes_leg_avg) -> float | None:
    """Order acks report the YES-leg price; a NO contract's price is 1 minus that."""
    if yes_leg_avg is None:
        return None
    return float(yes_leg_avg) if held == "yes" else round(1 - float(yes_leg_avg), 4)


def _audit_lines(cfg: NotifySettings, state: dict) -> tuple[list[str], list[str]]:
    """(the bot's fills, problems) from the live audit log. Fills come from the bot's own
    order records (exact market, side and count); Kalshi's fill feed describes some buys as
    the opposite sell. Problems: errors, halts, refusals, skipped buys (summarised), and the
    daily goal being reached."""
    out, trades, audits = [], [], state.setdefault("audit", {})
    f = Path(cfg.executor_data_dir) / "audit_live.jsonl"
    if not f.is_file():
        return trades, out
    seen = audits.setdefault(f.name, {})
    if "offset" not in seen:
        seen["offset"] = f.stat().st_size   # first sight: start at the end, no replay
        return trades, out
    if f.stat().st_size < seen["offset"]:
        seen["offset"] = 0                  # file replaced: start over
    blocked: dict[str, int] = {}
    standing = seen.setdefault("standing", {})
    plan: dict = {}
    with f.open("rb") as fh:
        fh.seek(seen["offset"])
        for raw in fh:
            if not raw.endswith(b"\n"):
                break                       # partial line still being written
            seen["offset"] += len(raw)
            rec = json.loads(raw)
            ev = rec.get("event")
            if ev in ("order_submit", "exit_submit"):
                plan = rec.get("plan") or {}
            elif ev in ("order_ack", "exit_ack") and rec.get("fill_count"):
                held = plan.get("outcome") or plan.get("held") or "yes"
                n, px = rec["fill_count"], _fill_price(held, rec.get("average_fill_price"))
                px = px if px is not None else plan.get("best_ask") or plan.get("min_price") or 0
                verb = "Bought" if ev == "order_ack" else f"Sold ({plan.get('reason', 'exit')})"
                trades.append(f"{verb} {n:g} {held.upper()} {plan.get('ticker', '?')} at "
                              f"{px * 100:.0f}¢ ({_money(n * px)})")
            elif ev == "maker_fill" and rec.get("fill_count"):
                # A resting buy filled after it was placed (prices are the side bought).
                n, px = rec["fill_count"], float(rec.get("price") or 0)
                trades.append(f"Bought {n:g} {str(rec.get('outcome', 'yes')).upper()} "
                              f"{rec.get('ticker', '?')} at {px * 100:.0f}¢ ({_money(n * px)}, "
                              "resting order)")
            elif ev in ("order_error", "exit_error"):
                out.append(f"Order problem on {rec.get('ticker')}: {str(rec.get('error'))[:80]}")
            elif ev == "order_blocked":
                why = _BLOCK_REASON.match(str(rec.get("reason")))
                key = (why.group(0) if why else str(rec.get("reason"))).strip()
                blocked[key] = blocked.get(key, 0) + 1
            elif ev == "cycle_end" and rec.get("halted"):
                if standing.get("halted") != rec["halted"]:
                    out.append(f"Trading HALTED: {rec['halted']}")
                    standing["halted"] = rec["halted"]
            elif ev == "cycle_start":
                standing["halted"] = ""
            elif ev == "profit_target_reached" and standing.get("target") != rec.get("day"):
                standing["target"] = rec.get("day")
                out.append(f"Daily goal reached ({_money(rec.get('up_usd', 0), sign=True)}): "
                           "no new buys until tomorrow")
            elif ev == "cycle_refused" and standing.get("refused") != rec.get("reason"):
                standing["refused"] = rec.get("reason")
                out.append(f"Trading paused: {rec.get('reason')}")
    out += [f"{n} buy{'s' if n > 1 else ''} skipped: {why}" for why, n in blocked.items()]
    return trades, out


def _desk_lines(cfg: NotifySettings, state: dict) -> list[str]:
    d = Path(cfg.report_dir) / "desk-meetings"
    last = state.get("desk_last", "")
    new = sorted(p for p in d.glob("*.txt") if p.name > last) if d.is_dir() else []
    if not new:
        return []
    state["desk_last"] = new[-1].name
    buys = sum(p.read_text(encoding="utf-8").count("recorded BUY") for p in new)
    return [(f"Desk: {len(new)} meeting{'s' if len(new) > 1 else ''}, "
            f"{buys or 'no'} buy{'' if buys == 1 else 's'}")]


def _stale_lines(cfg: NotifySettings, state: dict, now: datetime) -> list[str]:
    """One line when the job stops completing runs (PC off or asleep), one when it's back."""
    times = []
    for f in Path(cfg.report_dir).glob("loop-*-latest.json"):
        ts = datetime.fromisoformat(json.loads(f.read_text(encoding="utf-8"))["now"])
        times.append(ts if ts.tzinfo else ts.replace(tzinfo=UTC))
    if not times:
        return []
    latest = max(times)
    if now - latest > timedelta(hours=cfg.stale_after_hours):
        if not state.get("stale"):
            state["stale"] = True
            return [(f"Bot hasn't finished a run since {latest.astimezone(ET):%a %I:%M %p} "
                    "(PC off or asleep?)")]
        return []
    if state.get("stale"):
        state["stale"] = False
        return ["Bot is running again"]
    return []


def _target_lines(cfg: NotifySettings, ledger, state: dict, now: datetime) -> list[str]:
    """Once per day: yesterday against the owner's daily goal, with the 7-day record."""
    from executor.ledger import daily_pnl
    if ledger is None:
        return []
    d = daily_pnl(ledger[2]).dropna(subset=["pnl"])
    done = d[d["day"] < now.astimezone(ET).date()]
    if done.empty or state.get("target_day") == str(done["day"].iloc[-1]):
        return []
    last, week = done.iloc[-1], done.tail(7)
    state["target_day"] = str(last["day"])
    met = int((week["pnl"] >= cfg.daily_target_usd).sum())
    verdict = "goal met" if last["pnl"] >= cfg.daily_target_usd else "goal missed"
    return [(f"Yesterday: {_money(last['pnl'], sign=True)}, {verdict} "
            f"({_money(cfg.daily_target_usd)}). Last {len(week)} days: avg "
            f"{_money(week['pnl'].mean(), sign=True)}/day, goal met {met} of {len(week)}")]


def capital_step(report: dict, steps: str, announced: int) -> tuple[int, str] | None:
    """(trades threshold, message) for the highest bankroll step newly earned, or None.
    Earned = at least that many closed bot trades (settled plus early exits) and positive
    total net P&L across them."""
    rows = [r for kind in ("settled", "exit")
            for r in report.get("bot", {}).get(kind, {}).get("by_category", [])]
    trades = sum(r["trades"] for r in rows)
    pnl = sum(r["net_pnl"] for r in rows)
    fees = sum(r["fees"] for r in rows)
    pairs = sorted((int(a), float(b)) for a, b in (s.split(":") for s in steps.split(",") if s))
    earned = [(n, cap) for n, cap in pairs if trades >= n > announced]
    if pnl <= 0 or not earned:
        return None
    n, cap = earned[-1]
    return n, (f"Capital step earned: {trades} closed bot trades, net {_money(pnl, sign=True)} "
               f"after {_money(fees)} fees. If you want to scale up, raise "
               f"EXECUTOR_BANKROLL_CAP_USD to {_money(cap)} (and deposit on Kalshi). "
               "The bot won't change it itself.")


def _capital_lines(cfg: NotifySettings, state: dict) -> list[str]:
    from executor.performance import load_report
    try:
        report = load_report(cfg.executor_data_dir, cfg.sim_db_path)
    except Exception:  # noqa: BLE001 - no ledger yet, or one being written: try next digest
        return []
    step = capital_step(report, cfg.capital_steps, int(state.get("capital_step", 0)))
    if step is None:
        return []
    state["capital_step"] = step[0]
    return [step[1]]


def build_messages(cfg: NotifySettings, state: dict, now: datetime | None = None,
                   exec_cfg: ExecutorSettings | None = None) -> list[str]:
    """The message to send now (or none); mutates ``state`` to mark events reported."""
    now = now or datetime.now(UTC)
    ledger = _ledger(cfg)
    trades, problems = _audit_lines(cfg, state)
    lines = (_stale_lines(cfg, state, now) + trades + _settlement_lines(ledger, state)
             + problems + _desk_lines(cfg, state) + _target_lines(cfg, ledger, state, now)
             + _capital_lines(cfg, state))
    if not lines:
        return []
    return [_header(cfg, ledger, now) + "\n\n" + "\n".join(f"• {x}" for x in lines)]
