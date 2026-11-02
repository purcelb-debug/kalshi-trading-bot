"""Turn what changed since the last run into phone messages.

Sources (all local, read-only): ``reports/loop-*-latest.json`` from the paper loop,
``executor/data/audit_*.jsonl`` from the executor, and ``reports/desk-meetings/*.txt``
transcripts from the desk. Progress is kept in ``data/notify_state.json`` so each event
is sent once. Only events worth a phone buzz are sent; quiet cycles send nothing.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from executor.config import ExecutorSettings
from executor.gate import evaluate_gate
from notify.config import NotifySettings


def load_state(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def save_state(path: str, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(p)


def _loop_messages(cfg: NotifySettings, state: dict, now: datetime,
                   gate_weeks: int, gate_min_trades: int) -> list[str]:
    out, loops = [], state.setdefault("loops", {})
    for f in sorted(Path(cfg.report_dir).glob("loop-*-latest.json")):
        rep = json.loads(f.read_text(encoding="utf-8"))
        dep = rep["deployment"]
        seen = loops.setdefault(dep, {})
        ts = datetime.fromisoformat(rep["now"])
        ts = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
        if now - ts > timedelta(hours=cfg.stale_after_hours):
            if not seen.get("stale_warned"):
                out.append(f"[LOOP] {dep}: no completed cycle since {ts:%Y-%m-%d %H:%M} UTC "
                           f"(more than {cfg.stale_after_hours:g}h). PC asleep, or the "
                           "collection is failing; see data/logs/loop.log.")
                seen["stale_warned"] = True
            continue
        if seen.get("stale_warned"):
            out.append(f"[LOOP] {dep}: cycles resumed ({ts:%Y-%m-%d %H:%M} UTC).")
            seen["stale_warned"] = False
        if seen.get("last_now") == rep["now"]:
            continue
        seen["last_now"] = rep["now"]
        day = f"{ts:%Y-%m-%d}"
        if rep.get("alerts") and seen.get("alerts_day") != day:
            rules = ", ".join(f"{a['rule']} ({a.get('detail', '')})" for a in rep["alerts"])
            out.append(f"[ALERT] {dep}: {rules}. Alerts count against the go-live gate.")
            seen["alerts_day"] = day
        g = evaluate_gate(dep, rep.get("gate"), gate_weeks, gate_min_trades)
        if seen.get("gate_open") is not None and g.open != seen["gate_open"]:
            out.append(f"[GATE] {dep}: go-live gate is now {'OPEN' if g.open else 'CLOSED'}."
                       + ("" if g.open else "\n" + "\n".join(f"- {r}" for r in g.reasons)))
        seen["gate_open"] = g.open
    return out


def _audit_line(rec: dict, seen: dict) -> str | None:
    ev = rec.get("event")
    if ev == "order_ack" and rec.get("fill_count"):
        return (f"filled {rec['fill_count']:g} @ avg {rec.get('average_fill_price')} "
                f"(order {rec.get('order_id')})")
    if ev == "order_error":
        return f"ORDER ERROR {rec.get('ticker')}: {rec.get('error')}"
    if ev == "exit_ack" and rec.get("fill_count"):
        return (f"SOLD ({rec.get('reason')}) {rec['fill_count']:g} @ avg "
                f"{rec.get('average_fill_price')} (order {rec.get('order_id')})")
    if ev == "exit_error":
        return f"EXIT ERROR {rec.get('ticker')}: {rec.get('error')}"
    if ev == "order_blocked":
        return f"blocked {rec.get('ticker')}: {rec.get('reason')}"
    if ev == "cycle_end" and rec.get("halted"):
        return f"HALTED: {rec['halted']}"
    if ev in ("gate_closed", "gate_overridden"):
        # Standing conditions logged every cycle: report only when they change.
        reason = str(rec.get("reasons") or ev)
        standing = seen.setdefault("standing", {})
        if standing.get("gate") == f"{ev}:{reason}":
            return None
        standing["gate"] = f"{ev}:{reason}"
        return f"{ev.replace('_', ' ')}: {reason}"
    if ev == "cycle_refused":
        reason = str(rec.get("reason"))
        if seen.get("last_refusal") == reason:
            return None                     # same refusal every hour: say it once
        seen["last_refusal"] = reason
        return f"cycle refused: {reason}"
    if ev == "cycle_start":
        seen["last_refusal"] = ""           # a cycle ran, so a later refusal is news again
    return None


def _executor_messages(cfg: NotifySettings, state: dict) -> list[str]:
    out, audits = [], state.setdefault("audit", {})
    for f in sorted(Path(cfg.executor_data_dir).glob("audit_*.jsonl")):
        if f.name not in audits:
            # First sight of this log: start at its end rather than replaying history.
            audits[f.name] = {"offset": f.stat().st_size}
            continue
        seen = audits[f.name]
        if f.stat().st_size < seen["offset"]:
            seen["offset"] = 0              # file replaced: start over
        lines, dry = [], 0
        with f.open("rb") as fh:
            fh.seek(seen["offset"])
            for raw in fh:
                if not raw.endswith(b"\n"):
                    break                   # partial line still being written
                seen["offset"] += len(raw)
                rec = json.loads(raw)
                if rec.get("event") == "dry_run_order":
                    dry += 1
                elif (line := _audit_line(rec, seen)) is not None:
                    lines.append(line)
        mode = f.stem.removeprefix("audit_")
        if dry:
            lines.append(f"{dry} dry-run order(s) planned (nothing sent)")
        if lines:
            out.append(f"[EXECUTOR {mode}]\n" + "\n".join(lines))
    return out


def _desk_messages(cfg: NotifySettings, state: dict) -> list[str]:
    d = Path(cfg.report_dir) / "desk-meetings"
    last = state.get("desk_last", "")
    new = sorted(p for p in d.glob("*.txt") if p.name > last) if d.is_dir() else []
    if new:
        state["desk_last"] = new[-1].name
    return [p.read_text(encoding="utf-8") for p in new]


def _target_messages(cfg: NotifySettings, state: dict, now: datetime) -> list[str]:
    """Once per day, yesterday's profit against the owner's daily target."""
    from executor.ledger import daily_pnl, load
    try:
        _, _, b = load(cfg.executor_data_dir)
    except Exception:  # noqa: BLE001 - no ledger yet: nothing to grade
        return []
    d = daily_pnl(b).dropna(subset=["pnl"])
    today = now.astimezone(ZoneInfo("America/New_York")).date()
    done = d[d["day"] < today]
    if done.empty or state.get("target_day") == str(done["day"].iloc[-1]):
        return []
    last = done.iloc[-1]
    state["target_day"] = str(last["day"])
    week = done.tail(7)
    met = int((week["pnl"] >= cfg.daily_target_usd).sum())
    verdict = "MET" if last["pnl"] >= cfg.daily_target_usd else "MISSED"
    return [(f"[DAILY TARGET] {last['day']}: {verdict} - made ${last['pnl']:+.2f} vs "
             f"${cfg.daily_target_usd:.2f} target (account ${last['equity_close']:.2f}).\n"
             f"Last {len(week)} day(s): avg ${week['pnl'].mean():+.2f}/day, target met {met} "
             f"of {len(week)}.")]


def build_messages(cfg: NotifySettings, state: dict, now: datetime | None = None,
                   exec_cfg: ExecutorSettings | None = None) -> list[str]:
    """Messages to send now; mutates ``state`` to mark them sent."""
    now = now or datetime.now(UTC)
    exec_cfg = exec_cfg or ExecutorSettings()
    return (_loop_messages(cfg, state, now, exec_cfg.gate_paper_weeks,
                           exec_cfg.gate_min_paper_trades)
            + _executor_messages(cfg, state) + _desk_messages(cfg, state)
            + _target_messages(cfg, state, now))
