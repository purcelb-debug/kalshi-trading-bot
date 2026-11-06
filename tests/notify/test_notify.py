"""Notifier: chunking, delivery, and once-only digest messages from local files."""

import json
from datetime import UTC, datetime, timedelta

import httpx

from executor.config import ExecutorSettings
from executor.safety import AuditLog
from notify.config import NotifySettings
from notify.digest import build_messages
from notify.telegram import chunks, send

NOW = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
GOOD_GATE = {"holdout_brier_diff_vs_market_ci_high": -0.01, "holdout_net_pnl_ci_low": 1.0,
             "paper_days": 30, "paper_degradation_alerts": 0, "paper_settled_trades": 60}


def _cfg(tmp_path, **kw):
    return NotifySettings(_env_file=None, report_dir=str(tmp_path / "reports"),
                          executor_data_dir=str(tmp_path / "exec"),
                          state_path=str(tmp_path / "state.json"), **kw)


def _exec():
    return ExecutorSettings(_env_file=None)


def _report(tmp_path, now, alerts=(), gate=None):
    d = tmp_path / "reports"
    d.mkdir(exist_ok=True)
    (d / "loop-m_S-h30-latest.json").write_text(json.dumps(
        {"deployment": "m@S-h30", "now": str(now), "alerts": list(alerts), "gate": gate or {}}))


def test_chunks_respect_limit_and_keep_text():
    text = "\n".join("x" * 30 for _ in range(10)) + "\n" + "y" * 95
    parts = chunks(text, limit=40)
    assert all(len(p) <= 40 for p in parts) and "".join(parts) == text


def test_send_disabled_without_token(tmp_path):
    assert send(_cfg(tmp_path), "hi") == (False, "disabled")


def test_send_posts_each_chunk_and_hides_token_on_error(tmp_path):
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={"ok": True})

    cfg = _cfg(tmp_path, telegram_bot_token="SECRET", telegram_chat_id="-1")
    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        assert send(cfg, "a" * 5000, client=c) == (True, "")
    assert len(seen) == 2 and b"chat_id=-1" in seen[0].content

    def boom(req):
        raise httpx.ConnectError("down")

    with httpx.Client(transport=httpx.MockTransport(boom)) as c:
        ok, err = send(cfg, "x", client=c)
    assert not ok and "SECRET" not in err


def _balance(tmp_path, cash, positions, ts=NOW):
    from executor.ledger import _upsert, balances, ledger_engine
    eng = ledger_engine(str(tmp_path / "exec"))
    _upsert(eng, balances, [{"ts": ts, "cash": cash, "positions_value": positions}])
    eng.dispose()


def test_research_noise_sends_nothing(tmp_path):
    cfg, state = _cfg(tmp_path), {}
    log = AuditLog(tmp_path / "exec" / "audit_live.jsonl")
    log.write("cycle_start", mode="live")
    assert build_messages(cfg, state, NOW, _exec()) == []           # first look: baseline
    _report(tmp_path, NOW, alerts=[{"rule": "worse_than_market", "detail": "d"}],
            gate=GOOD_GATE)
    for _ in range(3):
        log.write("cycle_start", mode="live")
        log.write("gate_overridden", reasons=["registry status is 'candidate'"])
        log.write("no_forecasts", model_version="m")
    assert build_messages(cfg, state, NOW, _exec()) == []           # nothing happened


def test_stale_job_warns_once_then_says_running_again(tmp_path):
    cfg, state = _cfg(tmp_path), {}
    _report(tmp_path, NOW)
    later = NOW + timedelta(hours=3)
    msg = build_messages(cfg, state, later, _exec())
    assert len(msg) == 1 and "hasn't finished a run" in msg[0]
    assert build_messages(cfg, state, later, _exec()) == []
    _report(tmp_path, later)
    assert "Bot is running again" in build_messages(cfg, state, later, _exec())[0]


def test_one_short_summary_with_account_and_the_bots_own_trades(tmp_path):
    cfg, state = _cfg(tmp_path), {}
    _balance(tmp_path, 47.63, 3.96)
    log = AuditLog(tmp_path / "exec" / "audit_live.jsonl")
    log.write("cycle_end", results=0, halted="old history")
    assert build_messages(cfg, state, NOW, _exec()) == []           # history is not replayed
    log.write("order_submit", mode="live", plan={"ticker": "T-1", "outcome": "no",
                                                 "best_ask": 0.74, "max_price": 0.76})
    log.write("order_ack", order_id="o1", fill_count=4.0, average_fill_price=0.26)
    log.write("exit_submit", mode="live", plan={"ticker": "T-2", "held": "yes",
                                                "reason": "take_profit", "min_price": 0.7})
    log.write("exit_ack", order_id="x1", fill_count=5.0, average_fill_price=0.71)
    for _ in range(2):
        log.write("order_blocked", ticker="T-3", reason="bankroll cap $40.00 would be exceeded")
        log.write("cycle_refused", reason="kill switch present")
    (msg,) = build_messages(cfg, state, NOW, _exec())
    head, body = msg.split("\n\n")
    assert head.startswith("Kalshi bot · ") and "Account $51.59 = cash $47.63 + positions $3.96" in head
    assert body.splitlines() == ["• Bought 4 NO T-1 at 74¢ ($2.96)",
                                 "• Sold (take_profit) 5 YES T-2 at 71¢ ($3.55)",
                                 "• 2 buys skipped: bankroll cap",
                                 "• Trading paused: kill switch present"] or         body.splitlines() == ["• Bought 4 NO T-1 at 74¢ ($2.96)",
                              "• Sold (take_profit) 5 YES T-2 at 71¢ ($3.55)",
                              "• Trading paused: kill switch present",
                              "• 2 buys skipped: bankroll cap"]
    assert build_messages(cfg, state, NOW, _exec()) == []
    log.write("cycle_end", results=0, halted="daily loss stop hit")
    assert "Trading HALTED: daily loss stop hit" in build_messages(cfg, state, NOW, _exec())[0]


def test_desk_meetings_counted_not_quoted(tmp_path):
    cfg, state = _cfg(tmp_path), {}
    d = tmp_path / "reports" / "desk-meetings"
    d.mkdir(parents=True)
    (d / "20261102T1400.txt").write_text("long transcript\nrecorded BUY A-1 p_yes=0.7")
    (d / "20261102T1500.txt").write_text("another long transcript\nno trades")
    (msg,) = build_messages(cfg, state, NOW, _exec())
    assert msg.endswith("• Desk: 2 meetings, 1 pick (bot bought 0)") and "transcript" not in msg
    assert build_messages(cfg, state, NOW, _exec()) == []


def test_digest_survives_a_non_utf8_console(tmp_path):
    """The scheduled job logs in the Windows code page; a transcript with characters it
    can't encode once crashed the digest before sending, blocking every later message."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    d = tmp_path / "reports" / "desk-meetings"
    d.mkdir(parents=True)
    (d / "20261102T190734-desk_X.txt").write_text("[DESK MEETING] edge ≤ 3c → WAIT",
                                                   encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "TELEGRAM_BOT_TOKEN": "",
           "TELEGRAM_CHAT_ID": "", "REPORT_DIR": str(tmp_path / "reports"),
           "EXECUTOR_DATA_DIR": str(tmp_path / "ex"), "STATE_PATH": str(tmp_path / "s.json")}
    root = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, "-m", "notify.cli", "digest", "--dry"], cwd=root,
                       env=env, capture_output=True, timeout=120, check=False)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-500:]
    assert "• Desk: 1 meeting".encode() in r.stdout and "·".encode() in r.stdout


def test_daily_target_reported_once_per_finished_day(tmp_path):
    from executor.ledger import _upsert, balances, ledger_engine
    eng = ledger_engine(str(tmp_path))
    snaps = [("2026-11-02T14:00:00Z", 40.0), ("2026-11-03T02:00:00Z", 42.0),   # Nov 2 ET close 42
             ("2026-11-03T20:00:00Z", 51.5), ("2026-11-04T15:00:00Z", 51.0)]
    _upsert(eng, balances, [{"ts": datetime.fromisoformat(t), "cash": v,
                             "positions_value": 0.0} for t, v in snaps])
    eng.dispose()
    cfg = NotifySettings(_env_file=None, report_dir=str(tmp_path / "r"),
                         executor_data_dir=str(tmp_path), daily_target_usd=8.0)
    state = {}
    now = datetime(2026, 11, 4, 16, 0, tzinfo=UTC)
    (msg,) = build_messages(cfg, state, now=now, exec_cfg=ExecutorSettings(_env_file=None))
    assert "Account $51.00 = cash $51.00 + positions $0.00" in msg
    assert msg.endswith("• Yesterday: +$9.50, goal met ($8.00). Last 1 days: avg +$9.50/day, "
                        "goal met 1 of 1")
    again = build_messages(cfg, state, now=now, exec_cfg=ExecutorSettings(_env_file=None))
    assert again == []                                                  # once per day


def test_resting_order_fill_is_reported_as_a_buy(tmp_path):
    cfg, state = _cfg(tmp_path), {}
    _balance(tmp_path, 20.0, 1.35)
    log = AuditLog(tmp_path / "exec" / "audit_live.jsonl")
    log.write("cycle_start", mode="live")
    build_messages(cfg, state, NOW, _exec())                 # first sight: start at the end
    log.write("maker_fill", order_id="o1", ticker="T-1", outcome="no", fill_count=3.0,
              price=0.45, model_version="m@S-h30")
    (msg,) = build_messages(cfg, state, NOW, _exec())
    assert "• Bought 3 NO T-1 at 45¢ ($1.35, resting order)" in msg


def test_capital_step_needs_trades_and_profit_and_fires_once_per_step():
    from notify.digest import capital_step

    def report(trades, pnl):
        row = {"trades": trades, "net_pnl": pnl, "fees": 1.0}
        return {"bot": {"settled": {"by_category": [row]}, "exit": {"by_category": []}}}

    steps = "30:60,60:100"
    assert capital_step(report(29, 5.0), steps, 0) is None           # not enough trades
    assert capital_step(report(40, -0.5), steps, 0) is None          # not profitable
    n, msg = capital_step(report(31, 4.2), steps, 0)
    assert n == 30 and "$60.00" in msg and "+$4.20" in msg and "won't change it" in msg
    assert capital_step(report(45, 6.0), steps, 30) is None          # already told
    assert capital_step(report(70, 9.0), steps, 30)[0] == 60         # next step
    assert capital_step(report(70, 9.0), steps, 0)[0] == 60          # skip straight to the top


def test_halt_reminders_downtime_and_desk_picks_vs_buys(tmp_path):
    cfg, state = _cfg(tmp_path, halt_reminder_hours=3), {}
    log = AuditLog(tmp_path / "exec" / "audit_live.jsonl")
    log.write("cycle_end", results=0, halted="")
    build_messages(cfg, state, NOW, _exec())           # first sight: start at the end
    halted = "halted: 3 consecutive order errors (clear with `executor resume`)"

    def refuse():
        log.write("cycle_refused", reason=halted, mode="live")

    refuse()
    assert "Trading paused: " + halted in build_messages(cfg, state, NOW, _exec())[0]
    standing = state["audit"]["audit_live.jsonl"]["standing"]
    standing["reminded"] = (datetime.fromisoformat(standing["reminded"])
                            - timedelta(hours=3)).isoformat()      # 3h have passed
    refuse()
    assert "Still halted, no orders placed: " + halted in build_messages(cfg, state, NOW,
                                                                         _exec())[0]
    refuse()
    assert build_messages(cfg, state, NOW, _exec()) == []          # not again so soon
    log.write("resume", cleared="3 consecutive order errors")
    log.write("exchange_unavailable", ticker="T-1", error="409 exchange_is_paused")
    log.write("exchange_unavailable", ticker="T-1", error="409 exchange_is_paused")
    refuse()                                          # same words after a resume: news
    (msg,) = build_messages(cfg, state, NOW, _exec())
    assert "Trading paused: " + halted in msg
    assert "Kalshi unavailable (2x), no orders then; retrying each run" in msg

    d = tmp_path / "reports" / "desk-meetings"
    d.mkdir(parents=True)
    (d / "20261102T1600.txt").write_text("recorded BUY A-1 p_yes=0.7\nrecorded BUY B-1")
    log.write("order_submit", mode="live", plan={"ticker": "A-1", "outcome": "yes",
                                                 "max_price": 0.5})
    log.write("order_ack", order_id="o1", fill_count=2.0, average_fill_price=0.5)
    (msg,) = build_messages(cfg, state, NOW, _exec())
    assert msg.endswith("• Desk: 1 meeting, 2 picks (bot bought 1)")
