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


def test_alerts_once_per_day_and_gate_change(tmp_path):
    cfg, state = _cfg(tmp_path), {}
    _report(tmp_path, NOW, gate={**GOOD_GATE, "paper_settled_trades": 10})
    assert build_messages(cfg, state, NOW, _exec()) == []           # first look: baseline
    alert = [{"rule": "worse_than_market", "detail": "d"}]
    _report(tmp_path, NOW + timedelta(hours=1), alerts=alert, gate=GOOD_GATE)
    msgs = build_messages(cfg, state, NOW + timedelta(hours=1), _exec())
    assert any("[ALERT]" in m for m in msgs) and any("gate is now OPEN" in m for m in msgs)
    _report(tmp_path, NOW + timedelta(hours=2), alerts=alert, gate=GOOD_GATE)
    assert build_messages(cfg, state, NOW + timedelta(hours=2), _exec()) == []


def test_stale_loop_warns_once_then_resumes(tmp_path):
    cfg, state = _cfg(tmp_path), {}
    _report(tmp_path, NOW)
    later = NOW + timedelta(hours=3)
    assert "no completed cycle" in build_messages(cfg, state, later, _exec())[0]
    assert build_messages(cfg, state, later, _exec()) == []
    _report(tmp_path, later)
    assert "resumed" in build_messages(cfg, state, later, _exec())[0]


def test_executor_events_sent_once_and_refusals_deduplicated(tmp_path):
    cfg, state = _cfg(tmp_path), {}
    log = AuditLog(tmp_path / "exec" / "audit_demo.jsonl")
    log.write("cycle_end", results=0, halted="old history")
    assert build_messages(cfg, state, NOW, _exec()) == []      # history is not replayed
    log.write("cycle_refused", reason="kill switch present")
    log.write("cycle_refused", reason="kill switch present")
    log.write("order_ack", order_id="o1", client_order_id="c", fill_count=3.0,
              remaining_count=0.0, average_fill_price=0.42)
    log.write("dry_run_order", plan={})
    msgs = build_messages(cfg, state, NOW, _exec())
    assert len(msgs) == 1 and msgs[0].count("kill switch") == 1
    assert "filled 3" in msgs[0] and "1 dry-run" in msgs[0]
    assert build_messages(cfg, state, NOW, _exec()) == []
    log.write("cycle_end", results=0, halted="daily loss stop hit")
    assert "HALTED" in build_messages(cfg, state, NOW, _exec())[0]


def test_desk_transcripts_sent_in_order_once(tmp_path):
    cfg, state = _cfg(tmp_path), {}
    d = tmp_path / "reports" / "desk-meetings"
    d.mkdir(parents=True)
    (d / "20261102T1400.txt").write_text("first")
    (d / "20261102T1500.txt").write_text("second")
    assert build_messages(cfg, state, NOW, _exec()) == ["first", "second"]
    assert build_messages(cfg, state, NOW, _exec()) == []


def test_standing_gate_override_reported_once_across_cycles(tmp_path):
    cfg, state = _cfg(tmp_path), {}
    log = AuditLog(tmp_path / "exec" / "audit_live.jsonl")
    log.write("cycle_end", results=0, halted="")
    build_messages(cfg, state, NOW, _exec())
    for _ in range(3):
        log.write("cycle_start", mode="live")
        log.write("gate_overridden", reasons=["registry status is 'candidate'"])
    msgs = build_messages(cfg, state, NOW, _exec())
    assert len(msgs) == 1 and msgs[0].count("gate overridden") == 1
    log.write("cycle_start", mode="live")
    log.write("gate_overridden", reasons=["registry status is 'candidate'"])
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
    assert "≤".encode() in r.stdout


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
    msgs = [m for m in build_messages(cfg, state, now=now, exec_cfg=ExecutorSettings(
        _env_file=None)) if m.startswith("[DAILY TARGET]")]
    assert msgs == [("[DAILY TARGET] 2026-11-03: MET - made $+9.50 vs $8.00 target "
                    "(account $51.50).\nLast 1 day(s): avg $+9.50/day, target met 1 of 1.")]
    again = build_messages(cfg, state, now=now, exec_cfg=ExecutorSettings(_env_file=None))
    assert not any(m.startswith("[DAILY TARGET]") for m in again)      # once per day
