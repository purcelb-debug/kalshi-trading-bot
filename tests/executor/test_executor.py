import base64

import httpx
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from executor.auth import presign_text, sign
from executor.client import KalshiTradingClient, yes_book_side, yes_leg_price
from executor.config import ExecutorSettings
from executor.gate import evaluate_gate
from executor.safety import AuditLog, State, pre_trade_check, startup_checks


def test_presign_strips_query_and_both_key_types_verify():
    text = presign_text(1700000000000, "post", "https://x/trade-api/v2/portfolio/events/orders?a=1")
    assert text == "1700000000000POST/trade-api/v2/portfolio/events/orders"
    ed = ed25519.Ed25519PrivateKey.generate()
    ed.public_key().verify(base64.b64decode(sign(ed, text)), text.encode())
    r = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    r.public_key().verify(base64.b64decode(sign(r, text)), text.encode(),
                          padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                      salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())


def test_no_buy_maps_to_yes_leg_ask():
    assert (yes_book_side("yes"), yes_book_side("no")) == ("bid", "ask")
    assert yes_leg_price("no", 0.30) == 0.70


def test_order_body_and_endpoint_allowlist():
    sent = {}

    def handler(req):
        sent["path"], sent["json"] = req.url.path, req.read()
        assert "KALSHI-ACCESS-SIGNATURE" in req.headers
        return httpx.Response(201, json={"order_id": "o1", "fill_count": "3.00",
                                         "remaining_count": "0.00"})

    c = KalshiTradingClient("https://k/trade-api/v2", "kid", ed25519.Ed25519PrivateKey.generate(),
                            transport=httpx.MockTransport(handler))
    ack = c.place_ioc_buy("T", "no", 3, 0.30)
    assert ack.fill_count == 3 and sent["path"] == "/trade-api/v2/portfolio/events/orders"
    assert b'"side":"ask"' in sent["json"].replace(b" ", b"") and b'"0.7000"' in sent["json"]
    assert b"immediate_or_cancel" in sent["json"]
    with pytest.raises(PermissionError):
        c._call("POST", "/portfolio/orders/cancel_all")


def test_gate_closed_without_full_evidence():
    assert not evaluate_gate("m", None, 4).open
    ok = {"holdout_brier_diff_vs_market_ci_high": -0.001, "holdout_net_pnl_ci_low": 5.0,
          "paper_days": 28, "paper_degradation_alerts": 0, "paper_settled_trades": 50}
    assert evaluate_gate("m", ok, 4).open
    assert not evaluate_gate("m", {**ok, "paper_days": 27}, 4).open
    assert not evaluate_gate("m", {**ok, "holdout_net_pnl_ci_low": 0.0}, 4).open


def test_gate_needs_minimum_settled_paper_trades():
    ok = {"holdout_brier_diff_vs_market_ci_high": -0.001, "holdout_net_pnl_ci_low": 5.0,
          "paper_days": 28, "paper_degradation_alerts": 0, "paper_settled_trades": 50}
    assert evaluate_gate("m", ok, 4, min_paper_trades=50).open
    few = evaluate_gate("m", {**ok, "paper_settled_trades": 49}, 4, min_paper_trades=50)
    assert not few.open and any("49 settled trades" in r for r in few.reasons)
    missing = {k: v for k, v in ok.items() if k != "paper_settled_trades"}
    assert not evaluate_gate("m", missing, 4).open


def test_live_refuses_without_limits_and_kill_switch_blocks(tmp_path):
    cfg = ExecutorSettings(_env_file=None, mode="live", live_key_id="k", live_key_path="p",
                           kill_switch_path=str(tmp_path / "K"))
    assert "limits not configured" in startup_checks(cfg, State()).reason
    cfg = cfg.model_copy(update={"bankroll_cap_usd": 50, "max_order_usd": 5,
                                 "daily_loss_stop_usd": 10})
    assert startup_checks(cfg, State()).ok
    args = {"cash_usd": 100, "open_exposure_usd": 0, "market_exposure_usd": 0, "total_value_usd": 100}
    assert not pre_trade_check(cfg, State(), cost_usd=6, **args).ok          # > max order
    assert not pre_trade_check(cfg, State(), cost_usd=4, **{**args, "open_exposure_usd": 48}).ok
    assert not pre_trade_check(cfg, State(day_start_value_usd=111), cost_usd=1, **args).ok
    assert pre_trade_check(cfg, State(), cost_usd=4, **args).ok
    (tmp_path / "K").write_text("x")
    assert not startup_checks(cfg, State()).ok


def test_audit_chain_detects_edit(tmp_path):
    a = AuditLog(tmp_path / "a.jsonl")
    a.write("x", n=1)
    a.write("y", n=2)
    assert a.verify()
    p = tmp_path / "a.jsonl"
    p.write_text(p.read_text().replace('"n": 1', '"n": 9'))
    assert not a.verify()


def test_predlab_never_imports_executor():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "predlab"
    assert not [f for f in root.rglob("*.py") if "executor" in f.read_text("utf-8")
                and ("import executor" in f.read_text("utf-8")
                     or "from executor" in f.read_text("utf-8"))]


def test_certified_kappa_caps_shrink(tmp_path):
    from executor.gate import certified_shrink
    from predlab.db.repo import init_sim_db, make_engine
    from predlab.db.schema import model_registry
    from predlab.timeutil import utcnow
    sim = make_engine(f"sqlite:///{tmp_path / 'sim.db'}")
    init_sim_db(sim)
    assert certified_shrink(sim, "m@S-h30") is None
    with sim.begin() as c:
        c.execute(model_registry.insert().values(
            model_version="m@S-h30", model_name="m", params_hash="x", trained_ts=utcnow(),
            status="rejected", validation_json={"gate": {"shrink_kappa": 0.15}}))
    assert certified_shrink(sim, "m@S-h30") == 0.15


def test_shipped_env_example_loads_as_dry_run_with_limits_unset(tmp_path):
    """Regression (2026-10-30): blank limit lines in .env.example failed to parse."""
    from pathlib import Path
    example = Path(__file__).resolve().parents[2] / "executor" / ".env.example"
    cfg = ExecutorSettings(_env_file=str(example))
    assert cfg.mode == "dry_run"
    assert cfg.missing_limits() == ["bankroll_cap_usd", "max_order_usd", "daily_loss_stop_usd"]


def test_gate_override_needs_exact_phrase_and_live_mode():
    from executor.config import GATE_OVERRIDE_PHRASE
    base = {"_env_file": None, "bankroll_cap_usd": 50, "max_order_usd": 5, "daily_loss_stop_usd": 10}
    assert not ExecutorSettings(mode="live", gate_override="yes", **base).gate_overridden
    assert not ExecutorSettings(mode="demo", gate_override=GATE_OVERRIDE_PHRASE,
                                **base).gate_overridden
    assert ExecutorSettings(mode="live", gate_override=GATE_OVERRIDE_PHRASE, **base).gate_overridden


def test_override_trades_past_closed_gate_but_limits_still_apply(tmp_path):
    from executor.config import GATE_OVERRIDE_PHRASE
    from executor.runner import run_cycle
    from predlab.db.repo import init_sim_db, make_engine
    sim = make_engine(f"sqlite:///{tmp_path / 'sim.db'}")
    init_sim_db(sim)

    class Client:
        def balance_usd(self):
            return 100.0, 0.0

        def positions(self):
            return []

    common = {"_env_file": None, "mode": "live", "live_key_id": "k", "live_key_path": "p",
                  "data_dir": str(tmp_path), "kill_switch_path": str(tmp_path / "KILL"),
                  "bankroll_cap_usd": 50, "max_order_usd": 5, "daily_loss_stop_usd": 10}
    closed = run_cycle(ExecutorSettings(**common), sim, "m@S-h30", client=Client())
    assert closed[0]["status"] == "gate_closed"
    over = run_cycle(ExecutorSettings(**common, gate_override=GATE_OVERRIDE_PHRASE), sim,
                     "m@S-h30", client=Client())
    assert over == [{"status": "no_forecasts"}]          # got past the gate
    assert "gate_overridden" in (tmp_path / "audit_live.jsonl").read_text()
    nolimits = {k: v for k, v in common.items() if k != "daily_loss_stop_usd"}
    refused = run_cycle(ExecutorSettings(**nolimits, gate_override=GATE_OVERRIDE_PHRASE), sim,
                        "m@S-h30", client=Client())
    assert refused[0]["status"] == "refused"


def test_reads_retry_on_429_but_orders_never_retry():
    key = ed25519.Ed25519PrivateKey.generate()
    calls = {"GET": 0, "POST": 0}

    def handler(req):
        calls[req.method] += 1
        if calls[req.method] == 1:
            return httpx.Response(429, json={"error": "too_many_requests"})
        return httpx.Response(200, json={"balance_dollars": "24.51", "portfolio_value": 0})

    slept = []
    c = KalshiTradingClient("https://x/trade-api/v2", "kid", key,
                            transport=httpx.MockTransport(handler), sleep=slept.append)
    assert c.balance_usd() == (24.51, 0.0) and calls["GET"] == 2 and slept == [0.5]
    from executor.client import TradingError
    with pytest.raises(TradingError):
        c.place_ioc_buy("T", "yes", 1, 0.01)
    assert calls["POST"] == 1


def test_live_cycle_orders_once_per_market(tmp_path):
    import pandas as pd

    from executor.client import OrderAck
    from executor.config import GATE_OVERRIDE_PHRASE
    from executor.runner import run_cycle
    from predlab.backtest.execution import Quote
    from predlab.db.repo import init_sim_db, make_engine
    from predlab.forecasts import record_forecasts
    from predlab.models.base import Forecast
    from predlab.timeutil import utcnow
    sim = make_engine(f"sqlite:///{tmp_path / 'sim.db'}")
    init_sim_db(sim)
    now = pd.Timestamp(utcnow())
    record_forecasts(sim, [Forecast(t, "m@S-h30", 0.70, None, None, now, now, (), {})
                           for t in ("HELD", "NEW")])
    sent = []

    class Client:
        def balance_usd(self):
            return 25.0, 2.0

        def positions(self):
            return [{"ticker": "HELD", "side": "yes", "qty": 5, "exposure_usd": 2.0,
                     "fees_usd": 0.1}]

        def place_ioc_buy(self, ticker, outcome, count, max_price):
            sent.append((ticker, outcome, count, max_price))
            return OrderAck("o1", "c1", float(count), 0.0, 0.40, 0.02)

    q = Quote(now, "book", [(0.40, 500.0)], [(0.61, 500.0)], 0.39, 0.60)
    cfg = ExecutorSettings(_env_file=None, mode="live", live_key_id="k", live_key_path="p",
                           data_dir=str(tmp_path), kill_switch_path=str(tmp_path / "KILL"),
                           bankroll_cap_usd=25, max_order_usd=10, daily_loss_stop_usd=25,
                           gate_override=GATE_OVERRIDE_PHRASE, attended=False)
    out = {r["ticker"]: r for r in run_cycle(cfg, sim, "m@S-h30", client=Client(),
                                             quotes={"HELD": q, "NEW": q})}
    assert out["HELD"]["reason"] == "already held"
    assert out["NEW"]["status"] == "filled" and [s[0] for s in sent] == ["NEW"]
    assert sent[0][1] == "yes" and sent[0][3] >= 0.40
    assert sent[0][2] * 0.43 <= 10


def test_ledger_sync_and_realized_pnl(tmp_path):
    from executor.ledger import load, sync

    class Client:
        def fills(self, min_ts=None):
            return [{"fill_id": "f1", "order_id": "o1", "market_ticker": "T1",
                     "outcome_side": "no", "action": "buy", "count_fp": "10.00",
                     "yes_price_dollars": "0.3000", "no_price_dollars": "0.7000",
                     "is_taker": True, "fee_cost": "0.15", "created_time": "2026-10-30T23:00:00Z"}]

        def settlements(self, min_ts=None):
            return [{"ticker": "T1", "event_ticker": "E1", "market_result": "no",
                     "yes_count_fp": "0.00", "no_count_fp": "10.00",
                     "yes_total_cost_dollars": "0.0000", "no_total_cost_dollars": "7.0000",
                     "revenue": 1000, "fee_cost": "0.1500", "settled_time": "2026-11-01T06:00:00Z"}]

        def balance_usd(self):
            return 27.36, 0.0

    import json as _json
    (tmp_path / "audit_live.jsonl").write_text(_json.dumps({"event": "order_ack",
                                                            "order_id": "o1"}) + "\n")
    out = sync(Client(), str(tmp_path))
    assert out["fills_seen"] == 1 and out["settlements_seen"] == 1
    sync(Client(), str(tmp_path))                       # idempotent
    f, s, b = load(str(tmp_path))
    assert len(f) == 1 and f.iloc[0]["price"] == 0.70 and f.iloc[0]["outcome"] == "no"
    assert len(s) == 1 and s.iloc[0]["realized_pnl"] == pytest.approx(10.0 - 7.0 - 0.15)
    assert f.iloc[0]["source"] == "bot" and s.iloc[0]["source"] == "bot"
    assert len(b) == 2 and b.iloc[-1]["equity"] == pytest.approx(27.36)
