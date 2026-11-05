"""Hand-computed portfolios protect attribution and prevent fees being counted twice."""

import json
import sqlite3

import pandas as pd
import pytest
from typer.testing import CliRunner

from executor.performance import build_report, category, load_report, trade_records

ENTRY = "2026-11-01T12:00:00Z"
CLOSE = "2026-11-02T12:00:00Z"
NOW = "2026-11-03T12:00:00Z"
TICKER = "KXHIGHNY-26NOV02-T60"


def fill(oid="bot", count=2, price=0.4, fee=0.02, outcome="yes", action="buy",
         ts=ENTRY, ticker=TICKER):
    return {"order_id": oid, "ticker": ticker, "count": count, "price": price, "fee": fee,
                "outcome": outcome, "action": action, "ts": ts}


def settlement(ticker=TICKER, result="yes", fee=0.02):
    return {"ticker": ticker, "result": result, "settle_fee": fee, "cost": 0.8, "revenue": 2,
                "settled_ts": CLOSE}


def audit(oid="bot", outcome="yes", probability=0.8, manual=False):
    return [{"event": "cycle_start", "model_version": "desk@ALL-h48", "ts": ENTRY},
            {"event": "order_submit", "plan": {"ticker": TICKER, "outcome": outcome,
                 "p_side": probability, "manual": manual}, "ts": ENTRY},
            {"event": "order_ack", "order_id": oid, "fill_count": 2, "ts": ENTRY}]


def report(fills=None, settlements=None, events=None, forecasts=None, **kwargs):
    return build_report(pd.DataFrame([fill()] if fills is None else fills),
                        pd.DataFrame([settlement()] if settlements is None else settlements,
                                     columns=["ticker", "result", "settle_fee", "cost",
                                              "revenue", "settled_ts"]),
                        pd.DataFrame(audit() if events is None else events),
                        pd.DataFrame([] if forecasts is None else forecasts), now=NOW, **kwargs)


@pytest.mark.parametrize(("ticker", "expected"), [
    ("KXHIGHNY-X", "weather_temp"), ("KXLOWTCHI-X", "weather_temp"),
    ("KXRAINNYC-X", "rain"), ("KXRAINELECTION-X", "politics/approval"),
    ("KXTRUMP-X", "politics/approval"), ("KXAPRPOTUS-X", "politics/approval"),
    ("KXBTC15M-X", "crypto"), ("KXETH-X", "crypto"), ("KXSOL-X", "crypto"),
    ("KXXRP-X", "crypto"), ("KXDOGE-X", "crypto"), ("KXWTI-X", "energy"),
    ("KXBRENT-X", "energy"), ("KXNATGAS-X", "energy"), ("KXDIESEL-X", "energy"),
    ("KXAAAGAS-X", "energy"), ("KXGOLDH-X", "metals"), ("KXSILVER-X", "metals"),
    ("KXCOPPER-X", "metals"), ("KXINXU-X", "indices"), ("KXNASDAQ100-X", "indices"),
    ("KXDJI-X", "indices"), ("KXUST10Y-X", "rates"), ("KX30YMORT-X", "rates"),
    ("UNKNOWN-KXBTC", "other"),
])
def test_categories(ticker, expected):
    assert category(ticker) == expected


def test_known_pnl_fees_and_brier():
    row = report()["bot"]["settled"]["by_category"][0]
    assert row["trades"] == 1
    assert row["net_pnl"] == pytest.approx(1.18)
    assert row["fees"] == pytest.approx(0.02)
    assert row["capital_deployed"] == pytest.approx(0.8)
    assert row["return_on_deployed_pct"] == pytest.approx(147.5)
    assert row["average_entry_price"] == pytest.approx(0.4)
    assert row["brier_bot"] == pytest.approx(0.04)
    assert row["brier_market"] == pytest.approx(0.36)
    assert row["sample"] == "insufficient sample"
    assert row["pnl_per_trade_ci95"] == pytest.approx([1.18, 1.18])


def test_manual_same_ticker_never_contaminates_bot():
    rows = report(fills=[fill(), fill("manual", price=0.9, fee=0.1)],
                  settlements=[settlement(fee=0.12)])
    assert rows["bot"]["settled"]["by_category"][0]["net_pnl"] == pytest.approx(1.18)
    assert rows["manual"]["settled"]["by_category"][0]["net_pnl"] == pytest.approx(0.1)


def test_audited_manual_flag():
    result = report(events=audit(manual=True))
    assert result["bot"]["settled"]["by_category"] == []
    assert result["manual"]["settled"]["by_category"][0]["trades"] == 1


@pytest.mark.parametrize(("outcome", "price"), [("no", 0.4), ("yes", 0.6)])
def test_sell_quirk_uses_audited_contract(outcome, price):
    rows = report(fills=[fill(outcome=outcome, price=price, action="sell")],
                  events=audit(outcome="no"), settlements=[settlement(result="no")])
    row = rows["bot"]["settled"]["by_category"][0]
    assert row["net_pnl"] == pytest.approx(1.18)
    assert row["average_entry_price"] == pytest.approx(0.4)
    assert row["brier_bot"] == pytest.approx(0.04)


def test_partial_exit_then_settlement_and_extra_settlement_fee():
    events = audit() + [
        {"event": "exit_submit", "plan": {"ticker": TICKER, "held": "yes"}, "ts": CLOSE},
        {"event": "exit_ack", "order_id": "exit", "fill_count": 1, "ts": CLOSE}]
    rows = report(fills=[fill(), fill("exit", count=1, price=0.7, fee=0.03,
                                    action="sell", ts=CLOSE)],
                  events=events, settlements=[settlement(fee=0.06)])
    early = rows["bot"]["exit"]["by_category"][0]
    settled = rows["bot"]["settled"]["by_category"][0]
    assert early["net_pnl"] == pytest.approx(0.26)
    assert early["fees"] == pytest.approx(0.04)
    assert settled["net_pnl"] == pytest.approx(0.58)
    assert settled["fees"] == pytest.approx(0.02)
    assert rows["bot"]["open_by_category"] == []


def test_open_positions_and_days_keep_old_entry_basis():
    rows = report(settlements=[], days=1)
    assert rows["bot"]["open_by_category"][0]["cost_basis"] == pytest.approx(0.82)
    assert report(days=1)["bot"]["settled"]["by_category"][0]["trades"] == 1
    assert report(days=1, settlements=[{**settlement(), "settled_ts": ENTRY}])[
        "bot"]["settled"]["by_category"] == []
    with pytest.raises(ValueError):
        report(days=0)


def test_deployment_and_no_lookahead():
    forecasts = [{"ticker": TICKER, "model_version": "desk@ALL-h48", "p": 0.8,
                      "created_ts": "2026-11-01T11:00:00Z", "explanation": {}},
                  {"ticker": TICKER, "model_version": "future", "p": 0.99,
                       "created_ts": CLOSE, "explanation": {}}]
    events = audit()
    events[0]["model_version"] = None
    rows = report(events=events, forecasts=forecasts)
    assert rows["bot"]["settled"]["by_deployment"][0]["deployment"] == "desk@ALL-h48"
    forecasts[0]["explanation"] = {"withdrawn": True}
    rows = report(events=events, forecasts=forecasts)
    assert rows["bot"]["settled"]["by_deployment"][0]["deployment"] == "unknown"


def test_missing_probability_excluded_from_brier():
    row = report(events=audit(probability=None))["bot"]["settled"]["by_category"][0]
    assert row["brier_bot"] is None
    assert row["brier_trades"] == 0
    assert row["brier_market"] == pytest.approx(0.36)


def test_partial_fills_one_trade_and_seeded_bootstrap():
    rows = [fill(count=1, fee=0.01), fill(count=1, fee=0.01)]
    result = report(fills=rows)
    assert result == report(fills=rows)
    assert result["bot"]["settled"]["by_category"][0]["trades"] == 1


def test_brier_uses_quantity_weighted_entry_price():
    result = report(fills=[fill(count=1, price=0.2), fill(count=3, price=0.6)])
    row = result["bot"]["settled"]["by_category"][0]
    assert row["average_entry_price"] == pytest.approx(0.5)
    assert row["brier_market"] == pytest.approx(0.25)


def test_withdrawal_does_not_resurrect_prior_matching_forecast():
    events = audit()
    events[0]["model_version"] = None
    forecasts = [
        {"ticker": TICKER, "model_version": "desk@ALL-h48", "p": 0.8,
         "created_ts": "2026-11-01T10:00:00Z", "explanation": {}},
        {"ticker": TICKER, "model_version": "desk@ALL-h48", "p": 0.5,
         "created_ts": "2026-11-01T11:00:00Z", "explanation": {"withdrawn": True}},
    ]
    result = report(events=events, forecasts=forecasts)
    assert result["bot"]["settled"]["by_deployment"][0]["deployment"] == "unknown"


def test_empty_portfolio():
    assert report(fills=[], settlements=[])["bot"]["open_by_category"] == []
    assert report(fills=[])["bot"]["settled"]["by_category"] == []


def test_read_only_loader_and_cli_json(tmp_path, monkeypatch):
    data = tmp_path / "executor"
    data.mkdir()
    ledger = data / "ledger.db"
    sim = tmp_path / "sim.db"
    with sqlite3.connect(ledger) as conn:
        pd.DataFrame([fill()]).to_sql("fills", conn, index=False)
        pd.DataFrame([settlement()]).to_sql("settlements", conn, index=False)
    with sqlite3.connect(sim) as conn:
        pd.DataFrame(columns=["ticker", "created_ts", "model_version", "p", "explanation"]
                     ).to_sql("forecasts", conn, index=False)
    audit_path = data / "audit_live.jsonl"
    audit_path.write_text("\n".join(json.dumps(e) for e in audit()), encoding="utf-8")
    before = {path: path.read_bytes() for path in (ledger, sim, audit_path)}
    assert load_report(data, sim)["bot"]["settled"]["by_category"][0]["trades"] == 1
    from executor import cli

    class Settings:
        data_dir = str(data)
        sim_db_url = f"sqlite:///{sim}"

    monkeypatch.setattr(cli, "ExecutorSettings", Settings)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli.app, ["perf", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["bot"]["settled"]["by_category"][0]["trades"] == 1
    assert len(list((tmp_path / "reports").glob("perf-*.json"))) == 1
    assert before == {path: path.read_bytes() for path in before}


def test_unmatched_exit_warns_without_making_negative_position():
    events = audit() + [{"event": "exit_submit", "plan": {"held": "yes"}, "ts": CLOSE},
                        {"event": "exit_ack", "order_id": "exit", "ts": CLOSE}]
    closed, opened, warnings = trade_records(
        pd.DataFrame([fill("exit", ts=CLOSE)]),
        pd.DataFrame(columns=["settled_ts"]), pd.DataFrame(events), pd.DataFrame())
    assert closed == opened == []
    assert "Unmatched exit" in warnings[0]
