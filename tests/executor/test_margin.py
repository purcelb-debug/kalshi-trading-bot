"""Higher-margin trading: resting maker buys, no stops near close, per-strategy sizing."""

import json

import httpx
import pandas as pd
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from executor.client import KalshiTradingClient, OrderAck
from executor.config import GATE_OVERRIDE_PHRASE, ExecutorSettings
from predlab.backtest.decision import maker_price, plan_order
from predlab.backtest.execution import Quote
from predlab.backtest.fees import FeeSchedule
from predlab.timeutil import utcnow


def _cfg(tmp_path, **kw):
    return ExecutorSettings(_env_file=None, mode="live", live_key_id="k", live_key_path="p",
                            data_dir=str(tmp_path), kill_switch_path=str(tmp_path / "KILL"),
                            bankroll_cap_usd=40, max_order_usd=15, daily_loss_stop_usd=25,
                            gate_override=GATE_OVERRIDE_PHRASE, attended=False, **kw)


def _sim(tmp_path, rows):
    from predlab.db.repo import init_sim_db, make_engine
    from predlab.forecasts import record_forecasts
    from predlab.models.base import Forecast
    sim = make_engine(f"sqlite:///{tmp_path / 'sim.db'}")
    init_sim_db(sim)
    now = pd.Timestamp(utcnow())
    record_forecasts(sim, [Forecast(t, mv, p, None, None, now, now, (), e)
                           for t, mv, p, e in rows])
    return sim


def _events(tmp_path):
    return [json.loads(x) for x in (tmp_path / "audit_live.jsonl").read_text().splitlines()]


def test_maker_price_never_crosses_the_ask():
    assert maker_price(0.40, 0.45) == 0.41           # step in front of the bid
    assert maker_price(0.40, 0.41) == 0.40           # 1c spread: join the bid
    assert maker_price(None, 0.30) == 0.29           # no bid: just under the ask
    assert maker_price(0.45, 0.45) == 0.44           # locked book: still under the ask
    for bid in (None, 0.01, 0.2, 0.5, 0.97):
        for ask in (0.02, 0.21, 0.6, 0.99):
            if bid is None or bid < ask:
                assert maker_price(bid, ask) < ask


def test_maker_plan_prices_at_the_bid_with_maker_fees():
    q = Quote(pd.Timestamp(utcnow()), "book", [(0.45, 500.0)], [(0.58, 500.0)], 0.42, 0.55)
    kw = {"equity": 40, "cash": 40, "min_edge": 0.04, "kelly_fraction": 0.5,
          "max_order_usd": 15, "shrink": 1.0}
    taker, _ = plan_order("T", 0.50, q, FeeSchedule(), **kw)
    maker, _ = plan_order("T", 0.50, q, FeeSchedule(), maker=True, **kw)
    assert taker is None                             # 50% vs a 45c ask: 3.3c after taker fees
    assert maker.outcome == "yes" and maker.max_price == 0.43 and maker.best_ask == 0.45
    assert maker.ev == pytest.approx(0.50 - 0.43 - 0.0175 * 0.43 * 0.57)


def test_maker_order_body_and_cancel_allowlist():
    sent = []

    def handler(req):
        sent.append((req.method, req.url.path, req.read(), dict(req.url.params)))
        if req.method == "DELETE":
            return httpx.Response(200, json={"order_id": "o1", "reduced_by": "4.00",
                                             "ts_ms": 1})
        return httpx.Response(201, json={"order_id": "o1", "fill_count": "0.00",
                                         "remaining_count": "5.00"})

    c = KalshiTradingClient("https://k/trade-api/v2", "kid", ed25519.Ed25519PrivateKey.generate(),
                            transport=httpx.MockTransport(handler))
    ack = c.place_maker_buy("T", "no", 5, 0.30, expires_ts=1_800_000_000)
    body = json.loads(sent[0][2])
    assert ack.remaining_count == 5 and body["post_only"] is True
    assert body["time_in_force"] == "good_till_canceled"
    assert body["expiration_time"] == 1_800_000_000 and body["side"] == "ask"
    assert body["price"] == "0.7000" and body["reduce_only"] is False
    assert c.cancel_order("o1", "T") == 4.0
    assert sent[1][:2] == ("DELETE", "/trade-api/v2/portfolio/events/orders/o1")
    assert sent[1][3] == {"market_ticker": "T"}
    for bad in ("/portfolio/events/orders/o1/amend", "/portfolio/events/orders/",
                "/portfolio/orders/o1"):
        with pytest.raises(PermissionError):
            c._call("DELETE", bad)


def test_stop_is_skipped_when_the_market_closes_soon(tmp_path):
    from executor.runner import run_cycle
    exp = {"side": "yes", "take_profit": 0.70, "stop": 0.35}
    sim = _sim(tmp_path, [(t, "desk@ALL-h48", 0.6, exp) for t in ("SOON", "LATER")])
    exits = []

    class Client:
        def balance_usd(self):
            return 10.0, 5.0

        def positions(self):
            return [{"ticker": t, "side": "yes", "qty": 5, "exposure_usd": 2.5,
                     "fees_usd": 0.1} for t in ("SOON", "LATER")]

        def place_ioc_exit(self, ticker, held, count, min_price):
            exits.append(ticker)
            return OrderAck("x", "c", float(count), 0.0, min_price, 0.01)

    now = pd.Timestamp(utcnow())
    dip = Quote(now, "book", [(0.33, 9.0)], [(0.69, 9.0)], 0.30, 0.67)     # bid 30c < stop
    run_cycle(_cfg(tmp_path), sim, "desk@ALL-h48", client=Client(),
              quotes={"SOON": dip, "LATER": dip},
              closes={"SOON": now + pd.Timedelta(hours=10),
                      "LATER": now + pd.Timedelta(hours=40)})
    assert exits == ["LATER"]                       # only the far-dated stop fired
    jump = Quote(now, "book", [(0.76, 9.0)], [(0.26, 9.0)], 0.74, 0.24)
    exits.clear()
    run_cycle(_cfg(tmp_path), sim, "desk@ALL-h48", client=Client(),
              quotes={"SOON": jump}, closes={"SOON": now + pd.Timedelta(hours=10)})
    assert exits == ["SOON"]                        # take-profit still works near close


class MakerClient:
    """Fake Kalshi account with resting orders."""

    def __init__(self, manual=()):
        self.book, self.placed, self.cancelled, self.n = {}, [], [], 0
        for t in manual:
            self.book[f"manual-{t}"] = {"order_id": f"manual-{t}", "ticker": t,
                                        "status": "resting", "remaining": 10.0,
                                        "filled": 0.0, "cost": 0.5}

    def balance_usd(self):
        return 30.0, 0.0

    def positions(self):
        return []

    def orders(self, min_ts=None, status=None):
        return [dict(o) for o in self.book.values() if status is None or o["status"] == status]

    def place_maker_buy(self, ticker, outcome, count, price, expires_ts):
        self.n += 1
        oid = f"o{self.n}"
        self.placed.append((ticker, outcome, count, price))
        self.book[oid] = {"order_id": oid, "ticker": ticker, "status": "resting",
                          "remaining": float(count), "filled": 0.0, "cost": price}
        return OrderAck(oid, "c", 0.0, float(count), None, None)

    def cancel_order(self, oid, ticker):
        o = self.book[oid]
        o["status"] = "canceled"
        self.cancelled.append(oid)
        left, o["remaining"] = o["remaining"], 0.0
        return left

    def place_ioc_buy(self, *a):
        raise AssertionError("maker mode must not take the ask")


def test_maker_cycle_keeps_reprices_reports_fills_and_cancels(tmp_path):
    from executor.runner import run_cycle
    mv = "m@S-h30"
    sim = _sim(tmp_path, [("A", mv, 0.62, {}), ("MAN", mv, 0.62, {})])
    c = MakerClient(manual=["MAN"])
    cfg = _cfg(tmp_path, order_style="maker", min_edge=0.04, kelly_fraction=0.5, shrink=1.0)
    now = pd.Timestamp(utcnow())
    q1 = Quote(now, "book", [(0.50, 9.0)], [(0.55, 9.0)], 0.44, 0.48)
    out = {r["ticker"]: r for r in run_cycle(cfg, sim, mv, client=c, quotes={"A": q1, "MAN": q1})}
    assert out["A"]["status"] == "resting" and c.placed[0][:2] == ("A", "yes")
    assert c.placed[0][3] == 0.45                       # 1c over the 44c bid, under the 50c ask
    assert out["MAN"]["reason"] == "order resting"      # a manual order: never touched
    # Same book next cycle: the order keeps its place, nothing new is sent.
    run_cycle(cfg, sim, mv, client=c, quotes={"A": q1, "MAN": q1})
    assert len(c.placed) == 1 and c.cancelled == []
    # Partly filled, then the bid moves: the fill is reported, the rest re-priced.
    c.book["o1"].update(filled=3.0, remaining=c.book["o1"]["remaining"] - 3.0)
    q2 = Quote(now, "book", [(0.50, 9.0)], [(0.53, 9.0)], 0.46, 0.50)
    run_cycle(cfg, sim, mv, client=c, quotes={"A": q2, "MAN": q2})
    assert c.cancelled == ["o1"] and c.placed[-1][3] == 0.47
    fills = [e for e in _events(tmp_path) if e["event"] == "maker_fill"]
    assert [(f["ticker"], f["fill_count"], f["price"]) for f in fills] == [("A", 3.0, 0.45)]
    # Edge gone: the resting buy is cancelled and not replaced.
    q3 = Quote(now, "book", [(0.61, 9.0)], [(0.40, 9.0)], 0.60, 0.39)
    run_cycle(cfg, sim, mv, client=c, quotes={"A": q3, "MAN": q3})
    assert c.cancelled == ["o1", "o2"] and len(c.placed) == 2
    assert "manual-MAN" not in c.cancelled
    state = json.loads((tmp_path / "state_live.json").read_text())
    assert state["resting"] == {}


def test_resting_orders_count_against_the_bankroll_cap(tmp_path):
    from executor.runner import run_cycle
    mv = "m@S-h30"
    sim = _sim(tmp_path, [("A", mv, 0.80, {})])
    c = MakerClient(manual=["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"])  # $40 resting
    cfg = _cfg(tmp_path, order_style="maker", shrink=1.0)
    q = Quote(pd.Timestamp(utcnow()), "book", [(0.50, 9.0)], [(0.55, 9.0)], 0.44, 0.48)
    out = run_cycle(cfg, sim, mv, client=c, quotes={"A": q})
    # Refused by free cash (the $40 is held) and, failing that, the bankroll cap.
    assert out[0]["status"] in ("skipped", "blocked") and c.placed == []


def test_unreadable_orders_stop_new_buys(tmp_path):
    from executor.client import TradingError
    from executor.runner import run_cycle
    mv = "m@S-h30"
    sim = _sim(tmp_path, [("A", mv, 0.80, {})])

    class Broken(MakerClient):
        def orders(self, **kw):
            raise TradingError(503, "down")

    c = Broken()
    q = Quote(pd.Timestamp(utcnow()), "book", [(0.50, 9.0)], [(0.55, 9.0)], 0.44, 0.48)
    out = run_cycle(_cfg(tmp_path, order_style="maker"), sim, mv, client=c, quotes={"A": q})
    assert out == [{"status": "no_buys", "reason": "could not read resting orders"}]
    assert c.placed == []


def test_strategy_overrides_apply_per_deployment_only():
    cfg = ExecutorSettings(_env_file=None, shrink=0.5, kelly_fraction=0.5, strategy_overrides={
        "ny@KXHIGHNY-h30": {"shrink": 0.7, "kelly_fraction": 0.6}})
    ny = cfg.for_strategy("ny@KXHIGHNY-h30")
    assert (ny.shrink, ny.kelly_fraction) == (0.7, 0.6)
    assert cfg.for_strategy("desk@ALL-h48") is cfg
    with pytest.raises(ValueError, match="unknown"):
        ExecutorSettings(_env_file=None, strategy_overrides={"x": {"max_order_usd": 99}}
                         ).for_strategy("x")


def test_strategy_override_reads_from_env(monkeypatch):
    monkeypatch.setenv("EXECUTOR_STRATEGY_OVERRIDES", '{"ny@S-h30": {"shrink": 0.7}}')
    assert ExecutorSettings(_env_file=None).for_strategy("ny@S-h30").shrink == 0.7


def test_override_shrink_is_still_capped_by_certified_kappa(tmp_path):
    from executor.runner import run_cycle
    from predlab.db.schema import model_registry
    mv = "m@S-h30"
    sim = _sim(tmp_path, [("A", mv, 0.90, {})])
    with sim.begin() as conn:
        conn.execute(model_registry.insert().values(
            model_version=mv, model_name="m", params_hash="h", params_json={},
            trained_ts=utcnow(), validation_json={"gate": {"shrink_kappa": 0.05}},
            status="candidate", status_ts=utcnow()))
    c = MakerClient()
    cfg = _cfg(tmp_path, order_style="maker", min_edge=0.10,
               strategy_overrides={mv: {"shrink": 1.0}})
    # Mid 0.48: at kappa 0.05 the 90% forecast becomes ~0.51, under 10c of edge at 45c
    # (at the override's 1.0 it would trade).
    q = Quote(pd.Timestamp(utcnow()), "book", [(0.50, 9.0)], [(0.55, 9.0)], 0.44, 0.48)
    out = run_cycle(cfg, sim, mv, client=c, quotes={"A": q})
    assert c.placed == [] and "edge" in out[0]["reason"]


def test_cash_held_by_resting_orders_is_not_spent_twice(tmp_path):
    # Kalshi's balance still includes cash held by resting orders, so a $25 manual order
    # leaves $5 of a $30 balance to spend, on that order's exchange.
    from executor.runner import run_cycle
    mv = "m@S-h30"
    sim = _sim(tmp_path, [("A", mv, 0.80, {})])

    class Client(MakerClient):
        def cash_by_exchange(self):
            return {0: 30.0}

    c = Client(manual=["X1"])
    c.book["manual-X1"].update(remaining=50.0, exchange=0)          # 50 x 50c = $25 held
    cfg = _cfg(tmp_path, order_style="maker", shrink=1.0)
    q = Quote(pd.Timestamp(utcnow()), "book", [(0.50, 9.0)], [(0.55, 9.0)], 0.44, 0.48)
    run_cycle(cfg, sim, mv, client=c, quotes={"A": q}, exchanges={"A": 0})
    ((_, _, count, price),) = c.placed
    assert count * price <= 5.0 + 1e-9


def test_resting_buy_is_dropped_not_chased_when_price_and_forecast_fall(tmp_path):
    """2026-11-06: an AUDUSD buy was re-priced 9 times down as the price and the desk's own
    forecast fell, then went to zero. A falling price alone may still re-price."""
    from executor.runner import run_cycle
    from predlab.forecasts import record_forecasts
    from predlab.models.base import Forecast
    mv = "m@S-h30"
    sim = _sim(tmp_path, [("A", mv, 0.62, {}), ("B", mv, 0.62, {})])
    c = MakerClient()
    cfg = _cfg(tmp_path, order_style="maker", min_edge=0.04, kelly_fraction=0.5, shrink=1.0)
    now = pd.Timestamp(utcnow())
    q1 = Quote(now, "book", [(0.50, 9.0)], [(0.55, 9.0)], 0.44, 0.48)
    run_cycle(cfg, sim, mv, client=c, quotes={"A": q1, "B": q1})
    assert [p[0] for p in c.placed] == ["A", "B"]
    later = pd.Timestamp(utcnow())
    record_forecasts(sim, [Forecast("A", mv, 0.58, None, None, later, later, (), {})])
    q2 = Quote(now, "book", [(0.50, 9.0)], [(0.59, 9.0)], 0.40, 0.48)    # bid 44c -> 40c
    out = {r["ticker"]: r for r in run_cycle(cfg, sim, mv, client=c, quotes={"A": q2, "B": q2})}
    assert out["A"]["reason"] == "dropped: price and forecast fell"
    assert [p[0] for p in c.placed] == ["A", "B", "B"]    # B (same forecast) re-priced, A not
    assert c.placed[-1][3] == 0.41
    run_cycle(cfg, sim, mv, client=c, quotes={"A": q2, "B": q2})
    assert [p[0] for p in c.placed].count("A") == 1        # and never bought again
    state = json.loads((tmp_path / "state_live.json").read_text())
    assert state["dropped"] == ["A"] and all(o["ticker"] != "A" for o in state["resting"].values())
