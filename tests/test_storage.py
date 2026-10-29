from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, inspect, select

from predlab.data.integrity import check_observation, check_trade, split_valid
from predlab.db.repo import upsert
from predlab.db.schema import MARKET_MD, SIM_MD, ext_observations, mkt_trades
from predlab.timeutil import UTC, to_utc, utcnow


@pytest.mark.parametrize("value,expected", [
    ("2026-10-29T05:00:00Z", datetime(2026, 10, 29, 5, tzinfo=UTC)),
    ("2026-10-29T01:00:00-04:00", datetime(2026, 10, 29, 5, tzinfo=UTC)),
    ("2026-10-29T04:59:57.965509Z", datetime(2026, 10, 29, 4, 59, 57, 965509, tzinfo=UTC)),
    (1793286000, datetime(2026, 10, 29, 15, 0, tzinfo=UTC)),
    (datetime(2026, 1, 1, 12), datetime(2026, 1, 1, 12, tzinfo=UTC)),  # naive assumed UTC
    ("", None),
])
def test_to_utc(value, expected):
    assert to_utc(value) == expected


def test_historical_and_sim_schemas_are_disjoint():
    assert not set(MARKET_MD.tables) & set(SIM_MD.tables)
    assert all(not t.startswith("sim_") for t in MARKET_MD.tables)


def test_databases_hold_only_their_own_tables(market_engine, sim_engine):
    assert set(inspect(market_engine).get_table_names()) == set(MARKET_MD.tables)
    assert set(inspect(sim_engine).get_table_names()) == set(SIM_MD.tables)


def trade(tid, price=0.4, count=5.0, ts=None):
    return {"trade_id": tid, "ticker": "T", "created_time": ts or utcnow(), "yes_price": price,
            "no_price": round(1 - price, 4), "count": count, "taker_side": "yes",
            "fetched_at": utcnow()}


def test_upsert_dedupes_and_updates(market_engine):
    assert upsert(market_engine, mkt_trades, [trade("a"), trade("a", price=0.5), trade("b")],
                  ["trade_id"]) == 2
    upsert(market_engine, mkt_trades, [trade("a", price=0.6)], ["trade_id"])
    upsert(market_engine, mkt_trades, [trade("b", price=0.9)], ["trade_id"], update=False)
    with market_engine.connect() as c:
        rows = {r.trade_id: r.yes_price for r in c.execute(select(mkt_trades))}
    assert rows == {"a": 0.6, "b": 0.4}


def test_timestamps_round_trip_as_aware_utc(market_engine):
    ts = datetime(2026, 3, 8, 1, 30, tzinfo=timezone(timedelta(hours=-5)))
    upsert(market_engine, mkt_trades, [trade("z", ts=ts)], ["trade_id"])
    with market_engine.connect() as c:
        got = c.execute(select(mkt_trades.c.created_time)).scalar()
    assert got == ts and got.tzinfo == UTC and got.hour == 6


def test_upsert_chunks_large_batches(market_engine):
    upsert(market_engine, mkt_trades, [trade(f"t{i}") for i in range(1234)], ["trade_id"])
    with market_engine.connect() as c:
        assert c.execute(select(func.count()).select_from(mkt_trades)).scalar() == 1234


def test_trade_checks():
    good, issues = split_valid([
        trade("ok"), trade("zero", price=0.0), trade("neg", count=0),
        trade("future", ts=utcnow() + timedelta(days=3)),
    ], check_trade)
    assert [g["trade_id"] for g in good] == ["ok"]
    assert {i["check_name"] for i in issues} == {"price_range", "nonpositive_count",
                                                 "future_timestamp"}
    assert all("_hard" not in i for i in issues)


def test_observation_vintage_cannot_precede_observation():
    row = {"source": "s", "series_id": "x", "obs_ts": to_utc("2026-01-10"),
           "vintage_ts": to_utc("2026-01-01"), "value": 1.0}
    good, issues = split_valid([row], check_observation)
    assert good == [] and issues[0]["check_name"] == "vintage_before_observation"


def test_ext_observation_keeps_every_vintage(market_engine):
    base = {"source": "fred", "series_id": "UNRATE", "obs_ts": to_utc("2026-01-01"),
            "units": None, "fetched_at": utcnow()}
    upsert(market_engine, ext_observations, [
        {**base, "vintage_ts": to_utc("2026-02-06"), "value": 4.0},
        {**base, "vintage_ts": to_utc("2026-03-06"), "value": 4.1},  # revision
    ], ["source", "series_id", "obs_ts", "vintage_ts"])
    with market_engine.connect() as c:
        vals = c.execute(select(ext_observations.c.value).order_by(ext_observations.c.vintage_ts))
        assert list(vals.scalars()) == [4.0, 4.1]
