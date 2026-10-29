"""Kalshi collector against mocked responses shaped like the live API (checked 2026-10-29)."""

import httpx
import pytest
import respx
from sqlalchemy import func, select

from predlab.config import Settings
from predlab.data.collectors.kalshi import KalshiCollector, make_client, norm_orderbook
from predlab.data.http import HttpError
from predlab.db.schema import (
    data_quality_issues,
    mkt_candles,
    mkt_events,
    mkt_markets,
    mkt_orderbook_snapshots,
    mkt_trades,
    outcomes,
    source_health,
)
from predlab.timeutil import to_utc

BASE = "https://kalshi.test/trade-api/v2"
SERIES = {"series": {"ticker": "KXHIGHNY", "title": "Highest temperature in NYC",
                     "category": "Climate and Weather", "frequency": "daily",
                     "fee_type": "quadratic", "fee_multiplier": 1}}
EVENTS = {"cursor": "", "events": [{
    "event_ticker": "KXHIGHNY-26OCT28", "series_ticker": "KXHIGHNY",
    "category": "Climate and Weather", "title": "Highest temperature in NYC on Oct 28, 2026?",
    "strike_date": "2026-10-29T05:00:00Z", "mutually_exclusive": True}]}


def market(ticker, result="no", status="finalized", yes_bid="0.0000", yes_ask="0.0100"):
    return {
        "ticker": ticker, "event_ticker": "KXHIGHNY-26OCT28", "market_type": "binary",
        "status": status, "result": result, "title": f"{ticker}?",
        "open_time": "2026-10-27T14:00:00Z", "close_time": "2026-10-29T05:00:00Z",
        "expiration_time": "2026-11-04T14:00:00Z", "floor_strike": 67, "strike_type": "greater",
        "yes_bid_dollars": yes_bid, "yes_ask_dollars": yes_ask, "no_bid_dollars": "0.9900",
        "no_ask_dollars": "1.0000", "last_price_dollars": "0.0100", "volume_fp": "966.24",
        "open_interest_fp": "966.24", "updated_time": "2026-10-29T06:00:00Z",
    }


@pytest.fixture
def settings():
    return Settings(kalshi_base_url=BASE, kalshi_reads_per_sec=1000, http_max_retries=1)


CUTOFF = {"market_settled_ts": "2026-08-30T00:00:00Z", "trades_created_ts": "2026-08-30T00:00:00Z",
          "orders_updated_ts": "2026-10-15T00:00:00Z",
          "market_positions_last_updated_ts": "2026-08-30T00:00:00Z"}


def mock_bundle(markets_pages, archived=()):
    respx.get(f"{BASE}/series/KXHIGHNY").mock(return_value=httpx.Response(200, json=SERIES))
    respx.get(f"{BASE}/events").mock(return_value=httpx.Response(200, json=EVENTS))
    respx.get(f"{BASE}/historical/cutoff").mock(return_value=httpx.Response(200, json=CUTOFF))
    respx.get(f"{BASE}/historical/markets").mock(return_value=httpx.Response(
        200, json={"markets": list(archived), "cursor": ""}))
    return respx.get(f"{BASE}/markets").mock(
        side_effect=[httpx.Response(200, json=p) for p in markets_pages])


@respx.mock
async def test_bundle_normalizes_and_paginates(market_engine, settings):
    route = mock_bundle([
        {"markets": [market("KXHIGHNY-26OCT28-T67")], "cursor": "abc"},
        {"markets": [market("KXHIGHNY-26OCT28-T60", result="yes")], "cursor": ""},
    ])
    async with make_client(settings) as client:
        stats = await KalshiCollector(market_engine, client).collect_series_bundle(
            "KXHIGHNY", status="settled")
    assert route.call_count == 2
    assert stats.rows["mkt_markets"] == 2 and stats.rows["outcomes"] == 2
    with market_engine.connect() as c:
        m = c.execute(select(mkt_markets).where(mkt_markets.c.ticker == "KXHIGHNY-26OCT28-T67")).one()
        ev = c.execute(select(mkt_events)).one()
        health = c.execute(select(source_health)).one()
    assert m.category == "weather" and m.series_ticker == "KXHIGHNY"
    assert m.yes_ask == pytest.approx(0.01) and m.no_bid == pytest.approx(0.99)
    assert m.volume == pytest.approx(966.24)
    assert m.close_time == to_utc("2026-10-29T05:00:00Z") and m.close_time.tzinfo is not None
    assert ev.category == "weather"
    assert health.source == "kalshi" and health.consecutive_failures == 0


@respx.mock
async def test_rerun_is_idempotent_and_revisions_are_appended(market_engine, settings):
    tick = "KXHIGHNY-26OCT28-T67"
    mock_bundle([{"markets": [market(tick, result="no")], "cursor": ""},
                 {"markets": [market(tick, result="no")], "cursor": ""},
                 {"markets": [market(tick, result="yes")], "cursor": ""}])
    async with make_client(settings) as client:
        col = KalshiCollector(market_engine, client)
        for _ in range(3):
            await col.collect_series_bundle("KXHIGHNY")
    with market_engine.connect() as c:
        n_markets = c.execute(select(func.count()).select_from(mkt_markets)).scalar()
        outs = c.execute(select(outcomes).order_by(outcomes.c.recorded_at)).all()
    assert n_markets == 1
    assert [(o.result, o.revised) for o in outs] == [("no", False), ("yes", True)]


@respx.mock
async def test_integrity_issues_recorded_and_bad_rows_excluded(market_engine, settings):
    bad_price = market("BAD-PRICE", yes_ask="1.5000")
    crossed = market("CROSSED", yes_bid="0.6000", yes_ask="0.5000")
    unresolved = market("NO-RESULT", result="", status="finalized")
    mock_bundle([{"markets": [bad_price, crossed, unresolved], "cursor": ""}])
    async with make_client(settings) as client:
        await KalshiCollector(market_engine, client).collect_series_bundle("KXHIGHNY")
    with market_engine.connect() as c:
        tickers = set(c.execute(select(mkt_markets.c.ticker)).scalars())
        checks = {(r.entity_id, r.check_name) for r in c.execute(select(data_quality_issues))}
    assert tickers == {"CROSSED", "NO-RESULT"}  # hard error excluded, soft issues kept
    assert ("BAD-PRICE", "yes_ask_range") in checks
    assert ("CROSSED", "crossed_book") in checks
    assert ("NO-RESULT", "settled_without_binary_result") in checks


@respx.mock
async def test_trades_candles_orderbook(market_engine, settings):
    tick = "KXHIGHNY-26OCT28-T67"
    respx.get(f"{BASE}/markets/trades").mock(return_value=httpx.Response(200, json={
        "cursor": "", "trades": [
            {"trade_id": "t1", "ticker": tick, "created_time": "2026-10-29T04:59:57.965509Z",
             "yes_price_dollars": "0.0100", "no_price_dollars": "0.9900", "count_fp": "93.51",
             "taker_side": "yes"},
            {"trade_id": "t1", "ticker": tick, "created_time": "2026-10-29T04:59:57.965509Z",
             "yes_price_dollars": "0.0100", "no_price_dollars": "0.9900", "count_fp": "93.51",
             "taker_side": "yes"}]}))  # duplicate within the page
    respx.get(f"{BASE}/series/KXHIGHNY/markets/{tick}/candlesticks").mock(
        return_value=httpx.Response(200, json={"candlesticks": [
            {"end_period_ts": 1793286000, "open_interest_fp": "20.00", "volume_fp": "20.00",
             "price": {"open_dollars": "0.0100", "high_dollars": "0.0300", "low_dollars": "0.0100",
                       "close_dollars": "0.0200", "mean_dollars": "0.0150"},
             "yes_ask": {"open_dollars": "0.9900", "close_dollars": "0.0300"},
             "yes_bid": {"open_dollars": "0.0000", "close_dollars": "0.0100"}},
            {"end_period_ts": 1793289600, "volume_fp": "0.00",
             "price": {"previous_dollars": "0.0100"},
             "yes_ask": {"close_dollars": "0.0100"}, "yes_bid": {"close_dollars": "0.0000"}}]}))
    respx.get(f"{BASE}/markets/{tick}/orderbook").mock(return_value=httpx.Response(200, json={
        "orderbook_fp": {"yes_dollars": [["0.0100", "387.00"]],
                         "no_dollars": [["0.9800", "1913.00"], ["0.9900", "10.00"]]}}))
    async with make_client(settings) as client:
        col = KalshiCollector(market_engine, client)
        from predlab.data.collectors.base import RunStats
        st = RunStats("kalshi")
        assert await col.collect_trades(tick, st) == 1
        assert await col.collect_candles("KXHIGHNY", tick, 1793280000, 1793289600, st) == 2
        assert await col.snapshot_orderbook(tick, st) == 3
        with pytest.raises(ValueError):
            await col.collect_candles("KXHIGHNY", tick, 1, 2, st, period_minutes=5)
    with market_engine.connect() as c:
        tr = c.execute(select(mkt_trades)).one()
        cs = c.execute(select(mkt_candles).order_by(mkt_candles.c.end_ts)).all()
        ob = c.execute(select(mkt_orderbook_snapshots)).all()
    assert tr.yes_price == pytest.approx(0.01) and tr.count == pytest.approx(93.51)
    assert cs[0].price_close == pytest.approx(0.02) and cs[0].yes_ask_close == pytest.approx(0.03)
    assert cs[1].price_close is None  # no trades that hour: no fabricated price
    assert {(r.side, r.price) for r in ob} == {("yes", 0.01), ("no", 0.98), ("no", 0.99)}


def test_legacy_cent_orderbook_is_scaled():
    from predlab.timeutil import utcnow
    rows = norm_orderbook({"orderbook": {"yes": [[45, 10]], "no": None}}, "T", utcnow())
    assert rows == [{"ticker": "T", "snapshot_ts": rows[0]["snapshot_ts"], "side": "yes",
                     "price": 0.45, "size": 10.0}]


@respx.mock
async def test_failure_records_health(market_engine, settings):
    respx.get(f"{BASE}/series/KXHIGHNY").mock(return_value=httpx.Response(503))
    async with make_client(settings) as client:
        col = KalshiCollector(market_engine, client)
        client._sleep = _nosleep
        with pytest.raises(HttpError):
            await col.collect_series_bundle("KXHIGHNY")
        with pytest.raises(HttpError):
            await col.collect_series_bundle("KXHIGHNY")
    with market_engine.connect() as c:
        h = c.execute(select(source_health)).one()
    assert h.consecutive_failures == 2 and h.last_success is None and "503" in h.last_error


async def _nosleep(_):
    return None


@respx.mock
async def test_archive_tier_is_merged_and_trades_routed_by_cutoff(market_engine, settings):
    old = {**market("KXHIGHNY-26AUG01-T80", result="yes"),
           "open_time": "2026-07-30T14:00:00Z", "close_time": "2026-08-02T05:00:00Z"}
    straddle = {**market("KXHIGHNY-26AUG29-T80"),
                "open_time": "2026-08-28T14:00:00Z", "close_time": "2026-08-31T05:00:00Z"}
    mock_bundle([{"markets": [straddle], "cursor": ""}], archived=[old])
    def trades(tid):
        return httpx.Response(200, json={"cursor": "", "trades": [{
            "trade_id": tid, "ticker": "x", "created_time": "2026-08-01T00:00:00Z",
            "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "1.00"}]})
    hist = respx.get(f"{BASE}/historical/trades").mock(side_effect=[trades("h1"), trades("h2")])
    live = respx.get(f"{BASE}/markets/trades").mock(side_effect=[trades("l1")])
    async with make_client(settings) as client:
        st = await KalshiCollector(market_engine, client).collect_series_bundle(
            "KXHIGHNY", with_trades=True)
    assert st.rows["mkt_markets"] == 2
    # old market: archive only; straddling market: archive and live
    assert [c.request.url.params["ticker"] for c in hist.calls] == [
        "KXHIGHNY-26AUG29-T80", "KXHIGHNY-26AUG01-T80"]
    assert [c.request.url.params["ticker"] for c in live.calls] == ["KXHIGHNY-26AUG29-T80"]
    with market_engine.connect() as c:
        assert c.execute(select(func.count()).select_from(mkt_trades)).scalar() == 3
        assert c.execute(select(func.count()).select_from(outcomes)).scalar() == 2


async def test_user_scoped_archive_paths_stay_blocked(settings):
    from predlab.data.http import ReadOnlyViolation
    async with make_client(settings) as client:
        for path in ("/historical/fills", "/historical/orders", "/historical/positions"):
            with pytest.raises(ReadOnlyViolation):
                await client.get(path)
