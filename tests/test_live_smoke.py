"""Opt-in smoke test against Kalshi's real public endpoints: ``pytest -m live``.

Read-only and unauthenticated. Confirms the response shapes the normalizers expect.
"""

import pytest

from predlab.config import Settings
from predlab.data.collectors.kalshi import KalshiCollector, make_client

pytestmark = pytest.mark.live


async def test_kalshi_public_bundle(market_engine):
    async with make_client(Settings(kalshi_reads_per_sec=5)) as client:
        col = KalshiCollector(market_engine, client)
        from predlab.data.collectors.base import RunStats
        st = RunStats("kalshi")
        series = await col.collect_series("KXHIGHNY", st)
        page = await client.get("/markets", {"series_ticker": "KXHIGHNY", "status": "settled",
                                             "limit": 3})
        assert series["category"] == "weather"
        assert page["markets"] and "yes_ask_dollars" in page["markets"][0]
        ticker = page["markets"][0]["ticker"]
        await col.collect_trades(ticker, st)
        cut = await col.get_cutoffs()
        assert cut["market_settled_ts"] is not None and cut["trades_created_ts"] is not None
        old = await client.get("/historical/markets", {"series_ticker": "KXHIGHNY", "limit": 2})
        assert old["markets"] and "yes_ask_dollars" in old["markets"][0]
        assert client.requests_made >= 5
