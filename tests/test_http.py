import time

import httpx
import pytest
import respx

from predlab.data.http import (
    KALSHI_READ_PATHS,
    HttpError,
    RateLimiter,
    ReadOnlyClient,
    ReadOnlyViolation,
)

BASE = "https://api.example.test/v2"


def client(fake_sleep, **kw):
    return ReadOnlyClient(BASE, reads_per_sec=1000, max_retries=3, sleep=fake_sleep, **kw)


@respx.mock
async def test_retries_on_429_and_5xx_then_succeeds(fake_sleep):
    route = respx.get(f"{BASE}/markets").mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "2"}),
        httpx.Response(503),
        httpx.Response(200, json={"ok": True}),
    ])
    async with client(fake_sleep) as c:
        assert await c.get("/markets") == {"ok": True}
    assert route.call_count == 3
    assert fake_sleep.calls[0] == 2.0           # Retry-After honored
    assert 0 < fake_sleep.calls[1] <= 1.0       # exponential backoff, attempt 2, with jitter


@respx.mock
async def test_transport_errors_retry_then_give_up(fake_sleep):
    respx.get(f"{BASE}/markets").mock(side_effect=httpx.ConnectError("dns failure"))
    async with client(fake_sleep) as c:
        with pytest.raises(HttpError, match="gave up after 4 attempts"):
            await c.get("/markets")
    assert len(fake_sleep.calls) == 3


@respx.mock
async def test_client_errors_are_not_retried(fake_sleep):
    route = respx.get(f"{BASE}/markets/NOPE").mock(return_value=httpx.Response(404, text="nf"))
    async with client(fake_sleep) as c:
        with pytest.raises(HttpError) as ei:
            await c.get("/markets/NOPE")
    assert ei.value.status == 404 and route.call_count == 1 and fake_sleep.calls == []


async def test_write_verbs_are_impossible(fake_sleep):
    async with client(fake_sleep) as c:
        for verb in ("post", "put", "patch", "delete", "request", "stream", "send"):
            with pytest.raises(ReadOnlyViolation):
                getattr(c, verb)("/portfolio/orders", json={"side": "yes"})


@pytest.mark.parametrize("path", [
    "/portfolio/orders", "/portfolio/balance", "/orders", "/portfolio/positions",
    "/markets/X/orders",
])
async def test_trading_paths_blocked_even_without_allowlist(fake_sleep, path):
    async with client(fake_sleep) as c:
        with pytest.raises(ReadOnlyViolation):
            await c.get(path)


async def test_allowlist_rejects_unlisted_paths(fake_sleep):
    async with client(fake_sleep, allowed_paths=KALSHI_READ_PATHS) as c:
        with pytest.raises(ReadOnlyViolation):
            await c.get("/login")
        with pytest.raises(ReadOnlyViolation):
            await c.get("/api_keys")


@pytest.mark.parametrize("path", [
    "/series/KXHIGHNY", "/events", "/markets", "/markets/trades", "/markets/T-1",
    "/markets/T-1/orderbook", "/series/S/markets/T-1/candlesticks", "/exchange/status",
])
def test_allowlist_accepts_public_read_paths(path):
    assert any(p.match(path) for p in KALSHI_READ_PATHS)


@respx.mock
async def test_pagination_follows_cursor_and_stops_on_repeat(fake_sleep):
    route = respx.get(f"{BASE}/markets").mock(side_effect=[
        httpx.Response(200, json={"markets": [{"i": 1}, {"i": 2}], "cursor": "c1"}),
        httpx.Response(200, json={"markets": [{"i": 3}], "cursor": "c2"}),
        httpx.Response(200, json={"markets": [{"i": 4}], "cursor": "c2"}),  # repeated cursor
    ])
    async with client(fake_sleep) as c:
        items = [m async for m in c.paginate("/markets", "markets", {"limit": 2})]
    assert [m["i"] for m in items] == [1, 2, 3, 4]
    assert route.calls[1].request.url.params["cursor"] == "c1"
    assert route.calls[2].request.url.params["cursor"] == "c2"


@respx.mock
async def test_pagination_stops_on_empty_cursor(fake_sleep):
    respx.get(f"{BASE}/events").mock(return_value=httpx.Response(
        200, json={"events": [{"e": 1}], "cursor": ""}))
    async with client(fake_sleep) as c:
        assert len([e async for e in c.paginate("/events", "events")]) == 1


async def test_rate_limiter_spaces_requests():
    lim = RateLimiter(per_sec=50)  # 20 ms spacing
    t0 = time.monotonic()
    for _ in range(6):
        await lim.acquire()
    assert time.monotonic() - t0 >= 0.09  # 5 gaps x 20 ms, minus scheduler slack
