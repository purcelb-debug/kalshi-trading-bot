"""Async, GET-only HTTP client used by every collector.

Safety: this is the only module that constructs an HTTP client. It exposes ``get`` and
``paginate`` and nothing else, so no code in predlab can send a write request to a market.
Paths can additionally be restricted to an allowlist of read-only endpoints.
"""

import asyncio
import random
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx

from predlab.logs import get_logger

log = get_logger(__name__)

# Never allowed on any host, whatever the allowlist says.
FORBIDDEN_PATH = re.compile(r"(portfolio|/orders?\b|/order_groups|/positions|/fills|/balance)", re.IGNORECASE)

KALSHI_READ_PATHS = [
    re.compile(p)
    for p in (
        r"^/exchange/status$",
        r"^/series$",
        r"^/series/[^/]+$",
        r"^/events$",
        r"^/events/[^/]+$",
        r"^/markets$",
        r"^/markets/trades$",
        r"^/markets/[^/]+$",
        r"^/markets/[^/]+/orderbook$",
        r"^/series/[^/]+/markets/[^/]+/candlesticks$",
        # Archive tier: settled markets/candles/trades older than GET /historical/cutoff.
        # User-scoped archive paths (fills, orders, positions) stay blocked by FORBIDDEN_PATH.
        r"^/historical/cutoff$",
        r"^/historical/markets$",
        r"^/historical/markets/[^/]+$",
        r"^/historical/markets/[^/]+/candlesticks$",
        r"^/historical/trades$",
    )
]

RETRY_STATUS = {429, 500, 502, 503, 504}


class ReadOnlyViolation(PermissionError):
    pass


class HttpError(RuntimeError):
    def __init__(self, status: int | None, url: str, detail: str):
        super().__init__(f"{status} {url}: {detail}")
        self.status = status
        self.url = url


class RateLimiter:
    """Minimum spacing between requests (simple, conservative)."""

    def __init__(self, per_sec: float):
        self._interval = 1.0 / per_sec if per_sec > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next = max(now, self._next) + self._interval


class ReadOnlyClient:
    def __init__(
        self,
        base_url: str,
        *,
        allowed_paths: list[re.Pattern] | None = None,
        reads_per_sec: float = 5.0,
        max_retries: int = 5,
        timeout_s: float = 30.0,
        user_agent: str = "predlab-research",
        default_params: dict | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.allowed_paths = allowed_paths
        self.max_retries = max_retries
        self.default_params = default_params or {}
        self._limiter = RateLimiter(reads_per_sec)
        self._sleep = sleep
        self._client = httpx.AsyncClient(
            timeout=timeout_s, headers={"User-Agent": user_agent, "Accept": "application/json"},
            transport=transport, follow_redirects=True,
        )
        self.requests_made = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # Explicitly refuse any non-read verb.
    def _forbidden(self, *args, **kwargs):
        raise ReadOnlyViolation("predlab is read-only: only GET requests are permitted")

    post = put = patch = delete = request = stream = send = _forbidden

    def _check_path(self, path: str) -> None:
        if FORBIDDEN_PATH.search(path):
            raise ReadOnlyViolation(f"path not permitted: {path}")
        if self.allowed_paths is not None and not any(p.match(path) for p in self.allowed_paths):
            raise ReadOnlyViolation(f"path not in read-only allowlist: {path}")

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        return min(0.5 * (2**attempt), 30.0) * (0.5 + random.random() / 2)

    async def get(self, path: str, params: dict | None = None) -> dict | list:
        self._check_path(path)
        url = self.base_url + path
        merged = {**self.default_params, **(params or {})}
        last_detail = ""
        for attempt in range(self.max_retries + 1):
            await self._limiter.acquire()
            self.requests_made += 1
            try:
                resp = await self._client.get(url, params=merged)
            except httpx.TransportError as e:
                last_detail = f"transport error: {e!r}"
                status, retry_after = None, None
            else:
                if resp.status_code == 200:
                    return resp.json()
                status, retry_after = resp.status_code, resp.headers.get("Retry-After")
                last_detail = resp.text[:300]
                if status not in RETRY_STATUS:
                    raise HttpError(status, url, last_detail)
            if attempt == self.max_retries:
                break
            delay = self._backoff(attempt, retry_after)
            log.warning("http_retry", url=url, status=status, attempt=attempt + 1, delay_s=round(delay, 2))
            await self._sleep(delay)
        raise HttpError(status, url, f"gave up after {self.max_retries + 1} attempts: {last_detail}")

    async def paginate(
        self, path: str, items_key: str, params: dict | None = None,
        *, cursor_param: str = "cursor", max_pages: int = 10_000,
    ) -> AsyncIterator[dict]:
        """Yield items across cursor-paginated pages. Stops on empty/repeated cursor."""
        params = dict(params or {})
        seen: set[str] = set()
        for _ in range(max_pages):
            page = await self.get(path, params)
            for item in page.get(items_key) or []:
                yield item
            cursor = page.get("cursor")
            if not cursor or cursor in seen:
                return
            seen.add(cursor)
            params[cursor_param] = cursor
        log.warning("pagination_cap_reached", path=path, max_pages=max_pages)
