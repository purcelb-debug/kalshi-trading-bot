"""Kalshi public market-data collector (read-only, unauthenticated).

Uses only documented public GET endpoints: series, events, markets, trades, candlesticks
and order books. No credentials are loaded and no authenticated endpoint is reachable
(see predlab.data.http.KALSHI_READ_PATHS).
"""

from collections.abc import Callable
from datetime import datetime

from sqlalchemy import Engine, select

from predlab.categories import map_kalshi_category
from predlab.config import Settings
from predlab.data.collectors.base import RunStats, persist_issues, tracked_run
from predlab.data.http import KALSHI_READ_PATHS, ReadOnlyClient
from predlab.data.integrity import check_candle, check_market, check_trade, split_valid
from predlab.db.repo import upsert
from predlab.db.schema import (
    mkt_candles,
    mkt_events,
    mkt_markets,
    mkt_orderbook_snapshots,
    mkt_series,
    mkt_trades,
    outcomes,
)
from predlab.timeutil import to_utc, utcnow

SOURCE = "kalshi"
PERIODS = {1, 60, 1440}  # candlestick intervals Kalshi accepts, minutes
MARKET_BATCH = 5000      # markets written per batch while paginating


def _num(x) -> float | None:
    if x is None or x == "":
        return None
    return float(x)


def _price(d: dict, name: str) -> float | None:
    """Prefer the ``*_dollars`` string field; fall back to legacy integer cents."""
    if (v := d.get(f"{name}_dollars")) not in (None, ""):
        return float(v)
    if (v := d.get(name)) not in (None, ""):
        return float(v) / 100.0
    return None


def _qty(d: dict, name: str) -> float | None:
    if (v := d.get(f"{name}_fp")) not in (None, ""):
        return float(v)
    return _num(d.get(name))


def make_client(settings: Settings, **kw) -> ReadOnlyClient:
    return ReadOnlyClient(
        settings.kalshi_base_url,
        allowed_paths=KALSHI_READ_PATHS,
        reads_per_sec=settings.kalshi_reads_per_sec,
        max_retries=settings.http_max_retries,
        timeout_s=settings.http_timeout_s,
        user_agent=settings.user_agent,
        **kw,
    )


# ---------------------------------------------------------------- normalizers

def norm_series(s: dict, fetched_at: datetime) -> dict:
    return {
        "series_ticker": s["ticker"], "source": SOURCE, "title": s.get("title"),
        "source_category": s.get("category"), "category": map_kalshi_category(s.get("category")),
        "frequency": s.get("frequency"), "fee_type": s.get("fee_type"),
        "fee_multiplier": _num(s.get("fee_multiplier")), "raw_json": s, "fetched_at": fetched_at,
    }


def norm_event(e: dict, fetched_at: datetime) -> dict:
    return {
        "event_ticker": e["event_ticker"], "source": SOURCE, "series_ticker": e.get("series_ticker"),
        "title": e.get("title"), "sub_title": e.get("sub_title"),
        "source_category": e.get("category"), "category": map_kalshi_category(e.get("category")),
        "strike_date": to_utc(e.get("strike_date")),
        "mutually_exclusive": e.get("mutually_exclusive"), "raw_json": e, "fetched_at": fetched_at,
    }


def norm_market(m: dict, fetched_at: datetime, series_ticker: str | None,
                category: str | None) -> dict:
    return {
        "ticker": m["ticker"], "source": SOURCE, "event_ticker": m.get("event_ticker"),
        "series_ticker": series_ticker, "category": category, "title": m.get("title"),
        "market_type": m.get("market_type"), "status": m.get("status"),
        "open_time": to_utc(m.get("open_time")), "close_time": to_utc(m.get("close_time")),
        "expiration_time": to_utc(m.get("expiration_time")),
        "result": (m.get("result") or "").lower(), "expiration_value": m.get("expiration_value"),
        "strike_type": m.get("strike_type"), "floor_strike": _num(m.get("floor_strike")),
        "cap_strike": _num(m.get("cap_strike")),
        "yes_bid": _price(m, "yes_bid"), "yes_ask": _price(m, "yes_ask"),
        "no_bid": _price(m, "no_bid"), "no_ask": _price(m, "no_ask"),
        "last_price": _price(m, "last_price"), "volume": _qty(m, "volume"),
        "open_interest": _qty(m, "open_interest"), "updated_time": to_utc(m.get("updated_time")),
        "raw_json": m, "fetched_at": fetched_at,
    }


def norm_trade(t: dict, fetched_at: datetime) -> dict:
    return {
        "trade_id": t["trade_id"], "ticker": t["ticker"],
        "created_time": to_utc(t["created_time"]),
        "yes_price": _price(t, "yes_price"), "no_price": _price(t, "no_price"),
        "count": _qty(t, "count"), "taker_side": t.get("taker_side"), "fetched_at": fetched_at,
    }


def norm_candle(c: dict, ticker: str, period: int, fetched_at: datetime) -> dict:
    p, yb, ya = c.get("price") or {}, c.get("yes_bid") or {}, c.get("yes_ask") or {}

    def g(d, k):
        # Live tier: "<k>_dollars" strings. Archive tier: plain "<k>" dollar *strings*.
        # Legacy payloads: plain "<k>" integer cents.
        if f"{k}_dollars" in d:
            return _num(d[f"{k}_dollars"])
        v = d.get(k)
        if v is None or v == "":
            return None
        return float(v) if isinstance(v, str) else float(v) / 100.0

    return {
        "ticker": ticker, "period_minutes": period, "end_ts": to_utc(c["end_period_ts"]),
        "price_open": g(p, "open"), "price_high": g(p, "high"), "price_low": g(p, "low"),
        "price_close": g(p, "close"), "price_mean": g(p, "mean"),
        "yes_bid_open": g(yb, "open"), "yes_bid_close": g(yb, "close"),
        "yes_ask_open": g(ya, "open"), "yes_ask_close": g(ya, "close"),
        "volume": _qty(c, "volume"), "open_interest": _qty(c, "open_interest"),
        "fetched_at": fetched_at,
    }


def norm_orderbook(payload: dict, ticker: str, snapshot_ts: datetime) -> list[dict]:
    """Flatten a book into one row per (side, price) level. Both sides are bids."""
    rows = []
    if "orderbook_fp" in payload:
        book = payload["orderbook_fp"] or {}
        sides = {"yes": book.get("yes_dollars"), "no": book.get("no_dollars")}
        scale = 1.0
    else:  # legacy: integer cents
        book = payload.get("orderbook") or {}
        sides = {"yes": book.get("yes"), "no": book.get("no")}
        scale = 0.01
    for side, levels in sides.items():
        for price, size in levels or []:
            rows.append({"ticker": ticker, "snapshot_ts": snapshot_ts, "side": side,
                         "price": float(price) * scale, "size": float(size)})
    return rows


# ---------------------------------------------------------------- collector

class KalshiCollector:
    def __init__(self, engine: Engine, client: ReadOnlyClient):
        self.engine = engine
        self.client = client

    async def collect_series(self, series_ticker: str, stats: RunStats) -> dict:
        data = await self.client.get(f"/series/{series_ticker}")
        row = norm_series(data["series"], utcnow())
        stats.add("mkt_series", upsert(self.engine, mkt_series, [row], ["series_ticker"]))
        return row

    async def collect_events(self, series_ticker: str, stats: RunStats,
                             status: str | None = None) -> list[dict]:
        now = utcnow()
        params = {"series_ticker": series_ticker, "limit": 200}
        if status:
            params["status"] = status
        rows = [norm_event(e, now) async for e in self.client.paginate("/events", "events", params)]
        stats.add("mkt_events", upsert(self.engine, mkt_events, rows, ["event_ticker"]))
        return rows

    async def collect_markets(self, stats: RunStats, *, series_ticker: str | None = None,
                              event_ticker: str | None = None, status: str | None = None,
                              category: str | None = None, min_close_ts: int | None = None,
                              max_close_ts: int | None = None, historical: bool = False,
                              keep: Callable[[dict], bool] | None = None,
                              max_markets: int | None = None,
                              batch_size: int = MARKET_BATCH) -> list[dict]:
        """Live tier by default; ``historical=True`` reads the archive tier, which holds
        settled markets older than the cutoff and accepts a single ticker filter only.

        Markets are written in batches as pages arrive, so memory stays flat however large
        the series is. Only rows passing ``keep`` are returned; ``max_markets`` stops the
        download after that many markets and records a data-quality note saying so."""
        now = utcnow()
        params: dict = {"limit": 1000}
        filters = (("series_ticker", series_ticker), ("event_ticker", event_ticker))
        if not historical:
            filters += (("status", status), ("min_close_ts", min_close_ts),
                        ("max_close_ts", max_close_ts))
        elif series_ticker and event_ticker:
            raise ValueError("historical markets accept series_ticker or event_ticker, not both")
        for k, v in filters:
            if v is not None:
                params[k] = v
        path = "/historical/markets" if historical else "/markets"
        kept: list[dict] = []
        batch: list[dict] = []
        seen = 0

        def flush() -> None:
            good, issues = split_valid(batch, check_market)
            persist_issues(self.engine, stats, issues)
            stats.add("mkt_markets", upsert(self.engine, mkt_markets, good, ["ticker"]))
            stats.add("outcomes", self._record_outcomes(good, now))
            kept.extend(m for m in good if keep is None or keep(m))
            batch.clear()

        async for m in self.client.paginate(path, "markets", params):
            batch.append(norm_market(m, now, series_ticker, category))
            seen += 1
            if len(batch) >= batch_size:
                flush()
            if max_markets is not None and seen >= max_markets:
                stats.warnings.append(
                    f"{path} {series_ticker or event_ticker}: stopped at max_markets={max_markets}")
                break
        if batch:
            flush()
        return kept

    def _record_outcomes(self, markets: list[dict], now: datetime) -> int:
        """Append an outcome row only when a binary result is new or has changed."""
        resolved = [m for m in markets if m["result"] in ("yes", "no")]
        if not resolved:
            return 0
        tickers = [m["ticker"] for m in resolved]
        latest: dict[str, str] = {}
        with self.engine.connect() as conn:
            for i in range(0, len(tickers), 900):   # stay under SQLite's bound-variable limit
                q = (select(outcomes.c.ticker, outcomes.c.result, outcomes.c.recorded_at)
                     .where(outcomes.c.ticker.in_(tickers[i:i + 900]))
                     .order_by(outcomes.c.recorded_at))
                for r in conn.execute(q):
                    latest[r.ticker] = r.result
        new = [{
            "ticker": m["ticker"], "recorded_at": now, "result": m["result"],
            "resolved_ts": m.get("expiration_time") or m.get("close_time"),
            "source": SOURCE, "revised": m["ticker"] in latest,
        } for m in resolved if latest.get(m["ticker"]) != m["result"]]
        return upsert(self.engine, outcomes, new, ["ticker", "recorded_at"], update=False)

    async def get_cutoffs(self) -> dict[str, datetime]:
        """Archive cutoffs: data older than these lives only under /historical/."""
        data = await self.client.get("/historical/cutoff")
        return {k: to_utc(v) for k, v in data.items()}

    async def collect_trades(self, ticker: str, stats: RunStats, min_ts: int | None = None,
                             max_ts: int | None = None, historical: bool = False) -> int:
        now = utcnow()
        params: dict = {"ticker": ticker, "limit": 1000}
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        path = "/historical/trades" if historical else "/markets/trades"
        rows = [norm_trade(t, now) async for t in self.client.paginate(path, "trades", params)]
        good, issues = split_valid(rows, check_trade)
        persist_issues(self.engine, stats, issues)
        n = upsert(self.engine, mkt_trades, good, ["trade_id"], update=False)
        stats.add("mkt_trades", n)
        return n

    async def collect_candles(self, series_ticker: str, ticker: str, start_ts: int, end_ts: int,
                              stats: RunStats, period_minutes: int = 60,
                              historical: bool = False) -> int:
        if period_minutes not in PERIODS:
            raise ValueError(f"period_minutes must be one of {sorted(PERIODS)}")
        if start_ts <= 0:
            raise ValueError("start_ts must be a positive epoch timestamp")
        now = utcnow()
        path = (f"/historical/markets/{ticker}/candlesticks" if historical
                else f"/series/{series_ticker}/markets/{ticker}/candlesticks")
        data = await self.client.get(
            path, {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_minutes})
        rows = [norm_candle(c, ticker, period_minutes, now) for c in data.get("candlesticks") or []]
        good, issues = split_valid(rows, check_candle)
        persist_issues(self.engine, stats, issues)
        n = upsert(self.engine, mkt_candles, good, ["ticker", "period_minutes", "end_ts"])
        stats.add("mkt_candles", n)
        return n

    async def snapshot_orderbook(self, ticker: str, stats: RunStats) -> int:
        """Order-book history only exists from the moment we start snapshotting it."""
        data = await self.client.get(f"/markets/{ticker}/orderbook")
        rows = norm_orderbook(data, ticker, utcnow())
        n = upsert(self.engine, mkt_orderbook_snapshots, rows,
                   ["ticker", "snapshot_ts", "side", "price"], update=False)
        stats.add("mkt_orderbook_snapshots", n)
        return n

    async def collect_series_bundle(self, series_ticker: str, *, status: str | None = None,
                                    with_trades: bool = False, with_candles: bool = False,
                                    with_orderbooks: bool = False, include_archive: bool = True,
                                    candle_period: int = 60,
                                    since: datetime | None = None,
                                    max_markets: int | None = None) -> RunStats:
        """Series -> events -> markets (-> trades / candles / books) in one tracked run.

        With ``include_archive`` (and a status of settled or unspecified), markets that
        settled before the archive cutoff are read from /historical/ too, and each market's
        trades/candles are routed to whichever tier holds them.
        """
        async with tracked_run(self.engine, SOURCE) as stats:
            series = await self.collect_series(series_ticker, stats)
            await self.collect_events(series_ticker, stats, status=None)
            # Metadata and outcomes are stored for every market; only markets that close
            # after ``since`` are kept in memory for the heavy trade/candle/book pass.
            def recent(m: dict) -> bool:
                return since is None or (m["close_time"] is not None and m["close_time"] >= since)

            markets = await self.collect_markets(stats, series_ticker=series_ticker, status=status,
                                                 category=series["category"], keep=recent,
                                                 max_markets=max_markets)
            archived: set[str] = set()
            cut = {}
            if include_archive and status in (None, "settled"):
                cut = await self.get_cutoffs()
                old = await self.collect_markets(stats, series_ticker=series_ticker,
                                                 category=series["category"], historical=True,
                                                 keep=recent, max_markets=max_markets)
                archived = {m["ticker"] for m in old}
                live = {x["ticker"] for x in markets}
                markets += [m for m in old if m["ticker"] not in live]
            trade_cut = cut.get("trades_created_ts")
            for m in markets:
                if with_trades:
                    # A market can straddle the cutoff; trade_id dedupe absorbs any overlap.
                    if trade_cut and m["open_time"] and m["open_time"] < trade_cut:
                        await self.collect_trades(m["ticker"], stats, historical=True)
                    if not trade_cut or not m["close_time"] or m["close_time"] >= trade_cut:
                        await self.collect_trades(m["ticker"], stats)
                if with_candles and m["open_time"] and m["close_time"]:
                    await self.collect_candles(series_ticker, m["ticker"],
                                               int(m["open_time"].timestamp()),
                                               int(m["close_time"].timestamp()), stats,
                                               candle_period, historical=m["ticker"] in archived)
                if with_orderbooks and m["status"] in ("active", "open"):
                    await self.snapshot_orderbook(m["ticker"], stats)
        return stats
