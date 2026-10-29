"""Quote lookup and conservative taker-fill simulation.

Rules
* Only buys are simulated (buy YES or buy NO), held to settlement. Buying YES lifts the
  YES asks, which on Kalshi are the NO bids mirrored: ask = 1 - no_bid. Buying NO lifts
  1 - yes_bid.
* Fills never use the midpoint. They walk real ask levels up to the order's limit price.
* Depth source, best first:
    1. an order-book snapshot no older than ``max_book_age`` -> walk levels, taking at most
       ``participation`` of each level's displayed size;
    2. else the last completed candle no older than ``max_quote_age`` -> top-of-book ask
       plus ``slippage`` per contract, size capped at ``unknown_depth_cap``, and the fill is
       flagged ``liquidity_uncertain``;
    3. else no fill (``no_liquidity_data`` or ``stale_quote``).
* Every quote used has a timestamp <= the decision time (asserted).
"""

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import Engine, select

from predlab.db.schema import mkt_candles, mkt_orderbook_snapshots
from predlab.features.pointintime import LookAheadError


@dataclass(frozen=True)
class ExecConfig:
    max_book_age: timedelta = timedelta(hours=1)
    max_quote_age: timedelta = timedelta(hours=2)
    participation: float = 0.5          # max share of a displayed level we assume we get
    unknown_depth_cap: float = 10.0     # contracts, when only top-of-book is known
    slippage: float = 0.01              # $/contract added when depth is unknown
    min_price: float = 0.01
    max_price: float = 0.99


@dataclass
class Quote:
    ts: pd.Timestamp
    source: str                          # "book" | "candle"
    yes_asks: list[tuple[float, float]]  # (price, size) ascending; size inf if unknown
    no_asks: list[tuple[float, float]]
    yes_bid: float | None
    no_bid: float | None


@dataclass
class FillResult:
    qty: float
    avg_price: float
    levels: list[tuple[float, float]] = field(default_factory=list)
    liquidity_uncertain: bool = False
    reason: str = ""


class MarketData:
    """In-memory, time-indexed quotes for a set of tickers."""

    def __init__(self, engine: Engine, tickers, period_minutes: int = 60):
        tickers = list(tickers)
        books, candles = [], []
        with engine.connect() as c:
            for i in range(0, len(tickers), 900):
                chunk = tickers[i:i + 900]
                books.append(pd.DataFrame(c.execute(
                    select(mkt_orderbook_snapshots.c.ticker, mkt_orderbook_snapshots.c.snapshot_ts,
                           mkt_orderbook_snapshots.c.side, mkt_orderbook_snapshots.c.price,
                           mkt_orderbook_snapshots.c.size)
                    .where(mkt_orderbook_snapshots.c.ticker.in_(chunk))).mappings().all()))
                candles.append(pd.DataFrame(c.execute(
                    select(mkt_candles.c.ticker, mkt_candles.c.end_ts, mkt_candles.c.yes_bid_close,
                           mkt_candles.c.yes_ask_close)
                    .where(mkt_candles.c.ticker.in_(chunk),
                           mkt_candles.c.period_minutes == period_minutes)).mappings().all()))
        self._books: dict[str, tuple[list, list]] = {}
        b = pd.concat(books) if books else pd.DataFrame()
        if not b.empty:
            b["snapshot_ts"] = pd.to_datetime(b["snapshot_ts"], utc=True)
            for t, g in b.groupby("ticker"):
                snaps = sorted(g.groupby("snapshot_ts"), key=lambda kv: kv[0])
                self._books[t] = ([ts for ts, _ in snaps], [s for _, s in snaps])
        self._candles: dict[str, tuple[list, np.ndarray]] = {}
        cd = pd.concat(candles) if candles else pd.DataFrame()
        if not cd.empty:
            cd["end_ts"] = pd.to_datetime(cd["end_ts"], utc=True)
            for t, g in cd.sort_values("end_ts").groupby("ticker"):
                self._candles[t] = (list(g["end_ts"]),
                                    g[["yes_bid_close", "yes_ask_close"]].to_numpy(dtype=float))

    def quote(self, ticker: str, ts: pd.Timestamp, cfg: ExecConfig) -> tuple[Quote | None, str]:
        if ticker in self._books:
            times, snaps = self._books[ticker]
            i = bisect_right(times, ts) - 1
            if i >= 0 and ts - times[i] <= cfg.max_book_age:
                return self._book_quote(times[i], snaps[i]), ""
        if ticker in self._candles:
            times, vals = self._candles[ticker]
            i = bisect_right(times, ts) - 1
            if i >= 0:
                if ts - times[i] > cfg.max_quote_age:
                    return None, "stale_quote"
                bid, ask = vals[i]
                yes_asks = [(float(ask), np.inf)] if ask > 0 and not np.isnan(ask) else []
                no_ask = 1 - bid if bid > 0 and not np.isnan(bid) else None
                return Quote(times[i], "candle", yes_asks,
                             [(round(no_ask, 4), np.inf)] if no_ask is not None else [],
                             float(bid) if bid > 0 else None,
                             round(1 - ask, 4) if ask > 0 else None), ""
        return None, "no_liquidity_data"

    @staticmethod
    def _book_quote(ts, snap: pd.DataFrame) -> Quote:
        yes_b = snap[snap["side"] == "yes"]
        no_b = snap[snap["side"] == "no"]
        # YES asks are NO bids mirrored, and vice versa.
        yes_asks = sorted((round(1 - p, 4), s) for p, s in zip(no_b["price"], no_b["size"]))
        no_asks = sorted((round(1 - p, 4), s) for p, s in zip(yes_b["price"], yes_b["size"]))
        return Quote(ts, "book", yes_asks, no_asks,
                     float(yes_b["price"].max()) if len(yes_b) else None,
                     float(no_b["price"].max()) if len(no_b) else None)


def simulate_buy(quote: Quote, side: str, qty: float, limit: float, cfg: ExecConfig,
                 decision_ts: pd.Timestamp) -> FillResult:
    """Fill a marketable limit buy of ``qty`` contracts on ``side`` against ``quote``."""
    if quote.ts > decision_ts:
        raise LookAheadError(f"quote at {quote.ts} used for a decision at {decision_ts}")
    asks = quote.yes_asks if side == "yes" else quote.no_asks
    if not asks:
        return FillResult(0.0, 0.0, reason="no_asks")
    uncertain = quote.source != "book"
    remaining, taken = qty, []
    for price, size in asks:
        px = price + (cfg.slippage if uncertain else 0.0)
        if px > limit or px > cfg.max_price or px < cfg.min_price:
            break
        avail = cfg.unknown_depth_cap if np.isinf(size) else np.floor(size * cfg.participation)
        take = min(remaining, avail)
        if take > 0:
            taken.append((round(px, 4), float(take)))
            remaining -= take
        if remaining <= 0 or uncertain:
            break
    filled = sum(q for _, q in taken)
    if filled == 0:
        return FillResult(0.0, 0.0, liquidity_uncertain=uncertain, reason="limit_or_depth")
    avg = sum(p * q for p, q in taken) / filled
    return FillResult(filled, avg, taken, uncertain,
                      "" if filled >= qty else "partial_fill")
