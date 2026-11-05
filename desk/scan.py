"""Broad scan for the ``desk@ALL`` deployment: every open Kalshi market closing soon, all
categories, filtered and ranked in code before the Scanner role sees any of it.

Read-only public endpoint (``GET /markets``). Multivariate combo markets are excluded.
Filters: both sides quoted, mid inside [min_mid, max_mid], spread <= max_spread, volume >=
min_volume, at least ``scan_min_hours`` to close. Ranking: 24h volume, at most
``per_event`` markets per event, so one busy event can't fill the table.
"""

import asyncio
import re
import time

import pandas as pd

from desk.config import DeskSettings
from predlab.config import get_settings
from predlab.data.collectors.kalshi import _qty, make_client, norm_market
from predlab.timeutil import utcnow

COLUMNS = ["ticker", "event_ticker", "title", "close_time", "yes_bid", "yes_ask", "volume",
           "volume_24h", "open_interest"]


async def _fetch(horizon_h: float) -> list[dict]:
    now = int(time.time())
    params = {"status": "open", "min_close_ts": now, "max_close_ts": now + int(horizon_h * 3600),
              "limit": 1000, "mve_filter": "exclude"}
    out, ts = [], utcnow()
    async with make_client(get_settings()) as c:
        async for m in c.paginate("/markets", "markets", params):
            row = norm_market(m, ts, None, None)
            out.append({k: row.get(k) for k in COLUMNS} | {"volume_24h": _qty(m, "volume_24h")})
    return out


def filter_rank(raw: pd.DataFrame, cfg: DeskSettings, *, exclude: set[str], now,
                only: set[str] | None = None) -> pd.DataFrame:
    """Code-side filters and ranking; pure, so it is tested without the network."""
    if raw.empty:
        return raw
    t = raw.copy()
    if only is not None:              # a re-review of named markets
        t = t[t["ticker"].isin(only)]
    t = t[t["yes_bid"].fillna(0).gt(0) & t["yes_ask"].fillna(1).lt(1)]
    t["market_mid"] = (t["yes_bid"] + t["yes_ask"]) / 2
    t["spread"] = t["yes_ask"] - t["yes_bid"]
    t = t[t["market_mid"].between(cfg.scan_min_mid, cfg.scan_max_mid)
          & t["spread"].le(cfg.scan_max_spread + 1e-9)
          & t["volume"].fillna(0).ge(cfg.scan_min_volume)
          & ~t["ticker"].isin(exclude)]
    t["event_ticker"] = t["event_ticker"].fillna(t["ticker"])
    if getattr(cfg, "scan_exclude_prefixes", None):
        pat = "^(?:" + "|".join(re.escape(p) for p in cfg.scan_exclude_prefixes) + ")"
        t = t[~t["ticker"].str.contains(pat, case=False, regex=True)]
    t["hours_to_close"] = (pd.to_datetime(t["close_time"], utc=True) - pd.Timestamp(now)
                           ).dt.total_seconds() / 3600
    t = t[t["hours_to_close"].ge(cfg.scan_min_hours)]
    t = t.sort_values(["volume_24h", "volume"], ascending=False, na_position="last")
    t = t.groupby("event_ticker", sort=False).head(cfg.scan_per_event)
    return t.head(cfg.scan_max_rows).reset_index(drop=True)


def scan_table(cfg: DeskSettings, *, exclude: set[str], now, fetch=None,
               spot=None, only: set[str] | None = None) -> pd.DataFrame:
    """``spot``: ticker list -> {ticker: underlying text}. Live by default; tests that pass
    their own ``fetch`` get no lookups unless they pass ``spot`` too."""
    rows = fetch() if fetch is not None else asyncio.run(_fetch(cfg.scan_horizon_h))
    t = filter_rank(pd.DataFrame(rows, columns=COLUMNS), cfg, exclude=exclude, now=now,
                    only=only)
    if spot is None and fetch is None and cfg.scan_spot:
        from desk.spot import annotate
        spot = annotate
    if spot is not None and len(t):
        t["underlying"] = t["ticker"].map(spot(t["ticker"].tolist()))
    return t


def scan_text(t: pd.DataFrame, now) -> str:
    def f(v, spec=".2f"):
        return "n/a" if pd.isna(v) else format(v, spec)

    has = "underlying" in t.columns
    rows = ["ticker | event | market | hours to close | mid | spread | 24h volume"
            + (" | underlying now" if has else "")]
    for r in t.itertuples():
        u = (" | " + (r.underlying if isinstance(r.underlying, str) else "n/a")) if has else ""
        rows.append(f"{r.ticker} | {r.event_ticker} | {r.title} | {f(r.hours_to_close, '.1f')} | "
                    f"{f(r.market_mid)} | {f(r.spread)} | {f(r.volume_24h, '.0f')}{u}")
    note = ("\n'underlying now' is a live value fetched by code just now (spot price, futures "
            "or index level, or the latest NWS forecast). Treat it as HARD data; the official "
            "settlement source can differ slightly.") if has else ""
    return (f"UTC time: {pd.Timestamp(now):%Y-%m-%d %H:%M}. REAL MONEY (small, capped account).\n"
            f"Open Kalshi markets closing within the next hours, all categories, already "
            f"filtered for liquidity and ranked by 24h volume.{note}\n\nMARKETS\n"
            + "\n".join(rows))
