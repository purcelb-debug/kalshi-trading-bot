"""The broad desk's paper record: each BUY's p_yes against the market mid at that moment,
scored on settled markets. The desk trades real money again only when this beats the
market on enough picks (owner's rule, 2026-11-06)."""

import pandas as pd
from sqlalchemy import Engine, select

from predlab.config import get_settings
from predlab.data.collectors.kalshi import make_client
from predlab.db.schema import forecasts

MIN_SETTLED = 50


def desk_picks(sim: Engine, deployment: str) -> pd.DataFrame:
    """First BUY per market (ticker, p, market): withdrawn records and picks without a
    market mid are left out."""
    with sim.connect() as c:
        rows = c.execute(select(forecasts.c.ticker, forecasts.c.p, forecasts.c.market_ref_p,
                                forecasts.c.explanation, forecasts.c.created_ts)
                         .where(forecasts.c.model_version == deployment)
                         .order_by(forecasts.c.created_ts)).all()
    keep = [(r.ticker, r.p, r.market_ref_p) for r in rows
            if r.market_ref_p is not None and not (r.explanation or {}).get("withdrawn")]
    t = pd.DataFrame(keep, columns=["ticker", "p", "market"])
    return t.drop_duplicates("ticker", keep="first").reset_index(drop=True)


async def fetch_results(tickers: list[str]) -> dict[str, str]:
    """Market -> 'yes' or 'no' for the settled ones."""
    out = {}
    async with make_client(get_settings()) as c:
        for t in tickers:
            r = ((await c.get(f"/markets/{t}")).get("market") or {}).get("result") or ""
            if r.lower() in ("yes", "no"):
                out[t] = r.lower()
    return out


def score(picks: pd.DataFrame, results: dict[str, str]) -> dict:
    s = picks[picks["ticker"].isin(results)].copy()
    s["y"] = [1.0 if results[t] == "yes" else 0.0 for t in s["ticker"]]
    n = len(s)
    desk = float(((s["p"] - s["y"]) ** 2).mean()) if n else None
    mkt = float(((s["market"] - s["y"]) ** 2).mean()) if n else None
    return {"picks": len(picks), "settled": n, "brier_desk": desk, "brier_market": mkt,
            "ready_for_live": bool(n >= MIN_SETTLED and desk < mkt)}
