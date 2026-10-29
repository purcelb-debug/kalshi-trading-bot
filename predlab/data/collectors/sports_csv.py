"""Loader for user-supplied historical sports results (CSV).

There is no free official sports-results API that fits cleanly, so sports history comes
from a CSV you provide. Each game becomes a binary market "home team wins".

Required columns: game_id, start_time (ISO, UTC if no offset), home, away,
home_score, away_score. Optional: league, home_win_prob_ref (an external reference
probability, e.g. a closing line converted to a probability).
"""

import csv
from pathlib import Path

from sqlalchemy import Engine

from predlab.categories import SPORTS
from predlab.data.collectors.base import persist_issues, tracked_run
from predlab.data.integrity import check_market, split_valid
from predlab.db.repo import upsert
from predlab.db.schema import ext_observations, mkt_markets, outcomes
from predlab.timeutil import to_utc, utcnow

REQUIRED = {"game_id", "start_time", "home", "away", "home_score", "away_score"}
SOURCE = "csv_sports"


def _market_row(r: dict, now) -> dict:
    start = to_utc(r["start_time"])
    hs, aws = float(r["home_score"]), float(r["away_score"])
    league = r.get("league") or "sports"
    return {
        "ticker": f"SPORTS-{r['game_id']}", "source": SOURCE, "event_ticker": league,
        "series_ticker": league, "category": SPORTS,
        "title": f"{r['home']} beats {r['away']}", "market_type": "binary",
        "status": "settled", "open_time": None, "close_time": start,
        "expiration_time": start,
        "result": "yes" if hs > aws else "no",  # a tie resolves "no": home did not win
        "expiration_value": f"{hs:g}-{aws:g}", "strike_type": None, "floor_strike": None,
        "cap_strike": None, "yes_bid": None, "yes_ask": None, "no_bid": None, "no_ask": None,
        "last_price": None, "volume": None, "open_interest": None, "updated_time": now,
        "raw_json": r, "fetched_at": now,
    }


async def load_sports_csv(engine: Engine, path: str | Path):
    now = utcnow()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"sports CSV missing columns: {sorted(missing)}")
        rows = list(reader)
    async with tracked_run(engine, SOURCE) as stats:
        good, issues = split_valid([_market_row(r, now) for r in rows], check_market)
        persist_issues(engine, stats, issues)
        stats.add("mkt_markets", upsert(engine, mkt_markets, good, ["ticker"]))
        outs = [{"ticker": m["ticker"], "recorded_at": now, "result": m["result"],
                 "resolved_ts": m["close_time"], "source": SOURCE, "revised": False}
                for m in good]
        stats.add("outcomes", upsert(engine, outcomes, outs, ["ticker", "recorded_at"],
                                     update=False))
        refs = [{"source": SOURCE, "series_id": f"ref_p:SPORTS-{r['game_id']}",
                 "obs_ts": to_utc(r["start_time"]), "vintage_ts": to_utc(r["start_time"]),
                 "value": float(r["home_win_prob_ref"]), "units": "prob", "fetched_at": now}
                for r in rows if r.get("home_win_prob_ref")]
        stats.add("ext_observations", upsert(engine, ext_observations, refs,
                                             ["source", "series_id", "obs_ts", "vintage_ts"]))
    return stats
