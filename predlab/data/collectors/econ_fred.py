"""FRED/ALFRED economic indicators with full vintage history.

Requesting the full real-time period returns every revision of every observation along
with the date it became public (``realtime_start``). That date is stored as vintage_ts,
which is what keeps economic features free of look-ahead (no revised data seen early).
Skipped, with health recorded, if no PREDLAB_FRED_API_KEY is set.
"""

from sqlalchemy import Engine

from predlab.config import Settings
from predlab.data.collectors.base import persist_issues, tracked_run
from predlab.data.http import ReadOnlyClient
from predlab.data.integrity import check_observation, split_valid
from predlab.db.repo import record_health, upsert
from predlab.db.schema import ext_observations
from predlab.timeutil import to_utc, utcnow

DEFAULT_SERIES = ("CPIAUCSL", "UNRATE", "PAYEMS", "GDPC1", "FEDFUNDS")


def norm_fred(obs: list[dict], series_id: str, fetched_at) -> list[dict]:
    return [{
        "source": "fred", "series_id": series_id, "obs_ts": to_utc(o["date"]),
        "vintage_ts": to_utc(o["realtime_start"]),
        "value": None if o.get("value") in (None, ".", "") else float(o["value"]),
        "units": None, "fetched_at": fetched_at,
    } for o in obs]


class FredCollector:
    def __init__(self, engine: Engine, settings: Settings, client: ReadOnlyClient | None = None):
        self.engine = engine
        self.api_key = settings.fred_api_key
        self.client = client or ReadOnlyClient(
            settings.fred_base_url, reads_per_sec=2, user_agent=settings.user_agent,
            default_params={"api_key": self.api_key, "file_type": "json"} if self.api_key else {},
        )

    async def aclose(self):
        await self.client.aclose()

    async def collect(self, series_ids=DEFAULT_SERIES, observation_start: str = "2015-01-01"):
        if not self.api_key:
            record_health(self.engine, "fred", False, 0.0, 0,
                          "skipped: PREDLAB_FRED_API_KEY not set")
            return None
        async with tracked_run(self.engine, "fred") as stats:
            for sid in series_ids:
                data = await self.client.get("/series/observations", {
                    "series_id": sid, "observation_start": observation_start,
                    "realtime_start": "1776-07-04", "realtime_end": "9999-12-31",
                    "limit": 100000,
                })
                rows = norm_fred(data.get("observations", []), sid, utcnow())
                good, issues = split_valid(rows, check_observation)
                persist_issues(self.engine, stats, issues)
                stats.add("ext_observations", upsert(
                    self.engine, ext_observations, good,
                    ["source", "series_id", "obs_ts", "vintage_ts"]))
        return stats
