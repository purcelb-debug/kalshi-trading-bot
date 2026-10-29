"""Independent weather reference data (not the Kalshi settlement source).

* NCEI daily summaries: historical observed TMAX/TMIN, no key required.
* NWS forecasts (api.weather.gov): point-in-time forecast snapshots, stored with the
  fetch time as vintage. Historical NWS forecasts are not available from this API, so
  forecast history only accumulates from the first snapshot onward.

Note: Kalshi weather rules name the settlement source in each market (as of 2026-10-29
the NYC high-temperature series cites The Weather Company). NCEI values are an
independent cross-check and can differ by rounding or revision.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import Engine

from predlab.config import Settings
from predlab.data.collectors.base import persist_issues, tracked_run
from predlab.data.http import ReadOnlyClient
from predlab.data.integrity import check_observation, split_valid
from predlab.db.repo import upsert
from predlab.db.schema import ext_observations
from predlab.timeutil import UTC, utcnow

# Assumption (unverified): NCEI daily summaries usually appear within ~1-2 days. We use a
# conservative 2-day publication lag as the vintage so backtests never see them early.
NCEI_PUBLICATION_LAG = timedelta(days=2)
OBS_KEYS = ["source", "series_id", "obs_ts", "vintage_ts"]


@dataclass(frozen=True)
class Station:
    key: str
    ncei_id: str
    lat: float
    lon: float


# Station choices are assumptions to check against each market's rules text.
DEFAULT_STATIONS = (
    Station("NYC", "USW00094728", 40.7789, -73.9692),   # Central Park
    Station("CHI", "USW00014819", 41.7868, -87.7522),   # Chicago Midway
    Station("MIA", "USW00012839", 25.7906, -80.3164),   # Miami Intl
)


def _day(d: str | date) -> datetime:
    d = date.fromisoformat(d) if isinstance(d, str) else d
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def norm_ncei(rec: dict, station: Station, fetched_at: datetime) -> list[dict]:
    obs = _day(rec["DATE"])
    rows = []
    for field in ("TMAX", "TMIN"):
        raw = rec.get(field)
        rows.append({
            "source": "ncei", "series_id": f"{field.lower()}:{station.key}", "obs_ts": obs,
            "vintage_ts": obs + NCEI_PUBLICATION_LAG,
            "value": float(raw) if raw not in (None, "") else None, "units": "degF",
            "fetched_at": fetched_at,
        })
    return rows


def norm_nws_forecast(periods: list[dict], station: Station, fetched_at: datetime) -> list[dict]:
    """Daytime-period temperatures as forecast highs, keyed by the period's local date."""
    rows = []
    for p in periods:
        if not p.get("isDaytime"):
            continue
        local_date = datetime.fromisoformat(p["startTime"]).date()
        rows.append({
            "source": "nws", "series_id": f"fcst_high:{station.key}", "obs_ts": _day(local_date),
            "vintage_ts": fetched_at, "value": float(p["temperature"]),
            "units": "degF" if p.get("temperatureUnit", "F") == "F" else "degC",
            "fetched_at": fetched_at,
        })
    return rows


class WeatherCollector:
    def __init__(self, engine: Engine, settings: Settings,
                 ncei: ReadOnlyClient | None = None, nws: ReadOnlyClient | None = None):
        self.engine = engine
        self.ncei = ncei or ReadOnlyClient(settings.ncei_base_url, reads_per_sec=2,
                                           user_agent=settings.user_agent)
        self.nws = nws or ReadOnlyClient("https://api.weather.gov", reads_per_sec=2,
                                         user_agent=settings.user_agent)

    async def aclose(self):
        await self.ncei.aclose()
        await self.nws.aclose()

    async def collect_ncei(self, start: date, end: date, stations=DEFAULT_STATIONS):
        async with tracked_run(self.engine, "ncei") as stats:
            now = utcnow()
            for st in stations:
                data = await self.ncei.get("", {
                    "dataset": "daily-summaries", "stations": st.ncei_id,
                    "dataTypes": "TMAX,TMIN", "startDate": start.isoformat(),
                    "endDate": end.isoformat(), "format": "json", "units": "standard",
                })
                rows = [r for rec in data for r in norm_ncei(rec, st, now)]
                good, issues = split_valid(rows, check_observation)
                persist_issues(self.engine, stats, issues)
                stats.add("ext_observations", upsert(self.engine, ext_observations, good, OBS_KEYS))
        return stats

    async def snapshot_nws_forecasts(self, stations=DEFAULT_STATIONS):
        async with tracked_run(self.engine, "nws") as stats:
            for st in stations:
                point = await self.nws.get(f"/points/{st.lat:.4f},{st.lon:.4f}")
                path = point["properties"]["forecast"].split("api.weather.gov", 1)[1]
                fc = await self.nws.get(path)
                rows = norm_nws_forecast(fc["properties"]["periods"], st, utcnow())
                stats.add("ext_observations", upsert(self.engine, ext_observations, rows, OBS_KEYS))
        return stats
