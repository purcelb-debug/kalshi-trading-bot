"""Per-cycle data collection for the research loop (public, read-only endpoints only).

Kalshi: the series' open markets (metadata), hourly candles and a fresh order-book
snapshot for markets near their decision point, and markets settled in the last
``settled_lookback`` (outcomes). Weather and FRED inputs are refreshed for series that
use them; a failure there is logged and the affected features come through as missing
(flagged), it does not stop the cycle.
"""

from datetime import timedelta

from sqlalchemy import Engine

from predlab.config import Settings
from predlab.data.collectors.base import tracked_run
from predlab.data.collectors.kalshi import SOURCE, KalshiCollector, make_client
from predlab.features.pointintime import ECON_SERIES, WEATHER_SERIES_STATION
from predlab.logs import get_logger
from predlab.timeutil import utcnow

log = get_logger(__name__)


def make_collect(market: Engine, settings: Settings, *, interval: timedelta,
                 settled_lookback: timedelta = timedelta(days=3)):
    async def collect(params: dict, cfg) -> None:
        series, horizon = params["series"], timedelta(hours=params["horizon_h"])
        # Candles and books for anything that could enter the window before the next cycle.
        lo, hi = horizon - cfg.late_tolerance - interval, horizon + interval
        async with make_client(settings) as client:
            col = KalshiCollector(market, client)
            async with tracked_run(market, SOURCE) as stats:
                ser = await col.collect_series(series, stats)
                now = utcnow()
                open_ = await col.collect_markets(stats, series_ticker=series, status="open",
                                                  category=ser["category"])
                await col.collect_markets(
                    stats, series_ticker=series, status="settled", category=ser["category"],
                    min_close_ts=int((now - settled_lookback).timestamp()))
                for m in open_:
                    if not m["open_time"] or not m["close_time"]:
                        continue
                    if lo <= m["close_time"] - now <= hi:
                        await col.collect_candles(series, m["ticker"],
                                                  int(m["open_time"].timestamp()),
                                                  int(utcnow().timestamp()), stats)
                        await col.snapshot_orderbook(m["ticker"], stats)
        if series in WEATHER_SERIES_STATION:
            await _weather(market, settings)
        if series in ECON_SERIES:
            await _fred(market, settings, ECON_SERIES[series])
    return collect


async def _weather(market: Engine, settings: Settings) -> None:
    from predlab.data.collectors.weather import WeatherCollector
    col = WeatherCollector(market, settings)
    try:
        today = utcnow().date()
        await col.collect_ncei(today - timedelta(days=14), today)
        await col.snapshot_nws_forecasts()
    except Exception as e:  # noqa: BLE001 - weather features become missing, flagged
        log.warning("loop_weather_failed", error=str(e))
    finally:
        await col.aclose()


async def _fred(market: Engine, settings: Settings, series_id: str) -> None:
    from predlab.data.collectors.econ_fred import FredCollector
    col = FredCollector(market, settings)
    try:
        await col.collect((series_id,), observation_start=f"{utcnow().year - 3}-01-01")
    except Exception as e:  # noqa: BLE001
        log.warning("loop_fred_failed", error=str(e))
    finally:
        await col.aclose()
