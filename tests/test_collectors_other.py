from datetime import date

import httpx
import numpy as np
import pytest
import respx
from sqlalchemy import func, select

from predlab.categories import CATEGORIES, map_kalshi_category
from predlab.config import Settings
from predlab.data.collectors.econ_fred import FredCollector, norm_fred
from predlab.data.collectors.sports_csv import load_sports_csv
from predlab.data.collectors.synthetic import generate
from predlab.data.collectors.weather import DEFAULT_STATIONS, WeatherCollector, norm_nws_forecast
from predlab.db.schema import (
    ext_observations,
    mkt_markets,
    mkt_orderbook_snapshots,
    outcomes,
    source_health,
    synthetic_truth,
)
from predlab.timeutil import to_utc, utcnow


@pytest.mark.parametrize("label,cat", [
    ("Climate and Weather", "weather"), ("Economics", "economics"), ("Financials", "financial"),
    ("Crypto", "financial"), ("Sports", "sports"), ("Politics", "public_events"),
    (None, "public_events"),
])
def test_category_map(label, cat):
    assert map_kalshi_category(label) == cat


@respx.mock
async def test_ncei_collector(market_engine):
    base = "https://ncei.test/data/v1"
    respx.get(base).mock(return_value=httpx.Response(200, json=[
        {"DATE": "2026-10-01", "STATION": "USW00094728", "TMAX": "76", "TMIN": "65"},
        {"DATE": "2026-10-02", "STATION": "USW00094728", "TMAX": "", "TMIN": "68"}]))
    col = WeatherCollector(market_engine, Settings(ncei_base_url=base))
    try:
        await col.collect_ncei(date(2026, 10, 1), date(2026, 10, 2), stations=DEFAULT_STATIONS[:1])
    finally:
        await col.aclose()
    with market_engine.connect() as c:
        rows = {(r.series_id, r.obs_ts.date().isoformat()): r for r in c.execute(select(ext_observations))}
    r = rows[("tmax:NYC", "2026-10-01")]
    assert r.value == 76.0 and (r.vintage_ts - r.obs_ts).days == 2  # conservative publication lag
    assert rows[("tmax:NYC", "2026-10-02")].value is None  # missing stays missing


def test_nws_forecast_uses_fetch_time_as_vintage():
    now = utcnow()
    rows = norm_nws_forecast([
        {"isDaytime": True, "startTime": "2026-10-30T06:00:00-04:00", "temperature": 64,
         "temperatureUnit": "F"},
        {"isDaytime": False, "startTime": "2026-10-30T18:00:00-04:00", "temperature": 50}],
        DEFAULT_STATIONS[0], now)
    assert len(rows) == 1 and rows[0]["vintage_ts"] == now
    assert rows[0]["obs_ts"] == to_utc("2026-10-30") and rows[0]["value"] == 64.0


def test_fred_vintages_and_missing_values():
    rows = norm_fred([
        {"date": "2026-01-01", "realtime_start": "2026-02-06", "value": "4.0"},
        {"date": "2026-01-01", "realtime_start": "2026-03-06", "value": "4.1"},
        {"date": "2026-02-01", "realtime_start": "2026-03-06", "value": "."}], "UNRATE", utcnow())
    assert [r["vintage_ts"].date().isoformat() for r in rows] == ["2026-02-06", "2026-03-06",
                                                                    "2026-03-06"]
    assert rows[2]["value"] is None


async def test_fred_skips_without_key(market_engine):
    col = FredCollector(market_engine, Settings(fred_api_key=""))
    try:
        assert await col.collect() is None
    finally:
        await col.aclose()
    with market_engine.connect() as c:
        h = c.execute(select(source_health)).one()
    assert h.source == "fred" and "skipped" in h.last_error


async def test_sports_csv(market_engine, tmp_path):
    p = tmp_path / "games.csv"
    p.write_text("game_id,start_time,home,away,home_score,away_score,league,home_win_prob_ref\n"
                 "1,2025-09-07T17:00:00Z,A,B,24,17,NFL,0.62\n"
                 "2,2025-09-07T20:00:00-04:00,C,D,10,10,NFL,\n", encoding="utf-8")
    await load_sports_csv(market_engine, p)
    with market_engine.connect() as c:
        res = dict(c.execute(select(mkt_markets.c.ticker, mkt_markets.c.result)).all())
        n_ref = c.execute(select(func.count()).select_from(ext_observations)).scalar()
        close2 = c.execute(select(mkt_markets.c.close_time)
                           .where(mkt_markets.c.ticker == "SPORTS-2")).scalar()
    assert res == {"SPORTS-1": "yes", "SPORTS-2": "no"}  # tie -> home did not win
    assert n_ref == 1 and close2 == to_utc("2025-09-08T00:00:00Z")


async def test_sports_csv_rejects_missing_columns(market_engine, tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("game_id,home\n1,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        await load_sports_csv(market_engine, p)


# ---------------------------------------------------------------- synthetic


@pytest.fixture(scope="module")
def synth_db(tmp_path_factory):
    from predlab.db.repo import init_market_db, make_engine
    eng = make_engine(f"sqlite:///{tmp_path_factory.mktemp('syn') / 'm.db'}")
    init_market_db(eng)
    res = generate(eng, seed=11, days=200)
    yield eng, res
    eng.dispose()


def test_synthetic_is_deterministic(synth_db, tmp_path):
    from predlab.db.repo import init_market_db, make_engine
    _, res = synth_db
    eng2 = make_engine(f"sqlite:///{tmp_path / 'm2.db'}")
    init_market_db(eng2)
    res2 = generate(eng2, seed=11, days=200)
    assert res.rows == res2.rows
    with synth_db[0].connect() as a, eng2.connect() as b:
        q = select(synthetic_truth.c.ticker, synthetic_truth.c.true_p).order_by(synthetic_truth.c.ticker)
        assert a.execute(q).all() == b.execute(q).all()
    eng2.dispose()


def test_synthetic_covers_all_categories_with_different_sample_sizes(synth_db):
    eng, _ = synth_db
    with eng.connect() as c:
        counts = dict(c.execute(select(mkt_markets.c.category, func.count())
                                .group_by(mkt_markets.c.category)).all())
    assert set(counts) == set(CATEGORIES)
    assert counts["sports"] > 5 * counts["economics"]


def test_synthetic_outcomes_match_true_probabilities(synth_db):
    eng, _ = synth_db
    with eng.connect() as c:
        rows = c.execute(select(synthetic_truth.c.true_p, outcomes.c.result)
                         .join(outcomes, outcomes.c.ticker == synthetic_truth.c.ticker)).all()
    p = np.array([r.true_p for r in rows])
    y = np.array([r.result == "yes" for r in rows], dtype=float)
    # Calibration of the generator itself: per-decile outcome rate vs mean true p.
    for lo in np.arange(0, 1, 0.2):
        m = (p >= lo) & (p < lo + 0.2)
        if m.sum() > 100:
            assert abs(y[m].mean() - p[m].mean()) < 0.06
    assert abs(y.mean() - p.mean()) < 0.03


def test_synthetic_features_never_postdate_market_open(synth_db):
    eng, _ = synth_db
    with eng.connect() as c:
        bad = c.execute(
            select(func.count()).select_from(ext_observations.join(
                mkt_markets, ext_observations.c.series_id.endswith(mkt_markets.c.ticker)))
            .where(ext_observations.c.vintage_ts > mkt_markets.c.open_time)).scalar()
    assert bad == 0


def test_synthetic_books_are_not_crossed(synth_db):
    eng, _ = synth_db
    with eng.connect() as c:
        snaps = c.execute(select(mkt_orderbook_snapshots)).all()
    best: dict = {}
    for r in snaps:
        k = (r.ticker, r.snapshot_ts)
        best.setdefault(k, {"yes": 0, "no": 0})
        best[k][r.side] = max(best[k][r.side], r.price)
    assert snaps and all(b["yes"] < 1 - b["no"] for b in best.values())
