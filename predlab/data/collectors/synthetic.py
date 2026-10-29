"""Seeded synthetic market generator for all five categories.

The generator writes the same tables real collectors write, plus ``synthetic_truth`` with
each market's true probability, so models and the simulator can be checked against a
known answer. Categories differ on purpose (sample size, signal strength, how efficient the
market price is, spreads, depth, order-book coverage, correlation, regime change). These
are synthetic constructs and say nothing about real categories.

Data-generating process per market i on day d:
    logit(p_i) = b0 + b . x_i + rho * z_d + u_i         (true probability)
    logit(m_i) = b0 + eff_x * (b . x_i) + rho * z_d + eff_u * u_i + noise   (market mid at close)
x (3 features, x3 pure noise) is published at open, z_d is a shared daily shock (correlated
outcomes), u_i is information only the market partly sees. Outcome ~ Bernoulli(p_i).
With ``drift_at`` set, the x1 coefficient flips sign at that fraction of the timeline, so
an edge learned early disappears later.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import Engine

from predlab.categories import ECONOMICS, FINANCIAL, PUBLIC_EVENTS, SPORTS, WEATHER
from predlab.data.collectors.base import tracked_run
from predlab.db.repo import upsert
from predlab.db.schema import (
    ext_observations,
    mkt_candles,
    mkt_events,
    mkt_markets,
    mkt_orderbook_snapshots,
    mkt_series,
    mkt_trades,
    outcomes,
    synthetic_truth,
)
from predlab.timeutil import UTC, utcnow

SOURCE = "synthetic"
FEATURES = ("x1", "x2", "x3")
OPEN_HOURS = 20          # market life: 00:00 -> 20:00 UTC on its day
BOOK_EVERY_HOURS = 4
BOOK_LEVELS = 3


@dataclass(frozen=True)
class CategorySpec:
    rate_per_day: float          # Poisson mean markets per day (sample size)
    b0: float = 0.0
    b: tuple = (1.0, 0.6, 0.0)   # x3 coefficient is 0: a pure-noise feature
    hidden_sd: float = 0.8       # sd of u (info outside the model's features)
    eff_x: float = 0.7           # share of the feature signal priced in
    eff_u: float = 0.5           # share of hidden info priced in
    price_noise_sd: float = 0.3
    rho: float = 0.0             # weight on shared daily shock
    half_spread: float = 0.02
    depth_mean: float = 150.0    # contracts per book level
    book_coverage: float = 0.5   # share of markets with order-book snapshots
    trades_per_hour: float = 2.0
    drift_at: float | None = None


DEFAULT_SPECS: dict[str, CategorySpec] = {
    WEATHER: CategorySpec(3.0, b=(1.2, 0.8, 0.0), hidden_sd=0.6, eff_x=0.8, half_spread=0.02,
                          depth_mean=200, book_coverage=0.6, rho=0.6),
    ECONOMICS: CategorySpec(0.3, b=(0.9, 0.5, 0.0), hidden_sd=0.7, eff_x=0.85, half_spread=0.03,
                            depth_mean=400, book_coverage=0.8),
    FINANCIAL: CategorySpec(2.0, b=(0.8, 0.4, 0.0), hidden_sd=1.0, eff_x=0.6, half_spread=0.02,
                            depth_mean=300, book_coverage=0.7, drift_at=0.6),
    SPORTS: CategorySpec(4.0, b=(1.0, 0.7, 0.0), hidden_sd=0.9, eff_x=0.9, eff_u=0.7,
                         half_spread=0.015, depth_mean=500, book_coverage=0.5),
    PUBLIC_EVENTS: CategorySpec(0.5, b=(1.1, 0.3, 0.0), hidden_sd=1.2, eff_x=0.5,
                                half_spread=0.05, depth_mean=60, book_coverage=0.2,
                                trades_per_hour=0.5),
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return np.log(p / (1.0 - p))


def _cents(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return float(min(hi, max(lo, round(p * 100) / 100)))


@dataclass
class SyntheticResult:
    markets: int = 0
    rows: dict = field(default_factory=dict)


def generate(engine: Engine, *, seed: int = 7, start: datetime = datetime(2024, 1, 1, tzinfo=UTC),
             days: int = 365, specs: dict[str, CategorySpec] | None = None) -> SyntheticResult:
    """Generate and persist a deterministic synthetic dataset. Returns row counts."""
    specs = specs or DEFAULT_SPECS
    rng = np.random.default_rng(seed)
    now = utcnow()
    tables: dict[str, list[dict]] = {k: [] for k in (
        "series", "events", "markets", "candles", "trades", "books", "obs", "outcomes", "truth")}

    for cat, spec in specs.items():
        series_ticker = f"SYN-{cat.upper()}"
        tables["series"].append({
            "series_ticker": series_ticker, "source": SOURCE, "title": f"Synthetic {cat}",
            "source_category": cat, "category": cat, "frequency": "daily",
            "fee_type": "quadratic", "fee_multiplier": 1.0, "raw_json": {"spec": spec.__dict__},
            "fetched_at": now,
        })
        for d in range(days):
            n = rng.poisson(spec.rate_per_day)
            if n == 0:
                continue
            day = start + timedelta(days=d)
            event_ticker = f"{series_ticker}-{day:%y%m%d}"
            tables["events"].append({
                "event_ticker": event_ticker, "source": SOURCE, "series_ticker": series_ticker,
                "title": f"Synthetic {cat} {day:%Y-%m-%d}", "sub_title": None,
                "source_category": cat, "category": cat, "strike_date": day,
                "mutually_exclusive": False, "raw_json": None, "fetched_at": now,
            })
            z = rng.normal()
            b = np.array(spec.b, dtype=float)
            if spec.drift_at is not None and d >= spec.drift_at * days:
                b[0] = -b[0]
            for j in range(n):
                _one_market(rng, tables, spec, cat, series_ticker, event_ticker, day, j, b, z,
                            seed, now)

    stats_rows = {}
    with_keys = [
        ("series", mkt_series, ["series_ticker"]),
        ("events", mkt_events, ["event_ticker"]),
        ("markets", mkt_markets, ["ticker"]),
        ("candles", mkt_candles, ["ticker", "period_minutes", "end_ts"]),
        ("trades", mkt_trades, ["trade_id"]),
        ("books", mkt_orderbook_snapshots, ["ticker", "snapshot_ts", "side", "price"]),
        ("obs", ext_observations, ["source", "series_id", "obs_ts", "vintage_ts"]),
        ("outcomes", outcomes, ["ticker", "recorded_at"]),
        ("truth", synthetic_truth, ["ticker"]),
    ]
    for name, table, keys in with_keys:
        stats_rows[table.name] = upsert(engine, table, tables[name], keys)
    return SyntheticResult(markets=len(tables["markets"]), rows=stats_rows)


def _one_market(rng, tables, spec: CategorySpec, cat, series_ticker, event_ticker, day, j, b, z,
                seed, now):
    ticker = f"{event_ticker}-M{j}"
    open_t = day
    close_t = day + timedelta(hours=OPEN_HOURS)
    x = rng.normal(size=len(FEATURES))
    u = rng.normal(scale=spec.hidden_sd)
    signal = float(b @ x)
    true_p = float(_sigmoid(spec.b0 + signal + spec.rho * z + u))
    final_mid = float(np.clip(_sigmoid(
        spec.b0 + spec.eff_x * signal + spec.rho * z + spec.eff_u * u
        + rng.normal(scale=spec.price_noise_sd)), 0.02, 0.98))
    result = "yes" if rng.random() < true_p else "no"

    # Mid-price path: random walk in logit space bridged from an uninformed open to final_mid.
    start_logit = float(_logit(np.clip(_sigmoid(spec.b0) + rng.normal(scale=0.05), 0.05, 0.95)))
    end_logit = float(_logit(final_mid))
    walk = np.cumsum(rng.normal(scale=0.15, size=OPEN_HOURS))
    walk -= np.linspace(0, walk[-1], OPEN_HOURS)  # pin the bridge's endpoint
    w = np.linspace(1 / OPEN_HOURS, 1.0, OPEN_HOURS)
    mids = _sigmoid(start_logit * (1 - w) + end_logit * w + walk)

    has_book = rng.random() < spec.book_coverage
    total_vol = 0.0
    last_trade = None
    for h in range(OPEN_HOURS):
        mid = float(mids[h])
        bid, ask = _cents(mid - spec.half_spread), _cents(mid + spec.half_spread)
        if ask <= bid:
            ask = _cents(bid + 0.01)
        hour_start = open_t + timedelta(hours=h)
        end_ts = hour_start + timedelta(hours=1)
        prices = []
        vol_before = total_vol
        for k in range(rng.poisson(spec.trades_per_hour)):
            taker_yes = rng.random() < 0.5
            px = ask if taker_yes else bid
            cnt = float(max(1, int(rng.exponential(20))))
            ts = hour_start + timedelta(seconds=float(rng.uniform(0, 3600)))
            tables["trades"].append({
                "trade_id": f"{ticker}-T{h}-{k}", "ticker": ticker, "created_time": ts,
                "yes_price": px, "no_price": round(1 - px, 2), "count": cnt,
                "taker_side": "yes" if taker_yes else "no", "fetched_at": now,
            })
            prices.append(px)
            total_vol += cnt
            last_trade = px
        tables["candles"].append({
            "ticker": ticker, "period_minutes": 60, "end_ts": end_ts,
            "price_open": prices[0] if prices else None,
            "price_high": max(prices) if prices else None,
            "price_low": min(prices) if prices else None,
            "price_close": prices[-1] if prices else None,
            "price_mean": float(np.mean(prices)) if prices else None,
            "yes_bid_open": bid, "yes_bid_close": bid, "yes_ask_open": ask, "yes_ask_close": ask,
            "volume": total_vol - vol_before, "open_interest": total_vol,
            "fetched_at": now,
        })
        if has_book and h % BOOK_EVERY_HOURS == BOOK_EVERY_HOURS - 1:
            for lvl in range(BOOK_LEVELS):
                yes_px = _cents(bid - 0.01 * lvl)
                no_px = _cents(1 - ask - 0.01 * lvl)
                for side, px in (("yes", yes_px), ("no", no_px)):
                    tables["books"].append({
                        "ticker": ticker, "snapshot_ts": end_ts, "side": side, "price": px,
                        "size": float(max(1, int(rng.exponential(spec.depth_mean)))),
                    })

    last_bid, last_ask = _cents(float(mids[-1]) - spec.half_spread), _cents(float(mids[-1]) + spec.half_spread)
    tables["markets"].append({
        "ticker": ticker, "source": SOURCE, "event_ticker": event_ticker,
        "series_ticker": series_ticker, "category": cat, "title": f"Synthetic {cat} market",
        "market_type": "binary", "status": "settled", "open_time": open_t, "close_time": close_t,
        "expiration_time": close_t + timedelta(hours=2), "result": result,
        "expiration_value": None, "strike_type": None, "floor_strike": None, "cap_strike": None,
        "yes_bid": last_bid, "yes_ask": last_ask, "no_bid": _cents(1 - last_ask),
        "no_ask": _cents(1 - last_bid), "last_price": last_trade, "volume": total_vol,
        "open_interest": total_vol, "updated_time": close_t, "raw_json": None, "fetched_at": now,
    })
    for name, val in zip(FEATURES, x):
        tables["obs"].append({
            "source": SOURCE, "series_id": f"{name}:{ticker}", "obs_ts": open_t,
            "vintage_ts": open_t, "value": float(val), "units": None, "fetched_at": now,
        })
    tables["outcomes"].append({
        "ticker": ticker, "recorded_at": close_t + timedelta(hours=2), "result": result,
        "resolved_ts": close_t + timedelta(hours=2), "source": SOURCE, "revised": False,
    })
    tables["truth"].append({"ticker": ticker, "true_p": true_p, "seed": seed})


async def generate_tracked(engine: Engine, **kw) -> SyntheticResult:
    async with tracked_run(engine, SOURCE) as stats:
        res = generate(engine, **kw)
        for t, n in res.rows.items():
            stats.add(t, n)
    return res
