# Phase 1: Data ingestion and storage (2026-10-29)

**Status:** complete, awaiting review. **Real-money execution is impossible** (verified below).

## What was built
| Piece | File |
|---|---|
| Settings (.env, SQLite default, Postgres URL optional) | `predlab/config.py` |
| Two disjoint schemas: historical vs simulation, separate DB files | `predlab/db/schema.py` |
| Idempotent upserts (executemany), health + data-quality bookkeeping | `predlab/db/repo.py` |
| Async GET-only client: rate limit, retry w/ exp. backoff + jitter, Retry-After, cursor pagination, path allowlist | `predlab/data/http.py` |
| Kalshi collector: series, events, markets (live + archive tier), outcomes w/ revision history, trades, candles, order-book snapshots | `predlab/data/collectors/kalshi.py` |
| Weather: NCEI observed daily TMAX/TMIN, NWS forecast snapshots | `predlab/data/collectors/weather.py` |
| Economics: FRED/ALFRED with full vintage history | `predlab/data/collectors/econ_fred.py` |
| Sports: CSV loader for user-supplied results | `predlab/data/collectors/sports_csv.py` |
| Synthetic generator, all 5 categories, saved ground truth | `predlab/data/collectors/synthetic.py` |
| Integrity checks (hard errors excluded, all issues recorded) | `predlab/data/integrity.py` |
| CLI: `init-db`, `ingest-kalshi`, `ingest-weather`, `ingest-fred`, `load-sports`, `synth`, `health` | `predlab/cli.py` |

## Test results
- `pytest`: **84 passed** (about 16 s). `pytest -m live`: **1 passed** (real Kalshi public GETs, including the archive tier).
- `ruff check predlab tests`: clean.
- End-to-end live run (temp DB): KXHIGHNY gave **9,826 markets** (2021-08 to 2026-10, 1,872 yes / 7,954 no). KXCPIYOY gave **671 markets** (2022-12 to 2026-10). An earlier run of KXHIGHNY with trades stored **533,381 public trades**. NCEI gave 156 observations and NWS 21 forecast rows. **0 data-quality issues.** Re-ingesting created no duplicate outcomes and no false revisions.

## No-real-money verification
- `tests/test_no_live_execution.py` scans every source file. It checks for trading endpoints, exchange SDK imports, private-key loading, live/real-money switches, and write verbs on any HTTP client. It also confirms that only `http.py` builds a client, that it only calls `.get`, and that `.env.example` has no exchange credential fields. **Passes.**
- `ReadOnlyClient.post/put/patch/delete/request/stream/send` raise `ReadOnlyViolation`. Paths that match portfolio, orders, positions, fills or balance are refused on any host, which includes `/historical/fills|orders|positions`. Kalshi paths must also match a read-only allowlist. All of this is tested.
- The manual grep for order or portfolio endpoints only matches the blocking code itself and one docstring.

## Found during the build (confirmed)
- **Kalshi archive tier.** Settled markets, their candles, and trades older than `GET /historical/cutoff` (currently `2026-08-30`) disappear from the live endpoints. Without the archive, KXHIGHNY returned only 402 markets. The archive endpoints are public. Source: docs.kalshi.com/getting_started/historical_data, read 2026-10-29.
- Kalshi prices now come as `*_dollars` strings and sizes as `*_fp` strings. Both sides of the book are bids only, so YES ask = 1 − best NO bid. The normalizers also accept the older integer-cent fields.
- Kalshi rejects `start_ts=0` on candlesticks, so the collector now checks for it.
- The NYC high-temperature rules name **The Weather Company** as the settlement source. NCEI is used only as an independent cross-check.

## Unsupported assumptions (flagged, not verified)
1. The NCEI publication lag is set conservatively to 2 days for vintage timestamps. The actual lag is unknown.
2. The NCEI station chosen for each city (Central Park, Midway, Miami Intl) may not match a market's settlement station.
3. NWS historical forecasts aren't available from api.weather.gov. Forecast history only builds up from our own snapshots.
4. FRED `realtime_start` is used as the date each value became public. It has day resolution and no release time of day, so features from the release day need a further lag (handled in Phase 2).
5. Naive timestamps are treated as UTC.
6. A tie in sports CSV data resolves as "no" (the home team didn't win).
7. All synthetic category differences are invented. They say nothing about real categories.

## Unresolved issues
- **Kalshi fee schedule isn't encoded yet** (Phase 3). The series payload has `fee_type: "quadratic"` and `fee_multiplier`. The formula still has to be checked against Kalshi's published schedule.
- Order-book depth history exists only from our own snapshots. Older backtests will price fills as `liquidity_uncertain`.
- Pulling trades for a whole series is heavy (about 1,000 requests and 533k rows for KXHIGHNY's live tier alone). The archive tier will be larger. A `--since` window or a per-market cap is needed before bulk trade backfills.
- No FRED key is configured, so the economics collector is skipped and marked that way in health.
- Sports and public-event categories have no official data feed. They rely on CSV, synthetic data, and Kalshi market data.
- PostgreSQL support is written (dialect-aware upsert) but **untested**, because no Postgres instance is available.
