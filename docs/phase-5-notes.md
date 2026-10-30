# Phase 5 notes: autonomous paper-trading research loop

PAPER TRADING — NOT REAL MONEY. Written 2026-10-30.

## What was built
- `predlab/loop/promotion.py`
  - **Deployment** = (model version, series, decision horizon), e.g.
    `logistic_l2-57cdc200f7@KXHIGHNY-h30`. Forecasts, the paper record and the executor
    gate all key on this, so evidence from one series cannot unlock another (the model
    version hash alone covers only the model name and params).
  - `register_candidate`: admits a deployment to forward paper trading using development
    data only. kappa (shrink toward the market) is fit on dev walk-forward predictions.
    The holdout is not touched and the gate stays closed.
  - `certify`: the single look at the untouched holdout. It computes the holdout Brier
    difference vs the market, plus a holdout paper backtest with kappa fixed from dev,
    then promotes or rejects. A deployment can be certified once, ever. Holdout hashes
    change as markets settle, so the "once" is per deployment and not per hash. The k-th
    certification for a series and horizon must clear CIs at level 1 − 0.05/k
    (Bonferroni). Promotion also has to beat the incumbent's holdout Brier, and the
    incumbent is then retired. Market-price coverage on the holdout must be ≥ 95%.
- `predlab/loop/runner.py`, one cycle per run:
  - collect, then refit on outcomes known by `now`.
  - Forecast each open market once, when its time to close first falls in
    [horizon − 2h, horizon], which is the trained and certified decision point. Each
    forecast goes into the hash-chained log with the market mid.
  - Rebuild the paper record.
  - Check degradation, then write the gate evidence and a report
    (`reports/loop-<deployment>-latest.json`).
- **Paper record = deterministic replay.** Each cycle re-runs the Phase 3 backtest engine
  over the immutable forward-forecast log, with quotes collected at decision time. The
  paper strategy is therefore exactly the certified one (kappa, entry band, Kelly
  sizing, caps, drawdown halt), and re-running a cycle reproduces it. It is stored as
  sim run `paper-<deployment>`.
- **Degradation alerts** are logged once per rule per UTC day and count against the gate:
  - `worse_than_market`: rolling paired Brier difference (last 100 resolved, minimum 30)
    with CI low > 0.
  - `paper_drawdown`: paper max drawdown above the 20% halt.
  - `edge_disappeared` (estimate ≥ 0 but not significant) is a warning only.
- **`paper_days`** = distinct UTC days with a completed cycle. Days with no cycle, or a
  failed collection, don't count.
- `predlab/loop/collect.py`: per-cycle read-only collection.
  - Kalshi: open markets, settlements from the last 3 days, and candles plus an order
    book for markets near their decision point.
  - NCEI/NWS for weather series and FRED for CPI series. Failures in those leave the
    features missing (flagged) and don't stop the cycle.
- `predlab/backtest/decision.py`: the order-decision rule, moved out of the executor.
  The executor imports it and also caps its shrink at the certified kappa.
- `predlab/features/pointintime.py`:
  - `as_of` for training (outcomes recorded by then).
  - `build_live_dataset` (open markets, features as of now, aligned to the model's columns).
  - `load_open_markets`.
- CLI: `register`, `certify`, `loop [--once] [--interval-minutes] [--cycles]`, `paper-status`.
- Tests: `tests/test_phase5.py` (10 tests) and one executor test. Full suite: 201 passed.

## Live state (2026-10-30)
- Registered **`logistic_l2-57cdc200f7@KXHIGHNY-h30`** as a candidate, with kappa = 0.8682
  from dev walk-forward. Holdout untouched.
- Live `loop --once` succeeded (Kalshi + NCEI + NWS collected, gate evidence written).
  The hourly loop was started in the Claude Code session. The next forecasts are due when
  the 2026-11-01 markets enter the window at 23:00 UTC on 2026-10-30.
- The loop only runs while that process (and the PC) is running. A persistent schedule
  (Windows Task Scheduler) has **not** been set up; that needs the owner's approval.

## Category comparison, markets closing ≥ 2026-07-01, 30h horizon
| rank | category | n | best | vs market Brier | notes |
|---|---|---|---|---|---|
| 1 | financial (KXINX) | 2,550 | market_platt | +0.0004 | Ranked first only on raw Brier (far-from-money strikes are easy). Market coverage 40%. No independent info. |
| 2 | weather (KXHIGHNY) | 726 | market_platt | −0.0014 | Independent info present. |
| 3 | public_events (KXAPRPOTUS) | 144 | lightgbm | −0.0148 | Small n, won 1 of 3 folds: treat as noise until more data. |
| 4 | sports (KXNFLGAME) | 306 | market_platt | +0.0113 | Worse than market. |
| – | economics (KXCPIYOY) | 89 | – | – | Skipped (< 100 markets since July). |

## Issues found and fixed
- `decision_clamped_to_open` was attached to the wrong rows. It was computed before the
  dataset was re-sorted. This existed before Phase 5 and is fixed by carrying it as a column.
- In synthetic replay, κ = 0.15 shrinks every edge below the 3¢ minimum, so the replay made
  no trades at all. A dedicated test forces trading to exercise fills and settlement.

## Unsupported assumptions / open issues
- Live features use the last completed hourly candle, so the market mid at forecast time
  can be up to an hour old. Training made the same assumption, but it's an assumption.
- Paper fills walk the order-book snapshot taken during collection, typically seconds
  before the decision. Real fills could differ if the book moved.
- `register` and `certify` refit on all data each cycle, with the same params, so the
  model version is unchanged. The model's coefficients do drift with new data, which is
  intended.
- The weather NWS snapshot is taken each cycle. Older cycles could not have seen later
  snapshots, which is enforced by `vintage_ts`.

## Real-money check
- `tests/test_no_live_execution.py` passes. `predlab/` still has no authenticated client,
  no order paths and no Kalshi credentials, and it does not import `executor`. The loop
  writes only to the local `sim_*`, `forecasts`, `model_registry` and `research_log` tables.
- The executor (separate, uncommitted) will place live orders only when
  `model_registry.status == "promoted"`. That requires `certify` to pass, plus ≥ 28 paper
  days with 0 alerts and the owner's money limits in `executor/.env`.
