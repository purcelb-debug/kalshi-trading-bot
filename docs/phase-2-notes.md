# Phase 2: Baseline probability models (2026-10-29)

**Status:** complete, awaiting review. **Real-money execution is impossible** (the static guard still passes, and nothing in Phase 2 makes network calls).

## What was built
| Piece | File |
|---|---|
| Point-in-time dataset: decision time = close − horizon. Candles count only after their period ends, observations only after their vintage. `LookAheadError` if any input postdates its decision time | `predlab/features/pointintime.py` |
| Feature sets: market state (mid, spread, volume, staleness), per-market observations (synthetic x1–x3, sports reference prob), weather (strike margin vs trailing 7-day NCEI TMAX and vs NWS forecast) | same |
| Data-quality flags per row: no_market_price, stale_market_price, wide_spread, missing_features, outcome_revised, decision_clamped_to_open | same |
| Metrics: Brier, log loss, ECE, reliability table, Murphy decomposition, skill score, bootstrap and **cluster** bootstrap CIs, paired-difference CIs | `predlab/evaluation/metrics.py` |
| Splits: frozen most-recent holdout (SHA-256 of its tickers), expanding walk-forward, label-availability and embargo rule, leak assertions | `predlab/evaluation/splits.py` |
| Holdout guard: each model version can be scored on a given holdout once (stored in `holdout_evaluations`) | same, `db/schema.py` |
| Models: market_implied (reference), frequency_bayes (Beta-Binomial, posterior credible interval), logistic (none/L2/L1, C chosen by time-ordered CV, cluster-bootstrap CI), hist_gbm, lightgbm (TreeSHAP explanations, no CI) | `predlab/models/` |
| Forecast record: probability, CI, model version, input_max_ts, data-quality flags, explanation | `predlab/models/base.py` |
| Append-only hash-chained forecast log with tamper check | `predlab/forecasts.py` |
| Walk-forward runner (paired: every model gets the same rows) and one-time holdout evaluation | `predlab/evaluation/walkforward.py` |
| CLI: `predlab evaluate`, `predlab holdout` | `predlab/cli.py` |

## Test results
- `pytest`: **127 passed**. `ruff`: clean. `test_no_live_execution.py`: passes.
- Metrics match hand calculations (Brier 0.065, log loss, ECE). A property test checks that the Murphy decomposition always sums to the Brier score.
- Leakage canaries: injecting a candle or observation dated after the decision time doesn't change the features. A deliberately leaky feature builder raises `LookAheadError`. Adding a validation row to training is caught.
- Recovery check: logistic regression (without market features) beats a constant forecast by more than 0.02 Brier, lands within 0.04 of the oracle that knows the true probability, and correlates above 0.75 with the true p. The pure-noise feature x3 is almost never the top driver.
- The holdout can't be scored twice by the same model version.

## Results so far (development folds only; the holdout is untouched)
**Synthetic** (365 days, all models, market features on):
- weather: logistic_l2 vs market Brier −0.0021, CI [−0.0077, +0.0039]. **Not distinguishable.**
- financial: the market beats every model.
- Tree models overfit at 700–1,100 rows.

**Real: KXHIGHNY** (9,826 markets, one feature `wx_margin_clim7`, no market features):
- logistic_l2 Brier 0.1445 [0.1436, 0.1454] vs base rate 0.1485, a difference of −0.0039 [−0.0044, −0.0035].
- **Caveat:** that comparison is against the base rate, not the market. No candles had been downloaded, so `market_implied` fell back to the base rate. The CLI now prints a warning whenever market-price coverage is under 95%.
- Real-market comparison: _pending candle download (see below)._

## Unsupported assumptions
1. Decision time = close − 6h by default. Real traders could act earlier or later. The `--horizon-hours` option exists, but results haven't been checked for sensitivity to it.
2. Label availability = Kalshi `expiration_time`, which is the *latest* possible settlement. That's conservative, and it throws away some training rows near fold boundaries.
3. When an outcome was revised, the *latest* result is used as the label, with a flag. A strict point-in-time label would use whatever result was public at training time.
4. Weather station per series is assumed (KXHIGHNY → Central Park). Kalshi settles on The Weather Company's data.
5. The NCEI 2-day publication lag is assumed (see Phase 1).
6. Cluster = event. Markets in the same event (strike ladders) are treated as correlated, but different events on the same day are treated as independent. A cross-event daily shock (the synthetic `rho`) isn't clustered yet.
7. Tree defaults were made more conservative once, **after** seeing synthetic development-fold results (never the holdout). That's a mild form of tuning on validation data.
8. Market mid = (bid + ask) / 2 of the last *completed* hourly candle. It's a reference probability only, never an execution price.

## Unresolved issues
- Probability calibration (isotonic/Platt) isn't applied yet. ECE is reported, and calibration is planned for Phase 4.
- Hist-GBM has no per-row explanations (it only notes global importance). LightGBM gives per-row SHAP values.
- `wx_margin_nws` will stay mostly empty until NWS forecast snapshots build up over time.
- Brier CIs on KXHIGHNY are narrow because of the large sample. Statistically significant doesn't mean economically meaningful. That's Phase 3's job (fees and spread).
- Bootstrap CI refits use the C chosen on the full training window. They don't re-run the C search, so they understate model-selection uncertainty.
