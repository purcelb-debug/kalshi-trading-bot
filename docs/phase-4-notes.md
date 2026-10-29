# Phase 4: Multi-category model evaluation (2026-10-29)

**PAPER TRADING — NOT REAL MONEY.** Status: complete, awaiting review.

## What was built
| Piece | File |
|---|---|
| Price-momentum features `mkt_mom6`, `mkt_mom24` (change in logit(mid) since the mid known 6h / 24h before decision; older quotes only) | `predlab/features/pointintime.py` |
| `MarketTrajectory`: time-series model on the market's own price path | `predlab/models/advanced.py` |
| `Calibrated`: Platt or isotonic recalibration of any model. The base model is fit on the earlier 70% of the training window and the calibrator on the later 30% | same |
| `Stacked`: non-negative, sum-to-one weights over members, chosen by log loss on the later part of training. Members are then refit on the full window | same |
| Registry of all 13 models (`market_platt`, `market_isotonic`, `logistic_l2_platt`, `lightgbm_isotonic`, `stacked`, …) | `predlab/models/registry.py` |
| Category ranking on the spec's 10 criteria, with a composite rank and a "not a recommendation" disclaimer | `predlab/evaluation/category_rank.py` |
| Stress suite, 12 scenarios + day-clustered CI | `predlab/backtest/stress.py` |
| Overfitting probes: shuffled-label retraining and a Bonferroni multiple-testing check | `predlab/evaluation/overfit.py` |
| CLI: `predlab compare-categories`, `predlab stress` | `predlab/cli.py` |

## Test results
- `pytest`: **175 passed**. `ruff`: clean. `test_no_live_execution.py`: passes.
- The momentum feature matches a hand calculation from raw candles.
- Platt recalibration of a deliberately distorted model (2.5× logit stretch) recovers a slope of 0.25–0.6 (inverse ≈ 0.4) and lowers both ECE and Brier.
- The stacked ensemble's weights sum to 1, are non-negative, and put more than 60% on an oracle member.
- Stress suite on an oracle: adverse price moves, wrong resolutions, 3× fees and forecast noise each lower net P&L, and per-fill fees rise under fee stress.
- **Stale-quote regression test:** a stale view can trigger an order, but the fill must use the current book. A cheap old quote with an expensive current book → order placed, **0 fills**.
- Shuffled-label probe: it flags a model that derives predictions from the labels, and doesn't flag logistic_l2.

## Bug found and fixed in this phase
- **The stale-quote and price-move stress scenarios priced *and filled* orders off the stressed quote.** A 3-hour-stale quote produced **+$308 (CI excluding 0)** on synthetic weather. That was money invented by filling at an old, less-informed price. The engine now separates the *decision* quote (what the strategy sees) from the *execution* quote (what the order actually trades against). After the fix, stale quotes show +$5, which is noise.

## Results
**Real KXHIGHNY, 30h horizon, walk-forward (6,551 validated markets, holdout untouched):**

| model | Brier | vs market (95% CI) |
|---|---|---|
| logistic_l2 | 0.1192 | −0.0077 [−0.0092, −0.0064] |
| logistic_l2_platt | 0.1195 | −0.0075 [−0.0089, −0.0060] |
| stacked | 0.1219 | −0.0050 [−0.0065, −0.0039] |
| market_platt | 0.1242 | −0.0027 [−0.0038, −0.0019] |
| market_isotonic | 0.1247 | −0.0022 [−0.0036, −0.0010] |
| market_trajectory | 0.1253 | −0.0016 [−0.0024, −0.0010] |
| market_implied | 0.1269 | reference |

- Recalibrating the market price by itself recovers about a third of the best model's forecasting edge. Price momentum adds little.
- `mkt_mom24` drops out on real data: markets open only about 8h before a 30h-ahead decision.
- The stacked ensemble is *worse* than its best member, so stacking doesn't help here.

**Real KXHIGHNY stress suite (logistic_l2, $1,000, out-of-sample signals):**

| scenario | net P&L | 95% CI | fills |
|---|---|---|---|
| base | −$79.29 | [−$158.23, −$1.21] | 304 |
| **forecast_noise_0.5** | **−$846.00** | [−$1,134.92, −$548.73] | 1,840 |
| adverse_price_move_2c | −$23.47 | [−$62.84, +$17.01] | 66 |
| stale_quotes_3h | −$48.35 | [−$103.19, +$2.47] | 101 |
| data_outage_20pct | −$64.33 | [−$134.76, +$7.15] | 241 |
| api_failures_10pct | −$52.10 | [−$128.65, +$27.58] | 271 |
| forecast_outage_20pct | −$59.76 | [−$127.28, +$6.47] | 242 |
| delayed_resolution_7d | −$79.29 | [−$158.23, −$1.21] | 304 |
| wrong_resolution_5pct | −$19.29 | [−$114.20, +$77.38] | 304 |
| fees_x2 | −$50.06 | [−$99.75, +$0.42] | 229 |
| fees_x3 | −$48.70 | [−$83.33, −$7.69] | 181 |
| thin_liquidity | −$40.44 | [−$79.87, −$1.38] | 304 |
| correlated_by_day (CI only) | −$79.29 | [−$160.41, +$3.12] | — |

- **The base strategy loses money on real data.** The 95% CI is entirely below zero, clustered by event.
- **Forecast uncertainty is the dominant risk.** 0.5 logit noise turns −$79 into **−$846 (−85%)**. Noise manufactures apparent edges on cheap long-shot contracts, and Kelly sizing treats them as real. The sizing rule currently ignores forecast uncertainty. That needs fixing before any money is at risk.
- Most other stresses look *better* than base only because they block trades of a losing strategy. For a strategy with no edge, they say little about resilience.
- Clustering by settlement day instead of by event widens the CI to include 0. Same-day markets are correlated beyond their event.
- **Overfitting probes:** logistic_l2 on shuffled labels gains −0.00013 vs a constant (no leak). Its *forecasting* edge vs the market survives Bonferroni: −0.0077, CI [−0.0091, −0.0063]. The model forecasts slightly better than the market, and the strategy still loses money.

**Real category comparison (30h horizon; only categories with downloaded data):**

| rank | category | n | best model | Brier | vs market | OOS win share | independent info | market coverage | est. cost/contract |
|---|---|---|---|---|---|---|---|---|---|
| 1 | weather | 9,826 | logistic_l2 | 0.1192 | −0.0077 | 0.8 (vs market) | 98% | 99% | $0.068 |
| 2 | economics | 671 | market_platt | 0.1274 | +0.0144 | 1.0 (vs base rate) | **0%** | 71% | $0.043 |

- Economics win share is measured against the base rate because market coverage is below 95%. Against the market price, the best economics model is worse.
- **Economics has no independent information yet.** The FRED CPI vintages are stored, but no feature builder links them to CPI markets.
- Financial, sports and public events have no real Kalshi data downloaded and were skipped.

## Recommended next steps (from these results)
1. Make sizing uncertainty-aware: shrink p toward the market by its forecast uncertainty, or size from the lower CI bound. Use the forecast-noise stress as the acceptance test.
2. Add a FRED-vintage feature builder for CPI and other economic markets (point-in-time consensus vs prior release).
3. Ingest one or two real series each for financial, sports and public events so the category ranking covers all five.

## Unsupported assumptions
1. The calibration split (70/30 within each training window) and the stacking split (70/30) are fixed choices that haven't been tuned.
2. Stress magnitudes (2¢ moves, 3h staleness, 20% outages, 10% API failures, 5% wrong resolutions, 7-day delays) are illustrative, not estimated from Kalshi incident data.
3. "Wrong resolution" flips outcomes at random. Real errors are probably correlated (one bad data source affects a whole event).
4. Category criteria are weighted equally in the composite rank. Freshness is measured against today, so it penalizes old synthetic data and categories without recent settlements.
5. The shuffled-label probe tolerance (0.005 Brier) is a judgment call.

## Known limitations
- Only categories with downloaded data can be compared on real markets. The real database has weather (KXHIGHNY) and economics (KXCPIYOY). Sports and public events are synthetic only.
- Stress tests reuse the base run's signals. They don't retrain models under stressed data.
