# Phase 3: Historical simulation (2026-10-29)

**PAPER TRADING — NOT REAL MONEY.** Status: complete, awaiting review.

## What was built
| Piece | File |
|---|---|
| Kalshi fee schedule: taker `ceil_cent(M·0.07·C·P·(1−P))`, maker `0.0175` where applicable, no-fee series, fee stress knob. Source: kalshi.com/fee-schedule, read 2026-10-29 | `predlab/backtest/fees.py` |
| Quote lookup: book snapshot (≤1h old), then last completed candle (≤2h), then no trade. Staleness is enforced, and a quote dated after the decision raises `LookAheadError` | `predlab/backtest/execution.py` |
| Fills: buy-only, held to settlement. Walks real ask levels up to a limit price and takes at most 50% of each displayed level. **Never fills at the midpoint.** Candle-only fills are capped at 10 contracts, charged +$0.01 slippage, and flagged `liquidity_uncertain` | same |
| Portfolio: cash, positions, per-fill fees, $1/$0 settlement, gross vs net P&L. Open positions are marked at the entry-time bid | `predlab/backtest/portfolio.py` |
| Event-driven engine: settlements are processed before decisions at equal timestamps. The strategy sees only the forecast and the quote. Sizing is fractional Kelly with caps per market, per event (correlation), and on total exposure | `predlab/backtest/engine.py` |
| Report: return, max drawdown, gross/net P&L, fees, fills, contracts, uncertain-fill share, rejections by reason, peak exposure, hit rate, Brier on all vs traded signals, **cluster-bootstrap 95% CI of net P&L** | same |
| Fee × slippage sensitivity grid. Runs are saved to the simulation DB only (`sim_runs`, `sim_orders`, `sim_fills`, `sim_positions`, `sim_equity`) | same |
| CLI: `predlab backtest --model …` trades the model's **out-of-sample walk-forward forecasts only**. The holdout isn't touched | `predlab/cli.py` |

## Test results
- `pytest`: **152 passed**. `ruff`: clean. `test_no_live_execution.py`: passes.
- Fees reproduce the published 100-contract ranges ($0.07 to $1.75 taker, $0.44 maker) and the round-up rule.
- Book walk: two-level book with a 50% participation cap → 75 of 100 contracts at the hand-calculated average price. A limit order stops at the first level. A NO buy uses 1 − YES bid.
- Midpoint test: bid 0.40 / ask 0.45 fills at 0.45, never 0.425.
- **Outcome independence:** flipping every outcome leaves every order identical, while P&L flips sign.
- Same-timestamp ordering: a settlement funds a decision at the same instant. In the control run where M1 loses, M2 can't afford a contract.
- Sanity bounds on synthetic data: an oracle (true p) makes a net profit, and trading on the market's own mid makes **zero** trades after spread and fees.
- Sensitivity: 3× fees with 3¢ slippage is worse than the baseline.

## Results so far (out-of-sample walk-forward signals; the holdout is untouched)
**Synthetic weather, logistic_l2, $1,000:** 329 fills (all capped candle fills), gross +$165.20, fees $48.28, **net +$116.92, 95% CI [−$29.90, +$265.63]**. Not distinguishable from zero. Max drawdown 4.75%. Net falls to +$21.40 at 3× fees and 3¢ slippage.

**Real KXHIGHNY** (9,826 markets 2021-08 to 2026-10, decision 30h before close, logistic_l2 with market and weather features, $1,000):
- **Forecasting:** Brier 0.1192 vs market mid 0.1269. Model − market = −0.0078, CI [−0.0093, −0.0064]. The edge is statistically real, but small. It looks like a calibration correction to the market price (market ECE 0.038).
- **Trading:** 283 fills, gross −$56.70, fees $14.75, **net −$71.45 (−7.1%), 95% CI [−$150.54, +$3.26]**. Max drawdown 9.8%. Hit rate 13%: the strategy mostly bought cheap long-shot contracts, and those mostly lost.
- **Conclusion: the forecasting edge does not survive spread, fees and conservative fills.** The sensitivity grid moves *toward* zero as costs rise, because fewer (losing) trades get through.
- Every fill is candle-based (`uncertain_fill_share` 1.0) because no historical order books exist. 828 signals were rejected as stale quotes and 57 for no liquidity data.
- Weather-only features (no market price) lose to the market by 0.018 Brier.

## Bugs found during Phase 3 (fixed)
1. **Archive candle prices were 100× too small (a Phase 1 normalizer bug).** `/historical/.../candlesticks` returns dollar *strings* under plain keys (`"close": "0.0100"`). The normalizer treated plain keys as legacy integer cents and divided by 100. As a result, KXHIGHNY's "market price" was about 0 for 99% of markets, the market reference scored *worse* than the base rate, and logistic regression looked implausibly good (Brier 0.022). It was caught because those numbers failed a plausibility check. Fix: strings are read as dollars and only numeric values are divided by 100. A regression test covers both formats. A new `sub_cent_price` integrity check would have flagged it at ingest. The affected candles were deleted and downloaded again.
2. **Decision horizon too short for daily weather.** With the default `close − 6h`, a NYC daily-high market is "decided" around 7pm ET on the day itself, when the high is usually already known. `evaluate` now warns when more than 50% of markets are priced ≤3¢ or ≥97¢ at decision time. Real weather runs use `--horizon-hours 30` (the evening before).

## Unsupported assumptions
1. Fee rounding is applied per fill (one fill per book level), which is more conservative than per order. There's no settlement fee. Neither is verified.
2. A 50% participation rate on displayed depth, the 10-contract cap when depth is unknown, and +1¢ slippage are judgment calls, not measured. The sensitivity grid covers slippage, but the participation rate and the depth cap aren't swept yet.
3. Taker-only execution. Resting (maker) orders, queue position and cancellations aren't modeled.
4. Positions are held to settlement. There's no early exit, so drawdown only reflects marks at the entry-time bid.
5. Capital is locked until `label_ts` (expiration_time, the latest possible settlement), which is conservative.
6. Order-book snapshots exist only from our own snapshotting. All historical real-market fills will come from candles and be flagged uncertain.

## Known limitations
- The sensitivity grid isn't strictly monotone. Higher costs remove borderline trades, and some of those were losers. It's a selection effect, and the CI shows the noise.
- Kelly sizing uses the model's own probability as truth. Uncertainty in the forecast isn't used in sizing yet (planned for Phase 4 stress tests).
- The per-event cap limits same-event correlation. Correlation across events on the same day (for example, a weather front hitting several cities) isn't capped yet.
