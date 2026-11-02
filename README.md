# kalshi-trading-bot

A research system for [Kalshi](https://kalshi.com) prediction markets. It collects
public market data, builds point-in-time features, compares forecasting models against the
market price, paper-trades every candidate, and sends one candidate to a small, capped
real-money executor. A six-role LLM "desk" paper-trades alongside the models, so the
two approaches can be compared on the same scorecard.

> **Not financial advice.** This is a personal research project. The live executor trades
> a deliberately tiny bankroll, and so far no model here has shown an edge over the market
> after fees (see [Results](#results)).

## How it fits together

```
                 public Kalshi API, NOAA/NCEI, NWS, FRED
                                  |
                                  v
  predlab/   data collectors -> point-in-time features -> models -> walk-forward evaluation
             -> holdout certification -> hourly paper-trading loop -> go-live gate evidence
                                  |                           |
                  desk/ (LLM, paper only)            executor/ (real money, capped)
                                  |                           |
                                  +-----> notify/ (Telegram) <+
```

| Package | What it does |
| --- | --- |
| `predlab/` | Research core. Data ingestion (Kalshi markets, candles, trades, order books; NCEI observed temperatures; NWS forecasts; FRED macro series), point-in-time feature building with look-ahead checks, 13 models (from market-implied baselines to logistic/LightGBM/stacked ensembles), walk-forward evaluation with an untouched holdout, an event-driven backtest engine with Kalshi's fee schedule, stress tests and overfitting probes, and the hourly paper-trading loop. **predlab never places orders**: `tests/test_no_live_execution.py` fails the build if it makes a write request or imports the trading packages. |
| `executor/` | Places real orders on Kalshi for one deployment. Hard caps in code: bankroll, per-order and per-market limits, daily loss stop, orders per day, consecutive-error halt, kill-switch file, never adds to a held position, and a go-live gate. It keeps a hash-chained audit log and reconciles positions from the exchange before every cycle. |
| `desk/` | Six LLM roles (Scanner, News Checker, Pricer, Risk Manager, Skeptic, Desk Lead) debate each market at its decision point. The roles that need web search run on Claude, the rest on Gemini. **Their output is untrusted text.** Code keeps only tickers that were on the table, one BUY per event, a few per meeting, and clips probabilities. Sizing and caps are never the model's call. The desk paper-trades only. |
| `notify/` | Sends Telegram messages only when something changes: alerts, gate flips, fills, halts, desk transcripts, a stale loop. Also sends a weekly scorecard. |
| `scripts/` | The hourly Windows Task Scheduler job (research loop, desk, executor, ledger sync, digest) and the task setup script. |

## Design choices

- **Evidence before money.** A model reaches the executor only after the go-live gate in
  `executor/gate.py` passes: a holdout CI that beats the market on Brier score, a holdout P&L CI above zero,
  at least 4 weeks of forward paper trading, zero degradation alerts and at least 50
  settled paper trades. The owner can override it with an explicit phrase, and every
  override is written to the audit log.
- **One look at the holdout.** `predlab certify` evaluates a model on held-out data exactly
  once per deployment, so the holdout can't be tuned against.
- **Shrink toward the market.** Forecasts are blended with the market mid using a factor
  fitted on development data, and trades need a minimum edge after fees.
- **LLMs propose, code decides.** A prompt-injected or confused desk transcript can, at
  most, change a probability for a market that was already on the table.

## Results

Walk-forward comparisons on development data (30h before close, holdout untouched).
Brier score difference vs the market price; below zero means better than the market:

| Series | Best model | vs market | Notes |
| --- | --- | --- | --- |
| NY high temp (KXHIGHNY) | logistic_l2 | -0.0077 [-0.0091, -0.0063] | Forecasts better than the market, but the strategy still lost money after fees in backtests |
| Chicago high temp (KXHIGHCHI) | market_platt | 0.0000 [-0.0015, 0.0014] | No learned model beats the market |
| Miami high temp (KXHIGHMIA) | market price | n/a | Every model is worse |
| CPI (KXCPIYOY) | market price | n/a | Every model is worse |
| S&P 500 range, Trump approval | n/a | n/a | No candle history, so they can't be compared with the market |

In short, Kalshi's prices are hard to beat, and the fees take most of what's left. The
stress tests also showed that forecast noise creates fake edges on cheap long-shot
contracts. That's why the gate exists.

## Setup (Windows, PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,llm]"
Copy-Item .env.example .env      # FRED key, ANTHROPIC_API_KEY / GEMINI_API_KEY for the desk
predlab init-db
```

Collect data and evaluate:

```powershell
predlab ingest-kalshi --series KXHIGHNY --status settled --candles --since 2024-12-01
predlab ingest-weather --start 2024-12-01 --end 2026-10-31
predlab evaluate --series KXHIGHNY --horizon-hours 30
predlab register --model logistic_l2 --series KXHIGHNY      # paper trading, holdout untouched
predlab loop --deployment "<id printed above>" --once
predlab scorecard --all
```

The desk and notifier are opt-in. See `desk/.env.example`, `notify/.env.example` and
`executor/.env.example`. Secrets live only in git-ignored `.env` files, and the exchange
private key is kept outside the repository.

## Tests

```powershell
pytest              # offline suite (no network; LLM and exchange clients are faked)
pytest -m live      # opt-in: real Kalshi public read-only endpoints
ruff check .
```

Design notes for each phase are in `docs/`.
