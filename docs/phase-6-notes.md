# Phase 6 notes: ideas adopted from the DESK Guide

PAPER TRADING for everything below. Written 2026-11-01.

Source: "The DESK Guide" (six Grok bots debating stock trades on Alpaca paper). predlab
already had the guide's safety core (caps in code, kill switch, audit log, no adding to a
held position, a statistical go-live gate). Only the parts that filled gaps were taken.

## What was built
- **Gate needs settled paper trades.** `executor/gate.py` now also requires
  `paper_settled_trades >= gate_min_paper_trades` (default 50, `EXECUTOR_GATE_MIN_PAPER_TRADES`).
  Before this, 28 quiet days with a handful of trades could open the gate. The loop writes
  the count (`n_settled` in the backtest summary) into the gate evidence every cycle.
- **Weekly scorecard.** `predlab scorecard --deployment X | --all [--days 7]` reports trades,
  win rate, average win and loss, expectancy per trade, worst losing streak, P&L, fees,
  drawdown and forward Brier vs market, all computed in code from the persisted paper run
  (`predlab/loop/scorecard.py`), and writes `reports/scorecard-*.json`.
- **Telegram notifier** (`notify/`, outside predlab, outbound only). `python -m notify.cli
  digest` runs as the last hourly step and sends only changes: degradation alerts (once a
  day), gate open/closed changes, executor fills, errors, blocks, halts, refusals and gate
  overrides (once per change), desk transcripts, and a warning when the loop report is more
  than 2h old. On first sight of an audit log it starts at the end and does not replay
  history. `weekly` sends the scorecard plus the gate verdict. It's off until `notify/.env`
  has a token and chat id (see `notify/.env.example`). Delivery failure keeps the state
  unadvanced, so messages resend next run.
- **Six-role Claude desk** (`desk/`, outside predlab). The roles are Scanner, News Checker
  (web search), Pricer, Risk Manager, Skeptic (web search, separate model setting) and
  Desk Lead, with prompts in `desk/prompts.json` (reloaded every meeting). The desk proposes
  `p_yes` and an action. Code then:
  - keeps only tickers that were on the table, at most 1 BUY per event and `DESK_MAX_BUYS`
    per meeting, and clips p to [0.02, 0.98], so malformed output means no trades;
  - records only BUYs as forecasts under `desk-claude@<series>-h<h>`;
  - replays them through the same engine, shrink (fixed 0.5), Kelly sizing, caps, alerts
    and gate evidence as fitted models (`update_paper_record`, extracted from `run_cycle`).
  - Each market is shown to the desk once, at its decision point. The daily meeting cap
    defaults to 4. Every call uses `fallbacks="default"` for refusals and logs token use in
    the `desk_meetings` table (sim.db), and transcripts go to `reports/desk-meetings/`.
- `scripts/run_loop.py` now runs: predlab loop, desk cycle, executor, sync, notify digest.
  The desk step does nothing until the desk is registered.
- `scripts/harden_task.ps1` (**not run**): turns on WakeToRun for the hourly task, registers a
  Friday 17:00 weekly-scorecard task, and prints the powercfg wake-timer commands.
- Guard: `tests/test_no_live_execution.py` now also fails if predlab imports `executor`,
  `notify` or `desk`.

## Decisions and limits
- WAIT/SKIP are not logged as market-mid forecasts. A stale candle mid could otherwise look
  like an edge to the engine and trade.
- **The desk can never pass holdout certification.** An LLM can't be run point-in-time on
  past markets without lookahead, so its registry status stays `candidate` and the live gate
  stays closed. The only path to live money is the owner's existing `GATE_OVERRIDE_PHRASE`.
- Weather markets resolve on one station's reading. The desk's web search can find the NWS
  forecast, but the quant model already uses NWS data, so the desk has to beat that, not
  just repeat it. The side-by-side scorecard (`predlab scorecard --all`) is the test.
- Desk cost is not estimated. It's measured per meeting in `desk_meetings.usage_json`.
- The PC still sleeps after 30 min on battery. Missed hours are reported by the stale alert
  after the next wake, and they don't count as paper days.

## Owner actions to turn things on
1. Telegram: create a bot with @BotFather, get the chat id, fill in `notify/.env`, then run
   `python -m notify.cli digest --test`.
2. Desk: put `ANTHROPIC_API_KEY` in `.env`, optionally copy `desk/.env.example` to
   `desk/.env`, then run `python -m desk.cli register --series KXHIGHNY` (meetings start on
   the next hourly run). `python -m desk.cli cycle --dry` previews the table for free.
3. Scheduling: read and run `scripts/harden_task.ps1`.
