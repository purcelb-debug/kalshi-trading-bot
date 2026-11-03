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
- **Six-role LLM desk** (`desk/`, outside predlab), deployment `desk@<series>-h<h>`. Roles
  that don't search (Scanner, Pricer, Risk, Desk Lead) run on `DESK_PROVIDER` (default Gemini
  `gemini-3.8-flash` on a free key, Interactions API with `store=False`). News Checker and
  Skeptic need web search and run on `DESK_SEARCH_PROVIDER` (default Claude `claude-opus-5-5`
  with its web search tool), so the Anthropic key is only spent on searching roles. The roles are Scanner, News Checker
  (web search), Pricer, Risk Manager, Skeptic (web search, separate model setting) and
  Desk Lead, with prompts in `desk/prompts.json` (reloaded every meeting). The desk proposes
  `p_yes` and an action. Code then:
  - keeps only tickers that were on the table, at most 1 BUY per event and `DESK_MAX_BUYS`
    per meeting, and clips p to [0.02, 0.98], so malformed output means no trades;
  - records only BUYs as forecasts under `desk@<series>-h<h>`;
  - replays them through the same engine, shrink (fixed 0.5), Kelly sizing, caps, alerts
    and gate evidence as fitted models (`update_paper_record`, extracted from `run_cycle`).
  - Each market is shown to the desk once, at its decision point. The daily meeting cap
    defaults to 4. Claude calls use `fallbacks="default"` for refusals. Google Search grounding
  needs a paid-tier Gemini key: on a free key, News and Skeptic answer without search and the
  transcript says so. Every call logs token use in
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
2. Desk: put `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` in `.env`, optionally copy `desk/.env.example` to
   `desk/.env`, then run `python -m desk.cli register --series KXHIGHNY` (meetings start on
   the next hourly run). `python -m desk.cli cycle --dry` previews the table for free.
3. Scheduling: read and run `scripts/harden_task.ps1`.

## 2026-11-01 later: wider coverage, minimum order
- **Minimum order.** `EXECUTOR_MIN_ORDER_USD` (owner set $3): an order that already passes the
  edge test is raised to the smallest size of at least that amount, still inside cash, the
  per-order cap and the other caps. It never creates a trade without edge. The paper record
  still sizes by plain Kelly, so live P&L will differ from the scorecard.
- **Desk:** the daily meeting cap (default 8) is shared by every desk deployment, and
  `desk.cli cycle --all` runs them all (one failing market doesn't stop the rest). News and
  Skeptic default to `claude-sonnet-5-5`.
- **Hourly job:** `scripts/run_loop.py` takes a list of quant deployments: all research cycles
  first, then the desk, then one executor run per deployment. Each step has a 20 min timeout
  and its run time is logged.
- **New series checked (development walk-forward, holdout untouched, 30h):** KXINX and
  KXAPRPOTUS have no candle history (0% market-price coverage), so they can't be compared
  with the market and were not added. On KXCPIYOY every model was worse than the market
  itself. None of the three trade.
- **Desk runs twice a day per city:** desk deployments are registered at two horizons for
  each of KXHIGHNY, KXHIGHCHI and KXHIGHMIA: 30h before close (about 7-8 PM ET the day before)
  and 12h before close (about 1-2 PM ET on the day). Each horizon is its own paper record,
  and a market is shown once per horizon. That's 6 meetings a day, inside the shared cap of 8.

## 2026-11-02: closer to the DESK post (owner's choices)
- **`desk@ALL-h48`** scans every open Kalshi market closing within 48h (~10,000; ~500 pass
  the filters: both sides quoted, mid 0.05-0.95, spread <= 6c, volume >= 100, >= 1h to close)
  and shows the desk the top 40 by 24h volume, at most 2 per event. Each market is shown
  once. There is no paper replay: these markets aren't in the research database, so its
  record is the executor ledger.
- **The desk trades real money**: `scripts/run_loop.py` runs the executor on `desk@ALL-h48`
  after each desk cycle, under the gate override, with all the shared caps.
- **Exits**: the Pricer sets take_profit / stop (prices of the side bought), and the Lead
  passes them in TRADES. Code clips them and drops both unless stop + 3c <= take_profit.
  Every executor cycle checks held positions for that deployment before buying, and sends
  an IOC `reduce_only` sell at the current bid when the bid is at or above take_profit, or at
  or below stop. Exited tickers are never bought again (`State.closed`). Telegram reports
  SOLD lines. The paper engine still holds to settlement.
- **Cadence and cost**: the job runs every 30 minutes. Desk limits per UTC day, shared by all
  desk deployments: 24 meetings and `DESK_DAILY_BUDGET_USD` (default $1), estimated from
  measured token use x `DESK_PRICES` plus $0.01 per web search. Each transcript ends with
  its estimated cost.
- **Reliability**: Gemini calls retry 503/429/connection errors (2s, 6s, 15s). A role whose
  provider still fails is answered by the other provider, and the transcript notes it.
  Claude clients use 4 retries.
- **Meeting limits (after the first broad meeting hit the 20-minute step timeout):** 20-row
  table, at most 3 searches per searching role, 180s per LLM request, 10-minute meeting
  deadline, at most 2 meetings per job run (others are "deferred" to the next run), and a
  28-minute limit for the desk step. A meeting that stops early (deadline, or both providers
  down) is still recorded: its spend counts toward the budget, its markets count as seen,
  and a partial transcript is never parsed for trades.
- **Success baseline (owner, 2026-11-02): at least $8 profit per day.** Measured as the
  change in account value (cash + positions) from one ET day's last balance snapshot to
  the next. Telegram sends a daily MET/MISSED line with the 7-day average. Deposits and
  withdrawals move the number too (the Nov 1 deposit shows as +$25.59). The target is a
  measurement only and never changes sizing or limits.
- **Why meetings kept timing out (found 2026-11-03):** gemini-3.8-flash's free tier allows
  20 requests a day. Once used up, Google returns 429 with Retry-After of ~12 hours, and the
  google-genai SDK slept for it inside the call (it honours Retry-After uncapped and its
  interactions client ignores the client retry setting). The desk now calls the Gemini REST
  API directly with httpx (`desk.gemini.GeminiRest`, no hidden retries), treats 429 as
  "this model is out of quota": next model in `DESK_GEMINI_MODELS` (3.8-flash, 3.6-flash,
  3.5-flash-lite, 3.1-flash-lite, each with its own free quota), then Claude. Every role
  also has a hard wall-clock limit (`DESK_ROLE_TIMEOUT_S`, 240s) enforced outside any SDK.
- **$8/day push mode (owner-approved 2026-11-03, real money):** executor limits raised to
  bankroll $40, max order $15 (min $3), Kelly 0.5, min edge 2c; daily loss stop unchanged at
  $25. Desk: $2/day LLM budget, 5 picks and 5 BUYs per meeting. New
  `EXECUTOR_DAILY_PROFIT_TARGET_USD=8`: once the account is up $8 since the day's first
  cycle (US Eastern day; the daily loss stop uses the same day now), no new buys until
  tomorrow; exits still run, and Telegram says "DAILY GOAL MET" once. These settings make
  more and larger bets. They don't add edge, and they make losing days bigger.
- **Free desk (2026-11-03, owner: stop paying for Anthropic):** all six roles run on free
  Gemini by default (`DESK_SEARCH_PROVIDER=gemini`). News Checker and Skeptic search the
  web through code: the model writes up to `DESK_SEARCHES_PER_ROLE` (3) queries, `desk.search`
  runs them on Tavily's free tier (1,000/month, `TAVILY_API_KEY`, capped at 30/day in
  `data/desk_search_usage.json`), and the model answers from the numbered, untrusted
  results, citing [S#] and URL. With no key or the cap reached, the roles answer without
  search and say so. No Claude client is created unless `DESK_SEARCH_PROVIDER=claude`.
  Live check: correct query, cited answer, 3.2s, $0. Limits: Gemini's free quota is about
  20 requests a day per model, so late in a busy day roles fall to flash-lite models.

- **Groq backup and cleaner search (2026-11-03):** with `GROQ_API_KEY` set, the free chain
  continues after the four Gemini models with Groq's `openai/gpt-oss-120b` and
  `qwen/qwen3.8-27b` (free: ~1,000 requests/day, 8,000 tokens/minute; short "try again in Ns"
  waits are honoured, too-large requests move on). Search results now drop page furniture
  (lines under 25 chars) and keep 1,500 chars from each of 3 results. Before this the NWS
  forecast numbers were cut off and the desk correctly said NO CLEAR INFORMATION; after it,
  gpt-oss-120b found the official Central Park point forecast (HARD, cited) in 5.9s, $0.
