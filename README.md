# predlab: prediction-market forecasting research

**PAPER TRADING — NOT REAL MONEY.** predlab only reads public data and simulates trades.
It has no code path that places, amends or cancels a real order, and no setting that enables one.
`tests/test_no_live_execution.py` enforces this.

Status: Phase 1 (data ingestion and storage) is done. Models, backtesting, the research
loop and the dashboard follow in Phases 2–6. See `docs/phase-1-notes.md`.

## Setup (Windows 11, PowerShell)
```powershell
cd "C:\Users\brend\orca\projects\Personal Projects"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dev]"
Copy-Item .env.example .env           # optional: add a free FRED key, set your contact User-Agent
predlab init-db
```

## Collect data
```powershell
predlab synth --days 365                                  # synthetic data for all 5 categories
predlab ingest-kalshi --series KXHIGHNY --series KXCPIYOY # public Kalshi markets (live + archive)
predlab ingest-kalshi --series KXHIGHNY --status open --books   # snapshot live order books
predlab ingest-weather --start 2026-01-01 --end 2026-10-27
predlab ingest-fred                                       # needs PREDLAB_FRED_API_KEY
predlab load-sports path\to\games.csv
predlab health                                            # source health, data-quality issues, row counts
```

## Tests
```powershell
pytest              # offline suite
pytest -m live      # opt-in: real Kalshi public read-only endpoints
```

Historical data goes to `data/market.db` and simulated data to `data/sim.db`. Both are
git-ignored, and so is `.env`.
