"""Executor settings, read from ``executor/.env`` (git-ignored) with the EXECUTOR_ prefix.

Safety defaults: mode is ``dry_run``; money limits have NO defaults, so demo/live mode
refuses to start until the owner sets them explicitly.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent

GATE_OVERRIDE_PHRASE = "I_ACCEPT_TRADING_AN_UNVALIDATED_STRATEGY"
#: Settings a deployment may override; money limits stay shared.
STRATEGY_KEYS = {"shrink", "kelly_fraction", "min_edge"}

BASE_URLS = {
    "demo": "https://external-api.demo.kalshi.co/trade-api/v2",
    "live": "https://api.elections.kalshi.com/trade-api/v2",
}


class ExecutorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXECUTOR_", env_file=str(ROOT / ".env"),
                                      env_file_encoding="utf-8", extra="ignore",
                                      env_ignore_empty=True)   # blank = not set

    mode: Literal["dry_run", "demo", "live"] = "dry_run"

    # Credentials: key id + path to a private-key file kept OUTSIDE the repository.
    demo_key_id: str = ""
    demo_key_path: str = ""
    live_key_id: str = ""
    live_key_path: str = ""

    # Money limits, in dollars. None = not configured = demo/live refuse to run.
    bankroll_cap_usd: float | None = Field(default=None, gt=0)
    #: Owner's choice (2026-11-06): the bankroll is whatever is available on Kalshi at the
    #: time (cash minus resting buys). Replaces bankroll_cap_usd; the other limits stay.
    cap_to_available: bool = False
    max_order_usd: float | None = Field(default=None, gt=0)
    daily_loss_stop_usd: float | None = Field(default=None, gt=0)
    max_market_usd: float | None = Field(default=None, gt=0)
    #: Most money in one event (all its strikes, e.g. one day's NY high), counting held
    #: positions and resting buys across every strategy. None = no event limit.
    max_event_usd: float | None = Field(default=None, gt=0)
    #: Orders that pass the edge test are raised to at least this size (None = Kelly only).
    min_order_usd: float | None = Field(default=3.0, gt=0)
    #: Owner's daily goal: once the account is up this much since the day's first cycle
    #: (US Eastern day), no new buys until tomorrow. Exits still run. None = no lock.
    daily_profit_target_usd: float | None = Field(default=None, gt=0)

    max_orders_per_day: int = Field(default=20, gt=0)
    max_consecutive_errors: int = Field(default=3, gt=0)
    min_edge: float = 0.03               # $/contract expected value after fees
    #: Skip a buy whose (shrunk) forecast is more than this above the price paid. A gap that
    #: wide on a traded market is almost always a forecast error (2026-11-06: the desk's
    #: 25-30c "edges" on FX and crypto went to zero). None = no limit.
    max_forecast_gap: float | None = Field(default=None, gt=0)
    kelly_fraction: float = 0.25
    shrink: float = Field(default=0.5, ge=0.0, le=1.0)   # pull forecasts toward market mid
    min_entry_price: float = 0.05
    max_entry_price: float = 0.95
    max_quote_age_s: float = 60.0         # order book must be this fresh to trade
    #: taker: IOC buy at the ask. maker: post-only buy 1c above the bid, kept while its
    #: side and price still hold, cancelled when the edge goes, and expiring on Kalshi
    #: after maker_ttl_min even if the bot stops running.
    order_style: Literal["taker", "maker"] = "taker"
    maker_ttl_min: float = Field(default=60.0, gt=0)
    #: Positions closing within this many hours keep their take-profit but drop the stop.
    no_stop_hours: float = Field(default=24.0, ge=0)
    #: Per-deployment overrides, as JSON in the env:
    #: {"<model_version>": {"shrink": 0.7, "kelly_fraction": 0.6}}. Shrink still never
    #: goes above the deployment's certified kappa.
    strategy_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)

    def for_strategy(self, model_version: str) -> "ExecutorSettings":
        o = self.strategy_overrides.get(model_version) or {}
        bad = set(o) - STRATEGY_KEYS
        if bad:
            raise ValueError(f"strategy_overrides for {model_version}: unknown {sorted(bad)}")
        return self.model_copy(update=o) if o else self

    # Go-live gate (live mode only).
    gate_paper_weeks: int = Field(default=4, ge=1)
    gate_min_paper_trades: int = Field(default=50, ge=1)
    #: Owner override: trade live although the gate is closed (unvalidated strategy).
    #: Only the exact phrase GATE_OVERRIDE_PHRASE enables it; every cycle is audited.
    gate_override: str = ""
    attended: bool = True                 # confirm each live order interactively

    kill_switch_path: str = str(ROOT / "KILL_SWITCH")
    data_dir: str = str(ROOT / "data")
    sim_db_url: str = "sqlite:///data/sim.db"   # where predlab writes forecasts/registry

    def credentials(self) -> tuple[str, str]:
        if self.mode == "demo":
            return self.demo_key_id, self.demo_key_path
        if self.mode == "live":
            return self.live_key_id, self.live_key_path
        return "", ""

    @property
    def gate_overridden(self) -> bool:
        return self.mode == "live" and self.gate_override == GATE_OVERRIDE_PHRASE

    def missing_limits(self) -> list[str]:
        need = (() if self.cap_to_available else ("bankroll_cap_usd",)) + (
            "max_order_usd", "daily_loss_stop_usd")
        return [n for n in need if getattr(self, n) is None]

    @property
    def bankroll_label(self) -> str:
        return "available balance" if self.cap_to_available else f"${self.bankroll_cap_usd:g}"

    @property
    def base_url(self) -> str:
        return BASE_URLS["demo" if self.mode == "dry_run" else self.mode]
