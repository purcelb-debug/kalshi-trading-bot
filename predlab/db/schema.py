"""Database schema.

Two independent metadata collections so downloaded history and simulated portfolio data
never share a database:

* MARKET_MD -> historical/public data (``market_db_url``)
* SIM_MD    -> forecasts, model registry and PAPER-TRADING simulation (``sim_db_url``)
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.types import TypeDecorator

from predlab.timeutil import UTC, to_utc


class UTCDateTime(TypeDecorator):
    """Stores naive UTC (SQLite has no tz support) and always returns aware UTC."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return to_utc(value).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect):
        if value is None:
            return None
        return value.replace(tzinfo=UTC)


MARKET_MD = MetaData()
SIM_MD = MetaData()

# ---------------------------------------------------------------- historical data

mkt_series = Table(
    "mkt_series", MARKET_MD,
    Column("series_ticker", String, primary_key=True),
    Column("source", String, nullable=False),
    Column("title", Text),
    Column("source_category", String),
    Column("category", String, index=True),
    Column("frequency", String),
    Column("fee_type", String),
    Column("fee_multiplier", Float),
    Column("raw_json", JSON),
    Column("fetched_at", UTCDateTime, nullable=False),
)

mkt_events = Table(
    "mkt_events", MARKET_MD,
    Column("event_ticker", String, primary_key=True),
    Column("source", String, nullable=False),
    Column("series_ticker", String, index=True),
    Column("title", Text),
    Column("sub_title", Text),
    Column("source_category", String),
    Column("category", String, index=True),
    Column("strike_date", UTCDateTime),
    Column("mutually_exclusive", Boolean),
    Column("raw_json", JSON),
    Column("fetched_at", UTCDateTime, nullable=False),
)

mkt_markets = Table(
    "mkt_markets", MARKET_MD,
    Column("ticker", String, primary_key=True),
    Column("source", String, nullable=False),
    Column("event_ticker", String, index=True),
    Column("series_ticker", String, index=True),
    Column("category", String, index=True),
    Column("title", Text),
    Column("market_type", String),
    Column("status", String, index=True),
    Column("open_time", UTCDateTime),
    Column("close_time", UTCDateTime, index=True),
    Column("expiration_time", UTCDateTime),
    Column("result", String),  # "yes" | "no" | "" (unresolved) | other (void/scalar)
    Column("expiration_value", String),
    Column("strike_type", String),
    Column("floor_strike", Float),
    Column("cap_strike", Float),
    # Top-of-book snapshot at fetch time, probabilities in [0, 1].
    Column("yes_bid", Float),
    Column("yes_ask", Float),
    Column("no_bid", Float),
    Column("no_ask", Float),
    Column("last_price", Float),
    Column("volume", Float),
    Column("open_interest", Float),
    Column("updated_time", UTCDateTime),
    Column("raw_json", JSON),
    Column("fetched_at", UTCDateTime, nullable=False),
)

mkt_candles = Table(
    "mkt_candles", MARKET_MD,
    Column("ticker", String, primary_key=True),
    Column("period_minutes", Integer, primary_key=True),
    Column("end_ts", UTCDateTime, primary_key=True),
    Column("price_open", Float),
    Column("price_high", Float),
    Column("price_low", Float),
    Column("price_close", Float),
    Column("price_mean", Float),
    Column("yes_bid_open", Float),
    Column("yes_bid_close", Float),
    Column("yes_ask_open", Float),
    Column("yes_ask_close", Float),
    Column("volume", Float),
    Column("open_interest", Float),
    Column("fetched_at", UTCDateTime, nullable=False),
)

mkt_trades = Table(
    "mkt_trades", MARKET_MD,
    Column("trade_id", String, primary_key=True),
    Column("ticker", String, nullable=False),
    Column("created_time", UTCDateTime, nullable=False),
    Column("yes_price", Float, nullable=False),
    Column("no_price", Float),
    Column("count", Float, nullable=False),
    Column("taker_side", String),
    Column("fetched_at", UTCDateTime, nullable=False),
    Index("ix_trades_ticker_time", "ticker", "created_time"),
)

# Kalshi books are bid-only on each side: a YES ask at p is a NO bid at 1-p.
mkt_orderbook_snapshots = Table(
    "mkt_orderbook_snapshots", MARKET_MD,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String, nullable=False),
    Column("snapshot_ts", UTCDateTime, nullable=False),
    Column("side", String, nullable=False),  # "yes" | "no" (bids)
    Column("price", Float, nullable=False),
    Column("size", Float, nullable=False),
    UniqueConstraint("ticker", "snapshot_ts", "side", "price", name="uq_ob_level"),
    Index("ix_ob_ticker_ts", "ticker", "snapshot_ts"),
)

ext_observations = Table(
    "ext_observations", MARKET_MD,
    Column("source", String, primary_key=True),
    Column("series_id", String, primary_key=True),
    Column("obs_ts", UTCDateTime, primary_key=True),
    # When the value became publicly known. Point-in-time features must filter on this.
    Column("vintage_ts", UTCDateTime, primary_key=True),
    Column("value", Float),
    Column("units", String),
    Column("fetched_at", UTCDateTime, nullable=False),
)

# Append-only outcome history, so delayed or revised resolutions stay visible.
outcomes = Table(
    "outcomes", MARKET_MD,
    Column("ticker", String, primary_key=True),
    Column("recorded_at", UTCDateTime, primary_key=True),
    Column("result", String, nullable=False),
    Column("resolved_ts", UTCDateTime),
    Column("source", String, nullable=False),
    Column("revised", Boolean, nullable=False, default=False),
)

# Ground truth for synthetic markets only (lets tests check model recovery).
synthetic_truth = Table(
    "synthetic_truth", MARKET_MD,
    Column("ticker", String, primary_key=True),
    Column("true_p", Float, nullable=False),
    Column("seed", Integer, nullable=False),
)

source_health = Table(
    "source_health", MARKET_MD,
    Column("source", String, primary_key=True),
    Column("last_attempt", UTCDateTime),
    Column("last_success", UTCDateTime),
    Column("last_error", Text),
    Column("last_latency_ms", Float),
    Column("rows_last_run", Integer),
    Column("consecutive_failures", Integer, default=0),
)

data_quality_issues = Table(
    "data_quality_issues", MARKET_MD,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source", String, nullable=False),
    Column("entity", String, nullable=False),
    Column("entity_id", String, nullable=False),
    Column("check_name", String, nullable=False),
    Column("detail", Text),
    Column("detected_at", UTCDateTime, nullable=False),
    UniqueConstraint("source", "entity", "entity_id", "check_name", name="uq_dq_issue"),
)

# ---------------------------------------------------------------- simulation data

forecasts = Table(
    "forecasts", SIM_MD,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String, nullable=False, index=True),
    Column("model_version", String, nullable=False),
    Column("created_ts", UTCDateTime, nullable=False),
    Column("input_max_ts", UTCDateTime, nullable=False),
    Column("p", Float, nullable=False),
    Column("ci_low", Float),
    Column("ci_high", Float),
    Column("market_ref_p", Float),
    Column("dq_flags", JSON),
    Column("explanation", JSON),
    Column("prev_hash", String, nullable=False),
    Column("row_hash", String, nullable=False, unique=True),
)

model_registry = Table(
    "model_registry", SIM_MD,
    Column("model_version", String, primary_key=True),
    Column("model_name", String, nullable=False),
    Column("category", String),
    Column("params_hash", String, nullable=False),
    Column("params_json", JSON),
    Column("trained_ts", UTCDateTime, nullable=False),
    Column("train_window_end", UTCDateTime),
    Column("validation_json", JSON),
    Column("status", String, nullable=False),  # candidate | promoted | retired | rejected
    Column("status_ts", UTCDateTime),
)

sim_runs = Table(
    "sim_runs", SIM_MD,
    Column("run_id", String, primary_key=True),
    Column("created_ts", UTCDateTime, nullable=False),
    Column("kind", String, nullable=False),  # backtest | loop | stress
    Column("config_json", JSON),
    Column("model_version", String),
    Column("data_cutoff_ts", UTCDateTime),
    Column("holdout_hash", String),
    Column("summary_json", JSON),
)

sim_orders = Table(
    "sim_orders", SIM_MD,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String, nullable=False, index=True),
    Column("ts", UTCDateTime, nullable=False),
    Column("ticker", String, nullable=False),
    Column("side", String, nullable=False),  # yes | no
    Column("limit_price", Float, nullable=False),
    Column("qty_requested", Float, nullable=False),
    Column("qty_filled", Float, nullable=False, default=0.0),
    Column("status", String, nullable=False),
    Column("reason", Text),
)

sim_fills = Table(
    "sim_fills", SIM_MD,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("order_id", Integer, nullable=False, index=True),
    Column("run_id", String, nullable=False, index=True),
    Column("ts", UTCDateTime, nullable=False),
    Column("ticker", String, nullable=False),
    Column("side", String, nullable=False),
    Column("price", Float, nullable=False),
    Column("qty", Float, nullable=False),
    Column("fee", Float, nullable=False),
    Column("liquidity_uncertain", Boolean, nullable=False),
)

sim_positions = Table(
    "sim_positions", SIM_MD,
    Column("run_id", String, primary_key=True),
    Column("ticker", String, primary_key=True),
    Column("side", String, primary_key=True),
    Column("qty", Float, nullable=False),
    Column("cost_basis", Float, nullable=False),
    Column("fees_paid", Float, nullable=False),
    Column("settled", Boolean, nullable=False, default=False),
    Column("payout", Float),
)

sim_equity = Table(
    "sim_equity", SIM_MD,
    Column("run_id", String, primary_key=True),
    Column("ts", UTCDateTime, primary_key=True),
    Column("cash", Float, nullable=False),
    Column("exposure", Float, nullable=False),
    Column("equity", Float, nullable=False),
    Column("gross_pnl", Float, nullable=False),
    Column("net_pnl", Float, nullable=False),
    Column("fees_cum", Float, nullable=False),
)

research_log = Table(
    "research_log", SIM_MD,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", UTCDateTime, nullable=False, index=True),
    Column("component", String, nullable=False),
    Column("level", String, nullable=False),
    Column("message", Text, nullable=False),
    Column("payload", JSON),
)
