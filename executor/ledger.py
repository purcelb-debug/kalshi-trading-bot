"""Local ledger of the LIVE account, synced from Kalshi (the exchange is the truth).

``executor sync`` copies fills, settlements and a balance snapshot into
``executor/data/ledger.db`` (git-ignored). The dashboard reads only this file and the
audit log, so it never needs credentials.

Realized P&L per settled market = revenue - cost of contracts held - fee_cost. The
settlement's ``fee_cost`` is the total trading fee on that market (it equals the sum of
the fills' fee_cost; checked against two real settlements on 2026-10-30), so fill fees are
not subtracted again. Kalshi reports ``revenue`` in cents and costs and fees in dollars
(docs.kalshi.com, Get Settlements / Get Fills, read 2026-10-30).

Bot vs other activity: a fill is the bot's when its order_id appears in an ``order_ack``
of the live audit log; everything else (manual trades, other apps) is "other".
"""

import json
from pathlib import Path

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    MetaData,
    String,
    Table,
    create_engine,
    select,
)
from sqlalchemy.dialects.sqlite import insert

from predlab.timeutil import to_utc, utcnow

MD = MetaData()

fills = Table(
    "fills", MD,
    Column("fill_id", String, primary_key=True),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("ticker", String, nullable=False),
    Column("order_id", String),
    Column("outcome", String),          # yes | no: the contract bought or sold
    Column("action", String),           # buy | sell
    Column("count", Float, nullable=False),
    Column("price", Float, nullable=False),   # $ per contract of ``outcome``
    Column("fee", Float, nullable=False),
    Column("is_taker", String),
    Column("raw", JSON),
)

settlements = Table(
    "settlements", MD,
    Column("ticker", String, primary_key=True),
    Column("settled_ts", DateTime(timezone=True), primary_key=True),
    Column("event_ticker", String),
    Column("result", String),
    Column("yes_count", Float), Column("no_count", Float),
    Column("cost", Float, nullable=False),
    Column("revenue", Float, nullable=False),
    Column("settle_fee", Float, nullable=False),
    Column("raw", JSON),
)

balances = Table(
    "balances", MD,
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("cash", Float, nullable=False),
    Column("positions_value", Float, nullable=False),
)


def ledger_engine(data_dir: str):
    path = Path(data_dir) / "ledger.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(f"sqlite:///{path}")
    MD.create_all(eng)
    return eng


def _f(d: dict, *keys, default=0.0) -> float:
    for k in keys:
        if d.get(k) not in (None, ""):
            return float(d[k])
    return default


def norm_fill(f: dict) -> dict:
    outcome = f.get("outcome_side") or f.get("side")
    price = _f(f, "yes_price_dollars") if outcome == "yes" else _f(f, "no_price_dollars")
    return {"fill_id": f["fill_id"], "ts": to_utc(f["created_time"]),
            "ticker": f.get("market_ticker") or f["ticker"], "order_id": f.get("order_id"),
            "outcome": outcome, "action": f.get("action"), "count": _f(f, "count_fp", "count"),
            "price": price, "fee": _f(f, "fee_cost"), "is_taker": str(f.get("is_taker")),
            "raw": f}


def norm_settlement(s: dict) -> dict:
    return {"ticker": s["ticker"], "settled_ts": to_utc(s["settled_time"]),
            "event_ticker": s.get("event_ticker"), "result": s.get("market_result"),
            "yes_count": _f(s, "yes_count_fp"), "no_count": _f(s, "no_count_fp"),
            "cost": _f(s, "yes_total_cost_dollars") + _f(s, "no_total_cost_dollars"),
            "revenue": _f(s, "revenue") / 100.0, "settle_fee": _f(s, "fee_cost"), "raw": s}


def _upsert(eng, table, rows):
    if not rows:
        return 0
    with eng.begin() as c:
        for r in rows:
            r = {**r, "raw": json.loads(json.dumps(r.get("raw"), default=str))} \
                if "raw" in r else r
            c.execute(insert(table).values(**r).on_conflict_do_nothing())
    return len(rows)


def sync(client, data_dir: str) -> dict:
    """Pull everything since the newest stored record (with a one-day overlap)."""
    eng = ledger_engine(data_dir)
    with eng.connect() as c:
        last_fill = c.execute(select(fills.c.ts).order_by(fills.c.ts.desc())).scalar()
        last_settle = c.execute(select(settlements.c.settled_ts)
                                .order_by(settlements.c.settled_ts.desc())).scalar()

    def since(ts):
        return None if ts is None else int(to_utc(ts).timestamp()) - 86400

    nf = _upsert(eng, fills, [norm_fill(f) for f in client.fills(since(last_fill))])
    ns = _upsert(eng, settlements,
                 [norm_settlement(s) for s in client.settlements(since(last_settle))])
    cash, pv = client.balance_usd()
    _upsert(eng, balances, [{"ts": utcnow(), "cash": cash, "positions_value": pv}])
    eng.dispose()
    return {"fills_seen": nf, "settlements_seen": ns, "cash": cash, "positions_value": pv}


def load(data_dir: str):
    """(fills, settlements with realized P&L, balance history) as DataFrames."""
    import pandas as pd
    eng = ledger_engine(data_dir)
    with eng.connect() as c:
        f = pd.DataFrame(c.execute(select(fills).order_by(fills.c.ts)).mappings().all(),
                         columns=[col.name for col in fills.columns])
        s = pd.DataFrame(c.execute(select(settlements).order_by(settlements.c.settled_ts))
                         .mappings().all(), columns=[col.name for col in settlements.columns])
        b = pd.DataFrame(c.execute(select(balances).order_by(balances.c.ts)).mappings().all(),
                         columns=[col.name for col in balances.columns])
    eng.dispose()
    for df, col in ((f, "ts"), (s, "settled_ts"), (b, "ts")):
        df[col] = pd.to_datetime(df[col], utc=True)
    bot_orders = _bot_order_ids(data_dir)
    f["source"] = ["bot" if o in bot_orders else "other" for o in f["order_id"]]
    bot_tickers = set(f.loc[f["source"] == "bot", "ticker"])
    s["source"] = ["bot" if t in bot_tickers else "other" for t in s["ticker"]]
    s["realized_pnl"] = s["revenue"] - s["cost"] - s["settle_fee"]
    s["cum_realized_pnl"] = s["realized_pnl"].cumsum()
    b["equity"] = b["cash"] + b["positions_value"]
    return f.drop(columns="raw"), s.drop(columns="raw"), b


def _bot_order_ids(data_dir: str) -> set[str]:
    path = Path(data_dir) / "audit_live.jsonl"
    ids = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e.get("event") == "order_ack" and e.get("order_id"):
                ids.add(e["order_id"])
    return ids
