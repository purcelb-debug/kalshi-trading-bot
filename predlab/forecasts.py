"""Append-only, hash-chained forecast log.

Each row's hash covers its content and the previous row's hash, so editing or deleting
any past forecast breaks every later hash and ``verify_chain`` reports it.
"""

import hashlib
import json

from sqlalchemy import Engine, select

from predlab.db.schema import forecasts
from predlab.models.base import Forecast
from predlab.timeutil import to_utc, utcnow

GENESIS = "0" * 64


def _content(row: dict) -> str:
    keys = ("ticker", "model_version", "created_ts", "input_max_ts", "p", "ci_low", "ci_high",
            "market_ref_p", "dq_flags", "explanation")
    norm = {k: (to_utc(row[k]).isoformat() if k.endswith("_ts") else row[k]) for k in keys}
    return json.dumps(norm, sort_keys=True, default=str, separators=(",", ":"))


def _hash(prev: str, row: dict) -> str:
    return hashlib.sha256((prev + _content(row)).encode()).hexdigest()


def _last_hash(conn) -> str:
    h = conn.execute(select(forecasts.c.row_hash).order_by(forecasts.c.id.desc()).limit(1)).scalar()
    return h or GENESIS


def record_forecasts(engine: Engine, items: list[Forecast],
                     market_ref: dict[str, float] | None = None) -> int:
    """Append forecasts in one transaction. ``created_ts`` is the wall-clock write time."""
    market_ref = market_ref or {}
    with engine.begin() as conn:
        prev = _last_hash(conn)
        now = utcnow()
        for f in items:
            if to_utc(f.input_max_ts) > now:
                raise ValueError(f"forecast for {f.ticker} uses inputs from the future")
            row = {"ticker": f.ticker, "model_version": f.model_version, "created_ts": now,
                   "input_max_ts": to_utc(f.input_max_ts), "p": round(f.p, 6),
                   "ci_low": None if f.ci_low is None else round(f.ci_low, 6),
                   "ci_high": None if f.ci_high is None else round(f.ci_high, 6),
                   "market_ref_p": market_ref.get(f.ticker), "dq_flags": list(f.dq_flags),
                   "explanation": f.explanation}
            row["prev_hash"], row["row_hash"] = prev, _hash(prev, row)
            conn.execute(forecasts.insert().values(**row))
            prev = row["row_hash"]
    return len(items)


def verify_chain(engine: Engine) -> list[int]:
    """Return ids of rows whose hash or back-link does not verify (empty = intact)."""
    bad, prev = [], GENESIS
    with engine.connect() as conn:
        for r in conn.execute(select(forecasts).order_by(forecasts.c.id)).mappings():
            row = dict(r)
            if row["prev_hash"] != prev or _hash(prev, row) != row["row_hash"]:
                bad.append(row["id"])
            prev = row["row_hash"]
    return bad
