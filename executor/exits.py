"""Exit rules for positions a desk deployment opened: sell before settlement when the bid
for the side held reaches the take-profit, or falls to the stop.

Levels come from the forecast that opened the position (``explanation.take_profit`` /
``explanation.stop``, prices of the side bought). Positions without levels, or held on the
other side from the one the desk named, are left to settle. Exits only ever close: the
order is ``reduce_only`` and sized to the contracts held.
"""

import json
import math
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import Engine, func, select

from executor.client import OrderAck
from predlab.backtest.execution import Quote
from predlab.db.schema import forecasts


@dataclass(frozen=True)
class ExitPlan:
    ticker: str
    held: str
    count: int
    min_price: float
    reason: str              # take_profit | stop
    bid: float
    level: float


def exit_levels(sim_engine: Engine, model_version: str) -> dict[str, dict]:
    """Latest {side, take_profit, stop} per ticker for this deployment, where set."""
    sub = (select(forecasts.c.ticker, func.max(forecasts.c.id).label("id"))
           .where(forecasts.c.model_version == model_version).group_by(forecasts.c.ticker)
           .subquery())
    with sim_engine.connect() as c:
        rows = c.execute(select(forecasts.c.ticker, forecasts.c.explanation)
                         .join(sub, sub.c.id == forecasts.c.id)).all()
    out = {}
    for ticker, raw in rows:
        e = json.loads(raw) if isinstance(raw, str) else (raw or {})
        if e.get("take_profit") is not None and e.get("stop") is not None:
            out[ticker] = {"side": (e.get("side") or "").lower(),
                           "take_profit": float(e["take_profit"]), "stop": float(e["stop"])}
    return out


def plan_exit(position: dict, levels: dict | None, quote: Quote | None,
              allow_stop: bool = True) -> ExitPlan | None:
    """Pure rule: one held position, its levels and a fresh book -> an exit or nothing.
    ``allow_stop`` False (market closes soon) keeps the take-profit and holds through
    dips: most stops on short-dated markets fired on ordinary noise and paid fees twice."""
    if not levels or quote is None:
        return None
    held = position["side"]
    if levels["side"] and levels["side"] != held:
        return None                          # bought the other side: levels don't apply
    bid = quote.yes_bid if held == "yes" else quote.no_bid
    count = math.floor(position["qty"] + 1e-9)
    if bid is None or bid <= 0 or count < 1:
        return None
    if bid >= levels["take_profit"]:
        return ExitPlan(position["ticker"], held, count, round(bid, 2), "take_profit", bid,
                        levels["take_profit"])
    if allow_stop and bid <= levels["stop"]:
        return ExitPlan(position["ticker"], held, count, max(0.01, round(bid, 2)), "stop", bid,
                        levels["stop"])
    return None


def ack_dict(ack: OrderAck) -> dict:
    return {k: (None if isinstance(v, float) and pd.isna(v) else v)
            for k, v in ack.__dict__.items()}
