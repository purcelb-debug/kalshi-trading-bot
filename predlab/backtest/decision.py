"""The single order-decision rule shared by the backtest-style paper loop and the
separate ``executor`` package: shrink toward the market, stay inside the entry band,
require edge after fees, size by fractional Kelly. Pure arithmetic; places nothing."""

import math
from dataclasses import dataclass

from predlab.backtest.engine import _ev, _kelly, quote_mid, shrink_to_market
from predlab.backtest.execution import Quote
from predlab.backtest.fees import FeeSchedule


@dataclass
class Plan:
    ticker: str
    outcome: str
    p_side: float
    best_ask: float
    max_price: float
    count: int
    ev: float
    cost_usd: float


def plan_order(ticker: str, p_yes: float, quote: Quote, fees: FeeSchedule, *, equity: float,
               cash: float, min_edge: float, kelly_fraction: float,
               max_order_usd: float | None, shrink: float = 0.5,
               min_entry_price: float = 0.05, max_entry_price: float = 0.95
               ) -> tuple[Plan | None, str]:
    p_yes = shrink_to_market(p_yes, quote_mid(quote), shrink)
    best, outside_band = None, False
    for outcome, q, asks in (("yes", p_yes, quote.yes_asks), ("no", 1 - p_yes, quote.no_asks)):
        if not asks:
            continue
        a = asks[0][0]
        if not min_entry_price <= a <= max_entry_price:
            outside_band = True
            continue
        ev = _ev(q, a, fees)
        if best is None or ev > best[2]:
            best = (outcome, q, ev, a)
    if best is None:
        return None, (f"price outside {min_entry_price:.2f}-{max_entry_price:.2f} entry band"
                      if outside_band else "no_asks")
    outcome, q, ev, a = best
    if ev < min_edge:
        return None, f"edge {ev:.3f} < min_edge {min_edge}"
    unit = a + fees.per_contract(a) + 0.01
    budget = min(kelly_fraction * _kelly(q, a) * equity, cash,
                 max_order_usd if max_order_usd is not None else float("inf"))
    count = math.floor(budget / unit)
    if count < 1:
        return None, "size below one contract"
    max_price = round(min(0.99, q - min_edge - fees.per_contract(a)), 2)
    if max_price < a:
        return None, "limit below best ask"
    return Plan(ticker, outcome, q, a, max_price, count, ev, round(count * unit, 2)), ""
