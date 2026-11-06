"""The single order-decision rule shared by the backtest-style paper loop and the
separate ``executor`` package: shrink toward the market, stay inside the entry band,
require edge after fees, size by fractional Kelly (optionally raised to a minimum order).
Pure arithmetic; places nothing."""

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


def maker_price(bid: float | None, ask: float) -> float:
    """Best resting buy price that can't cross: 1c above the bid if that's still below the
    ask, else join the bid; with no bid, 1c below the ask."""
    if bid is None or bid <= 0 or bid >= ask - 1e-9:
        return round(max(0.01, ask - 0.01), 2)
    return round(bid + 0.01, 2) if bid + 0.01 < ask - 1e-9 else round(bid, 2)


def plan_order(ticker: str, p_yes: float, quote: Quote, fees: FeeSchedule, *, equity: float,
               cash: float, min_edge: float, kelly_fraction: float,
               max_order_usd: float | None, shrink: float = 0.5,
               min_entry_price: float = 0.05, max_entry_price: float = 0.95,
               min_order_usd: float | None = None,
               maker: bool = False) -> tuple[Plan | None, str]:
    """``maker``: price a resting post-only buy 1c above the side's bid (joining the bid
    when the spread is 1c), so edge, size and the limit are all at that price, with maker
    fees. ``max_price`` is then the exact price to post and ``best_ask`` stays the ask."""
    p_yes = shrink_to_market(p_yes, quote_mid(quote), shrink)
    best, outside_band = None, False
    for outcome, q, asks, bid in (("yes", p_yes, quote.yes_asks, quote.yes_bid),
                                  ("no", 1 - p_yes, quote.no_asks, quote.no_bid)):
        if not asks:
            continue
        a = asks[0][0]
        price = maker_price(bid, a) if maker else a
        if not min_entry_price <= price <= max_entry_price:
            outside_band = True
            continue
        ev = q - price - fees.per_contract_maker(price) if maker else _ev(q, price, fees)
        if best is None or ev > best[2]:
            best = (outcome, q, ev, a, price)
    if best is None:
        return None, (f"price outside {min_entry_price:.2f}-{max_entry_price:.2f} entry band"
                      if outside_band else "no_asks")
    outcome, q, ev, ask, a = best
    if ev < min_edge:
        return None, f"edge {ev:.3f} < min_edge {min_edge}"
    unit = a + (fees.per_contract_maker(a) if maker else fees.per_contract(a) + 0.01)
    cap = min(cash, max_order_usd if max_order_usd is not None else float("inf"))
    count = math.floor(min(kelly_fraction * _kelly(q, a) * equity, cap) / unit)
    if min_order_usd is not None and count * unit < min_order_usd:
        # Owner rule: an order that clears the edge test is at least min_order_usd, still
        # inside cash and the per-order cap (and the executor's other caps after this).
        count = math.ceil(min_order_usd / unit - 1e-9)
        if count * unit > cap:
            return None, f"minimum order ${min_order_usd:g} exceeds cash or per-order cap"
    if count < 1:
        return None, "size below one contract"
    if maker:
        return Plan(ticker, outcome, q, ask, a, count, ev, round(count * unit, 2)), ""
    max_price = round(min(0.99, q - min_edge - fees.per_contract(a)), 2)
    if max_price < a:
        return None, "limit below best ask"
    return Plan(ticker, outcome, q, a, max_price, count, ev, round(count * unit, 2)), ""
