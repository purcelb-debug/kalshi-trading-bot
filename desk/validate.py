"""Parse the Desk Lead's TRADES list and enforce the desk's structural rules in code.

Everything the model wrote, including what it read on the web, is untrusted. The most a
hostile or confused transcript can do here is change a probability for a market that was
already on the table; sizes and money limits are applied later by the backtest engine
and the executor, which never see the transcript.
"""

import json
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError, field_validator


class _Row(BaseModel):
    ticker: str
    action: str
    p_yes: float | None = None
    side: str = ""
    reason: str = ""
    take_profit: float | None = None     # price of the side bought, to sell at or above
    stop: float | None = None            # price of the side bought, to sell at or below

    @field_validator("action")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


@dataclass(frozen=True)
class Decision:
    ticker: str
    p_yes: float
    side: str
    reason: str
    take_profit: float | None = None
    stop: float | None = None


def _exits(r: _Row, gap: float) -> tuple[float | None, float | None, str]:
    """Clip exit levels to tradable prices; drop both unless stop + gap <= take_profit."""
    if r.take_profit is None or r.stop is None:
        return None, None, ""
    tp, st = min(max(r.take_profit, 0.02), 0.99), min(max(r.stop, 0.01), 0.98)
    if st + gap > tp:
        return None, None, f"{r.ticker}: exit levels {r.stop}/{r.take_profit} unusable, held to settlement"
    return round(tp, 2), round(st, 2), ""


def book_exits(r: _Row, p: float, book: tuple[float, float], gap: float
               ) -> tuple[str, float, float, str]:
    """(side, take_profit, stop, note) from the live book, in the price of the side the
    executor will actually buy (YES if p > mid, else NO). The desk's own levels are kept
    when they bracket the entry by at least spread + gap; levels written in the other
    side's price are flipped; otherwise code sets them: take profit after about two thirds
    of the expected edge, stop once the price has moved the whole edge against us."""
    bid, ask = book
    side = "yes" if p > (bid + ask) / 2 else "no"
    entry, p_side = (ask, p) if side == "yes" else (1 - bid, 1 - p)
    room = (ask - bid) + gap
    ok = lambda tp, st: tp is not None and st is not None and tp >= entry + room \
        and st <= entry - room and 0.01 <= st < tp <= 0.99
    tp, st, note = r.take_profit, r.stop, ""
    if not ok(tp, st):
        flipped = (1 - tp, 1 - st) if tp is not None and st is not None else None
        if flipped and ok(*flipped):
            tp, st = flipped
            note = f"{r.ticker}: exit levels were in the other side's price, flipped"
        else:
            edge = max(p_side - entry, 0.0)
            tp = min(0.99, entry + max(room, edge * 2 / 3))
            st = max(0.01, entry - max(room, edge))
            note = f"{r.ticker}: exit levels set by code (desk gave {r.take_profit}/{r.stop})"
    return side, round(tp, 2), round(st, 2), note


def parse_trades(lead_text: str, allowed: dict[str, str], *, max_buys: int,
                 p_floor: float, exit_gap: float = 0.03,
                 books: dict[str, tuple[float, float]] | None = None
                 ) -> tuple[list[Decision], list[str]]:
    """(BUY decisions to record, notes on everything dropped). ``allowed`` maps each ticker
    shown to the desk to its event. ``books`` (ticker -> YES bid, ask) lets code check
    or set every exit level. Malformed output means no trades."""
    try:
        tail = lead_text[lead_text.rindex("TRADES:") + len("TRADES:"):]
        rows = json.loads(tail[tail.index("["): tail.rindex("]") + 1])
    except ValueError as e:
        return [], [f"no valid TRADES list ({e}); nothing recorded"]
    if not isinstance(rows, list):
        return [], ["TRADES is not a list; nothing recorded"]
    out, notes, events = [], [], set()
    for raw in rows:
        try:
            r = _Row.model_validate(raw)
        except ValidationError:
            notes.append(f"malformed row dropped: {str(raw)[:120]}")
            continue
        if r.action != "BUY":
            continue
        if r.ticker not in allowed:
            notes.append(f"{r.ticker}: not on the table, dropped")
            continue
        if r.p_yes is None or not 0.0 <= r.p_yes <= 1.0:
            notes.append(f"{r.ticker}: p_yes missing or outside [0, 1], dropped")
            continue
        if allowed[r.ticker] in events:
            notes.append(f"{r.ticker}: second BUY on event {allowed[r.ticker]}, dropped")
            continue
        if len(out) >= max_buys:
            notes.append(f"{r.ticker}: over {max_buys} BUYs this meeting, dropped")
            continue
        events.add(allowed[r.ticker])
        p = min(max(r.p_yes, p_floor), 1.0 - p_floor)
        side = r.side.lower()
        if books and r.ticker in books:
            side, tp, st, why = book_exits(r, p, books[r.ticker], exit_gap)
        else:
            tp, st, why = _exits(r, exit_gap)
        if why:
            notes.append(why)
        out.append(Decision(r.ticker, p, side, r.reason[:300], tp, st))
    return out, notes
