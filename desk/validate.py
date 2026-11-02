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


def parse_trades(lead_text: str, allowed: dict[str, str], *, max_buys: int,
                 p_floor: float, exit_gap: float = 0.03) -> tuple[list[Decision], list[str]]:
    """(BUY decisions to record, notes on everything dropped). ``allowed`` maps each ticker
    shown to the desk to its event. Malformed output means no trades."""
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
        tp, st, why = _exits(r, exit_gap)
        if why:
            notes.append(why)
        out.append(Decision(r.ticker, p, r.side.lower(), r.reason[:300], tp, st))
    return out, notes
