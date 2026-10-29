"""Kalshi trading fees, as published at https://kalshi.com/fee-schedule (read 2026-10-29).

    taker fee = round_up_to_cent(M * 0.07   * C * P * (1 - P))
    maker fee = round_up_to_cent(M * 0.0175 * C * P * (1 - P))   (only on series with maker fees)

C = contracts, P = price in dollars, M = series fee multiplier. Consistency check against
the published ranges for 100 contracts: taker $0.07 (P=0.01) to $1.75 (P=0.50), maker
$0.02 to $0.44. Some series charge no fees.

Assumptions (unverified): rounding is applied per fill (more conservative than per order),
and there is no settlement fee.
"""

import math
from dataclasses import dataclass

TAKER_COEF = 0.07
MAKER_COEF = 0.0175
SOURCE = "https://kalshi.com/fee-schedule (retrieved 2026-10-29)"


def round_up_cent(x: float) -> float:
    # round() first so float noise like 1.7500000001 does not add a cent
    return math.ceil(round(x * 100, 9)) / 100


@dataclass(frozen=True)
class FeeSchedule:
    multiplier: float = 1.0
    maker_fees: bool = False
    no_fees: bool = False
    #: stress-test knob: scales every fee (1.0 = published schedule)
    stress: float = 1.0

    def fee(self, contracts: float, price: float, role: str = "taker") -> float:
        if contracts <= 0 or self.no_fees:
            return 0.0
        if not 0.0 < price < 1.0:
            raise ValueError(f"price must be in (0, 1), got {price}")
        if role == "maker":
            if not self.maker_fees:
                return 0.0
            coef = MAKER_COEF
        elif role == "taker":
            coef = TAKER_COEF
        else:
            raise ValueError(f"role must be taker or maker, got {role!r}")
        return round_up_cent(self.stress * self.multiplier * coef * contracts * price * (1 - price))

    def per_contract(self, price: float) -> float:
        """Unrounded marginal taker cost per contract (used for edge estimates only)."""
        if self.no_fees:
            return 0.0
        return self.stress * self.multiplier * TAKER_COEF * price * (1 - price)

    @classmethod
    def from_series(cls, fee_type: str | None, fee_multiplier: float | None,
                    stress: float = 1.0) -> "FeeSchedule":
        """Map a series' fee_type/fee_multiplier to a schedule.

        Unknown fee types fall back to the standard taker schedule (conservative)."""
        ft = (fee_type or "quadratic").lower()
        mult = 1.0 if fee_multiplier is None else float(fee_multiplier)
        if ft in ("none", "no_fee", "no_fees", "free"):
            return cls(no_fees=True, stress=stress)
        return cls(multiplier=mult, maker_fees="maker" in ft, stress=stress)
