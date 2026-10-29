"""Simulated cash, positions, fees and settlement. PAPER TRADING, NOT REAL MONEY.

Accounting
* Buying C contracts at price P costs C*P plus the fee; settlement pays $1 per winning
  contract and $0 per losing contract.
* gross P&L = payouts - contract cost; net P&L = gross - fees.
* Open positions are marked at the *bid* observed at entry (what we could have sold for
  then), not at cost or mid, so the equity curve shows the spread as an immediate loss.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Position:
    ticker: str
    side: str
    qty: float = 0.0
    cost: float = 0.0
    fees: float = 0.0
    mark: float = 0.0            # liquidation value per contract used for equity
    cluster: str = ""
    category: str = ""
    settled: bool = False
    payout: float = 0.0


@dataclass
class Portfolio:
    cash: float
    initial: float = 0.0
    positions: dict[tuple[str, str], Position] = field(default_factory=dict)
    gross_realized: float = 0.0
    fees_paid: float = 0.0
    equity_log: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.initial = self.initial or self.cash

    @property
    def open_positions(self):
        return [p for p in self.positions.values() if not p.settled and p.qty > 0]

    @property
    def exposure(self) -> float:
        """Capital at risk in open positions (cost basis)."""
        return sum(p.cost for p in self.open_positions)

    def market_exposure(self, ticker: str) -> float:
        return sum(p.cost for p in self.open_positions if p.ticker == ticker)

    @property
    def equity(self) -> float:
        return self.cash + sum(p.qty * p.mark for p in self.open_positions)

    def buy(self, ticker, side, qty, price, fee, mark, cluster="", category="") -> None:
        total = qty * price + fee
        if total > self.cash + 1e-9:
            raise ValueError("insufficient simulated cash")  # caller sizes before buying
        pos = self.positions.setdefault((ticker, side), Position(ticker, side, cluster=cluster,
                                                                 category=category))
        pos.qty += qty
        pos.cost += qty * price
        pos.fees += fee
        pos.mark = mark if mark is not None else 0.0
        self.cash -= total
        self.fees_paid += fee

    def settle(self, ticker: str, result: str) -> float:
        """Pay out every open position on ``ticker``. Returns gross P&L realized."""
        gross = 0.0
        for (t, side), pos in self.positions.items():
            if t != ticker or pos.settled:
                continue
            pos.payout = pos.qty * (1.0 if side == result else 0.0)
            pos.settled = True
            self.cash += pos.payout
            gross += pos.payout - pos.cost
        self.gross_realized += gross
        return gross

    def snapshot(self, ts) -> dict:
        row = {"ts": ts, "cash": self.cash, "exposure": self.exposure, "equity": self.equity,
               "gross_pnl": self.gross_realized, "fees_cum": self.fees_paid,
               "net_pnl": self.equity - self.initial}
        self.equity_log.append(row)
        return row

    def equity_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_log)
