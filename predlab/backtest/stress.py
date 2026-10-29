"""Stress tests: rerun a backtest under unfavorable assumptions and report the damage.

Each scenario changes one thing relative to the base run:
  forecast_noise       logit-noise on every forecast (forecast uncertainty)
  adverse_price_move   asks move against us by N cents before the order (sudden market change)
  stale_quotes         decisions priced off quotes N hours old
  data_outage          a share of quotes is unavailable
  api_failures         a share of order attempts fails
  forecast_outage      a share of forecasts never arrives
  delayed_resolution   settlement N days later (capital locked longer)
  wrong_resolution     a share of outcomes resolves the other way
  fees_x2 / fees_x3    fee schedule multiplied
  thin_liquidity       depth cap / participation halved
  correlated_by_day    P&L CI re-clustered by settlement day instead of event
"""

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd

from predlab.backtest.engine import StrategyConfig, run_backtest
from predlab.backtest.execution import ExecConfig, Quote
from predlab.backtest.fees import FeeSchedule


class StressedMarket:
    """Wraps MarketData; same ``quote`` interface, with injected faults."""

    def __init__(self, market, *, adverse_shift: float = 0.0, stale_lag: timedelta | None = None,
                 outage_rate: float = 0.0, api_failure_rate: float = 0.0, seed: int = 0):
        self.market, self.shift, self.lag = market, adverse_shift, stale_lag
        self.outage, self.fail = outage_rate, api_failure_rate
        self.rng = np.random.default_rng(seed)

    def decision_quote(self, ticker, ts, cfg):
        """What the strategy sees: possibly ``stale_lag`` old, never shifted."""
        if self.outage and self.rng.random() < self.outage:
            return None, "data_outage"
        if self.lag:
            return self.market.quote(ticker, ts - self.lag,
                                     replace(cfg, max_quote_age=cfg.max_quote_age + self.lag))
        return self.market.quote(ticker, ts, cfg)

    def quote(self, ticker, ts, cfg):
        """What the order actually trades against: current book, possibly moved against
        us by ``adverse_shift``; order submission can fail with ``api_failure_rate``."""
        if self.fail and self.rng.random() < self.fail:
            return None, "api_failure"
        q, why = self.market.quote(ticker, ts, cfg)
        if q is None or not self.shift:
            return q, why

        def up(levels):
            return [(min(0.99, round(p + self.shift, 4)), s) for p, s in levels]

        return Quote(q.ts, q.source, up(q.yes_asks), up(q.no_asks),
                     None if q.yes_bid is None else max(0.0, q.yes_bid - self.shift),
                     None if q.no_bid is None else max(0.0, q.no_bid - self.shift)), why


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _day_ci(res, resolutions, n_boot=2000, seed=0):
    pos = res.positions
    if pos.empty or not pos["settled"].any():
        return None
    s = pos[pos["settled"]].copy()
    day = pd.to_datetime(resolutions.set_index("ticker").loc[s["ticker"], "resolve_ts"],
                         utc=True).dt.date.to_numpy()
    s["net"] = s["payout"] - s["cost"] - s["fees"]
    by = s.assign(day=day).groupby("day")["net"].sum().to_numpy()
    if len(by) < 10:
        return None
    rng = np.random.default_rng(seed)
    boots = [by[rng.integers(0, len(by), len(by))].sum() for _ in range(n_boot)]
    return [round(float(np.quantile(boots, 0.025)), 4), round(float(np.quantile(boots, 0.975)), 4)]


def run_stress_suite(signals: pd.DataFrame, resolutions: pd.DataFrame, market, *,
                     strategy: StrategyConfig | None = None, execution: ExecConfig | None = None,
                     fee_for: dict | None = None, default_fees: FeeSchedule | None = None,
                     seed: int = 0, fee_for_stress=None) -> pd.DataFrame:
    """``fee_for_stress(multiplier) -> fee_for dict`` lets per-series schedules be stressed."""
    strategy = strategy or StrategyConfig()
    execution = execution or ExecConfig()
    default_fees = default_fees or FeeSchedule()
    rng = np.random.default_rng(seed)

    def bt(sig=signals, res=resolutions, md=market, ex=execution, ff=fee_for, df=default_fees):
        return run_backtest(sig, res, md, strategy=strategy, execution=ex, fee_for=ff,
                            default_fees=df)

    def fees_x(m):
        return {"ff": fee_for_stress(m) if fee_for_stress else None,
                    "df": FeeSchedule(stress=m)}

    flip = rng.random(len(resolutions)) < 0.05
    keep = rng.random(len(signals)) >= 0.2
    scenarios = {
        "base": {},
        "forecast_noise_0.5": {"sig": signals.assign(
            p=_sigmoid(_logit(signals["p"]) + rng.normal(0, 0.5, len(signals))))},
        "adverse_price_move_2c": {"md": StressedMarket(market, adverse_shift=0.02, seed=seed)},
        "stale_quotes_3h": {"md": StressedMarket(market, stale_lag=timedelta(hours=3), seed=seed)},
        "data_outage_20pct": {"md": StressedMarket(market, outage_rate=0.2, seed=seed)},
        "api_failures_10pct": {"md": StressedMarket(market, api_failure_rate=0.1, seed=seed)},
        "forecast_outage_20pct": {"sig": signals[keep].reset_index(drop=True)},
        "delayed_resolution_7d": {"res": resolutions.assign(
            resolve_ts=pd.to_datetime(resolutions["resolve_ts"], utc=True) + timedelta(days=7))},
        "wrong_resolution_5pct": {"res": resolutions.assign(result=np.where(
            flip, np.where(resolutions["result"] == "yes", "no", "yes"), resolutions["result"]))},
        "fees_x2": fees_x(2.0),
        "fees_x3": fees_x(3.0),
        "thin_liquidity": {"ex": replace(execution, participation=execution.participation / 2,
                                          unknown_depth_cap=max(1.0, execution.unknown_depth_cap / 2))},
    }
    rows, base = [], None
    for name, kw in scenarios.items():
        r = bt(**kw)
        s = r.summary
        if name == "base":
            base = s["net_pnl"]
            day_ci = _day_ci(r, resolutions)
        rows.append({"scenario": name, "net_pnl": s["net_pnl"], "net_pnl_ci95": s["net_pnl_ci95"],
                     "return": s["return"], "max_drawdown": s["max_drawdown"],
                     "n_fills": s["n_fills"], "fees": s["fees"],
                     "delta_vs_base": None if base is None else round(s["net_pnl"] - base, 4)})
    rows.append({"scenario": "correlated_by_day (CI only)", "net_pnl": base,
                 "net_pnl_ci95": day_ci, "return": None, "max_drawdown": None, "n_fills": None,
                 "fees": None, "delta_vs_base": 0.0})
    return pd.DataFrame(rows)
