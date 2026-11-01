"""Go-live gate: a model version may trade real money only if ALL of these hold.

1. Untouched holdout: its Brier score beat the market's on the same holdout, with the
   paired 95% CI of (model - market) entirely below 0.
2. Holdout paper backtest: the 95% CI of net P&L after fees has a lower bound > 0.
3. Paper record: at least ``gate_paper_weeks`` of forward paper trading in the research
   loop, with zero degradation alerts in that period.
4. Paper trade count: at least ``gate_min_paper_trades`` settled paper trades, so a quiet
   month with a handful of trades cannot open the gate.

Evidence is read from predlab's ``model_registry.validation_json`` under the key
``"gate"`` (written by the research loop). Missing evidence = gate closed.
"""

from dataclasses import dataclass

from sqlalchemy import Engine, select

from predlab.db.schema import model_registry


@dataclass(frozen=True)
class GateResult:
    open: bool
    model_version: str
    reasons: tuple[str, ...]


def evaluate_gate(model_version: str, evidence: dict | None, paper_weeks_required: int,
                  min_paper_trades: int = 50) -> GateResult:
    if not evidence:
        return GateResult(False, model_version, ("no gate evidence recorded for this model",))
    reasons = []
    hi = evidence.get("holdout_brier_diff_vs_market_ci_high")
    if hi is None or not hi < 0:
        reasons.append(f"holdout: not shown better than market (CI high = {hi})")
    lo = evidence.get("holdout_net_pnl_ci_low")
    if lo is None or not lo > 0:
        reasons.append(f"holdout backtest: net P&L CI lower bound not > 0 (= {lo})")
    days = evidence.get("paper_days") or 0
    if days < 7 * paper_weeks_required:
        reasons.append(f"paper record: {days} days < {7 * paper_weeks_required} required")
    trades = evidence.get("paper_settled_trades")
    if trades is None or trades < min_paper_trades:
        reasons.append(f"paper record: {trades} settled trades < {min_paper_trades} required")
    alerts = evidence.get("paper_degradation_alerts")
    if alerts is None or alerts > 0:
        reasons.append(f"paper record: degradation alerts = {alerts}")
    return GateResult(not reasons, model_version, tuple(reasons))


def load_gate(sim_engine: Engine, model_version: str, paper_weeks_required: int,
              min_paper_trades: int = 50) -> GateResult:
    with sim_engine.connect() as c:
        row = c.execute(select(model_registry.c.validation_json, model_registry.c.status)
                        .where(model_registry.c.model_version == model_version)).first()
    if row is None:
        return GateResult(False, model_version, ("model version not in registry",))
    if row.status != "promoted":
        return GateResult(False, model_version, ((f"registry status is {row.status!r}, "
                                                 "not 'promoted'"),))
    return evaluate_gate(model_version, (row.validation_json or {}).get("gate"),
                         paper_weeks_required, min_paper_trades)


def certified_shrink(sim_engine: Engine, model_version: str) -> float | None:
    """The shrink factor kappa fixed when this deployment was certified (None if absent)."""
    with sim_engine.connect() as c:
        v = c.execute(select(model_registry.c.validation_json)
                      .where(model_registry.c.model_version == model_version)).scalar()
    k = ((v or {}).get("gate") or {}).get("shrink_kappa")
    return None if k is None else float(k)
