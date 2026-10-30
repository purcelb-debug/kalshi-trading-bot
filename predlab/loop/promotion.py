"""Certification: one look at the untouched holdout, then promote or reject.

A *deployment* is (model version, series, decision horizon); its id is what forecasts,
the paper record and the executor's gate all key on, so evidence from one series can
never unlock another. Each deployment is certified at most once, ever: holdout hashes
change as new markets settle, so re-certifying on a later, mostly overlapping holdout
would be a second look.

Multiple testing: the k-th deployment certified for the same series and horizon has to
clear its confidence intervals at level 1 - 0.05 / k (Bonferroni), so trying many
models until one passes by luck does not work.

Promotion requires, on the holdout:
  * Brier(model) - Brier(market) CI entirely below 0, with >= 95% market-price coverage;
  * holdout paper-trading net P&L CI entirely above 0 (forecasts shrunk by kappa, which is
    learned only from development walk-forward predictions);
  * a lower holdout Brier than the currently promoted deployment, if any (which is retired).
Promotion only admits a deployment to the main paper simulation. Real money additionally
needs the forward paper record (see executor/gate.py).
"""

from dataclasses import asdict, dataclass, replace
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import Engine, select, update

from predlab.backtest.engine import (
    StrategyConfig,
    fee_schedules,
    run_backtest,
    save_run,
)
from predlab.backtest.execution import MarketData
from predlab.backtest.shrinkage import fit_kappa
from predlab.db.schema import model_registry
from predlab.evaluation.metrics import brier, paired_bootstrap_diff, summarize
from predlab.evaluation.splits import HoldoutAlreadyUsed, HoldoutGuard
from predlab.evaluation.walkforward import run_walk_forward
from predlab.features.pointintime import build_dataset
from predlab.models.registry import ALL_MODELS
from predlab.research_log import log_event
from predlab.timeutil import utcnow

MIN_COVERAGE = 0.95
BASE_ALPHA = 0.05


def deployment_version(model_version: str, series: str, horizon_h: float) -> str:
    return f"{model_version}@{series}-h{horizon_h:g}"


@dataclass
class Certification:
    deployment: str
    status: str                 # promoted | rejected
    reasons: list[str]
    evidence: dict


def _cluster_sum_ci(values: np.ndarray, clusters: np.ndarray, level: float,
                    n_boot: int = 2000, seed: int = 0) -> list[float] | None:
    """Cluster-bootstrap CI of a total (e.g. net P&L by event). None if < 10 clusters."""
    if len(values) == 0 or len(np.unique(clusters)) < 10:
        return None
    by = pd.Series(values).groupby(clusters).sum().to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.array([by[rng.integers(0, len(by), len(by))].sum() for _ in range(n_boot)])
    a = (1 - level) / 2
    return [round(float(np.quantile(boots, a)), 4), round(float(np.quantile(boots, 1 - a)), 4)]


def _prior_certifications(sim: Engine, series: str, horizon_h: float) -> list:
    with sim.connect() as c:
        rows = c.execute(select(model_registry)).mappings().all()
    return [r for r in rows if (r["params_json"] or {}).get("series") == series
            and (r["params_json"] or {}).get("horizon_h") == horizon_h]


def certify(market_engine: Engine, sim_engine: Engine, *, model: str, series: str,
            horizon_h: float = 30.0, folds: int = 5, holdout_frac: float = 0.2,
            min_train: int = 50, embargo_h: float = 0.0,
            strategy: StrategyConfig | None = None, n_boot: int = 1000,
            as_of=None) -> Certification:
    if model not in ALL_MODELS:
        raise ValueError(f"unknown model {model!r}")
    make = ALL_MODELS[model]
    probe = make()
    dv = deployment_version(probe.version, series, horizon_h)
    rows = _prior_certifications(sim_engine, series, horizon_h)
    existing = next((r for r in rows if r["model_version"] == dv), None)
    if existing is not None and existing["status"] != "candidate":
        raise HoldoutAlreadyUsed(f"{dv} was already certified; a deployment gets one look")
    prior = [r for r in rows if r["status"] != "candidate"]   # holdout looks so far
    k = len(prior) + 1
    level = 1 - BASE_ALPHA / k

    embargo = timedelta(hours=embargo_h)
    ds = build_dataset(market_engine, series=series, horizon=timedelta(hours=horizon_h),
                       as_of=as_of)
    factories = {"market_implied": ALL_MODELS["market_implied"], model: make}
    wf = run_walk_forward(ds, factories, n_folds=folds, holdout_frac=holdout_frac,
                          min_train=min_train, embargo=embargo, n_boot=200)
    if model in wf.errors:
        raise RuntimeError(f"{model} failed in walk-forward: {wf.errors[model]}")
    guard = HoldoutGuard(sim_engine)
    for key in (dv, probe.version):
        if guard.used(key, wf.holdout_hash):
            raise HoldoutAlreadyUsed(f"{key} already scored on holdout {wf.holdout_hash[:12]}")

    # kappa: how much the model's disagreement with the market predicted outcomes, from
    # development walk-forward predictions only.
    pr = wf.predictions[wf.predictions["model"] == model]
    mids = ds.meta.set_index("ticker").loc[pr["ticker"], "market_mid"].to_numpy()
    kappa = fit_kappa(pr["p"], mids, pr["y"])

    idx = np.arange(len(ds))
    hold = wf.holdout_rows
    start = ds.meta["decision_ts"].iloc[hold].min()
    dev_mask = np.isin(idx, wf.dev_rows) & (ds.meta["label_ts"] <= start - embargo).to_numpy()
    fitted = make().fit(ds.subset(dev_mask))
    hs = ds.subset(np.isin(idx, hold))
    p = np.asarray(fitted.predict(hs).p, dtype=float)
    mid = hs.meta["market_mid"].to_numpy(dtype=float)
    has = ~np.isnan(mid)
    clusters = hs.meta["cluster"].to_numpy()
    coverage = float(has.mean()) if len(has) else 0.0
    metrics = summarize(p, hs.y, clusters=clusters, n_boot=n_boot)
    diff = (paired_bootstrap_diff(brier, p[has], mid[has], hs.y[has], clusters=clusters[has],
                                  n_boot=n_boot, level=level) if has.sum() >= 30 else None)

    sig = hs.meta[["ticker", "decision_ts", "cluster", "category", "series_ticker"]].assign(p=p)
    res = pd.DataFrame({"ticker": hs.meta["ticker"], "resolve_ts": hs.meta["label_ts"],
                        "result": np.where(hs.y == 1.0, "yes", "no")})
    strat = replace(strategy or StrategyConfig(), shrink=kappa)
    bt = run_backtest(sig, res, MarketData(market_engine, sig["ticker"]), strategy=strat,
                      fee_for=fee_schedules(market_engine))
    save_run(sim_engine, bt, kind="holdout", model_version=dv, holdout_hash=wf.holdout_hash)
    pos = bt.positions
    settled = pos[pos["settled"]] if len(pos) else pos
    pnl_ci = (_cluster_sum_ci((settled["payout"] - settled["cost"] - settled["fees"]).to_numpy(),
                              settled["cluster"].to_numpy(), level) if len(settled) else None)

    gate = {
        "holdout_brier_diff_vs_market": None if diff is None else round(diff.estimate, 6),
        "holdout_brier_diff_vs_market_ci_high": None if diff is None else round(diff.high, 6),
        "holdout_net_pnl": bt.summary["net_pnl"],
        "holdout_net_pnl_ci_low": None if pnl_ci is None else pnl_ci[0],
        "ci_level": round(level, 6), "certification_rank": k,
        "holdout_market_coverage": round(coverage, 4), "shrink_kappa": kappa,
        "paper_days": 0, "paper_degradation_alerts": 0, "paper_start": None,
    }
    reasons = []
    if coverage < MIN_COVERAGE:
        reasons.append(f"market price coverage {coverage:.0%} < {MIN_COVERAGE:.0%}")
    if diff is None or not diff.high < 0:
        reasons.append(f"holdout Brier vs market CI high = {gate['holdout_brier_diff_vs_market_ci_high']}"
                       f" (needs < 0 at {level:.4f})")
    if pnl_ci is None or not pnl_ci[0] > 0:
        reasons.append(f"holdout net P&L CI low = {gate['holdout_net_pnl_ci_low']} "
                       f"(needs > 0 at {level:.4f}); net P&L {bt.summary['net_pnl']}")
    incumbent = next((r for r in prior if r["status"] == "promoted"), None)
    if incumbent is not None:
        inc_brier = (incumbent["validation_json"] or {}).get("holdout", {}).get("brier")
        if inc_brier is not None and not metrics["brier"]["estimate"] < inc_brier:
            reasons.append(f"holdout Brier {metrics['brier']['estimate']:.4f} does not beat "
                           f"incumbent {incumbent['model_version']} ({inc_brier:.4f})")
    status = "rejected" if reasons else "promoted"

    hold_metrics = {"n": len(hs), "brier": metrics["brier"]["estimate"],
                    "log_loss": metrics["log_loss"]["estimate"], "ece": metrics["ece"]["estimate"],
                    "train_rows": int(dev_mask.sum()), "holdout_hash": wf.holdout_hash}
    for key in (dv, probe.version):
        guard.record(key, wf.holdout_hash, {**hold_metrics, "deployment": dv})
    now = utcnow()
    validation = {
        "walk_forward": {m: {"brier": s["brier"]["estimate"], "n": s["n"],
                             "vs_market": (s.get("vs_market") or {}).get("estimate")}
                         for m, s in wf.summary.items()},
        "holdout": hold_metrics,
        "holdout_backtest": {k2: v for k2, v in bt.summary.items() if k2 != "rejections"},
        "holdout_run_id": bt.run_id, "strategy": asdict(strat), "gate": gate,
        "reasons": reasons,
    }
    with sim_engine.begin() as c:
        if status == "promoted" and incumbent is not None:
            c.execute(update(model_registry)
                      .where(model_registry.c.model_version == incumbent["model_version"])
                      .values(status="retired", status_ts=now))
        if existing is not None:   # upgrade the candidate row; its paper log is kept
            c.execute(model_registry.delete().where(model_registry.c.model_version == dv))
        c.execute(model_registry.insert().values(
            model_version=dv, model_name=model, category=str(ds.meta["category"].iloc[0]),
            params_hash=probe.version.rsplit("-", 1)[-1],
            params_json={"model": model, "model_version": probe.version, "series": series,
                         "horizon_h": horizon_h, "params": probe.params()},
            trained_ts=now, train_window_end=pd.Timestamp(start).to_pydatetime(),
            validation_json=_plain(validation), status=status, status_ts=now))
    log_event(sim_engine, "promotion", "info" if status == "promoted" else "warning",
              f"{dv} {status}", {"reasons": reasons, "gate": gate})
    return Certification(dv, status, reasons, gate)


def register_candidate(market_engine: Engine, sim_engine: Engine, *, model: str, series: str,
                       horizon_h: float = 30.0, folds: int = 5, holdout_frac: float = 0.2,
                       min_train: int = 50, embargo_h: float = 0.0,
                       strategy: StrategyConfig | None = None, as_of=None) -> Certification:
    """Admit a deployment to forward paper trading WITHOUT looking at the holdout.

    kappa is fit on development walk-forward predictions only. The gate stays closed (no
    holdout evidence) until ``certify`` is run, which upgrades this row; the forward paper
    log is kept and replayed under the certified strategy."""
    if model not in ALL_MODELS:
        raise ValueError(f"unknown model {model!r}")
    make = ALL_MODELS[model]
    probe = make()
    dv = deployment_version(probe.version, series, horizon_h)
    if any(r["model_version"] == dv for r in _prior_certifications(sim_engine, series, horizon_h)):
        raise ValueError(f"{dv} is already registered")
    ds = build_dataset(market_engine, series=series, horizon=timedelta(hours=horizon_h),
                       as_of=as_of)
    wf = run_walk_forward(ds, {"market_implied": ALL_MODELS["market_implied"], model: make},
                          n_folds=folds, holdout_frac=holdout_frac, min_train=min_train,
                          embargo=timedelta(hours=embargo_h), n_boot=200)
    if model in wf.errors:
        raise RuntimeError(f"{model} failed in walk-forward: {wf.errors[model]}")
    pr = wf.predictions[wf.predictions["model"] == model]
    mids = ds.meta.set_index("ticker").loc[pr["ticker"], "market_mid"].to_numpy()
    kappa = fit_kappa(pr["p"], mids, pr["y"])
    strat = replace(strategy or StrategyConfig(), shrink=kappa)
    gate = {"candidate": True, "shrink_kappa": kappa,
            "holdout_brier_diff_vs_market_ci_high": None, "holdout_net_pnl_ci_low": None,
            "paper_days": 0, "paper_degradation_alerts": 0, "paper_start": None}
    validation = {
        "walk_forward": {m: {"brier": s2["brier"]["estimate"], "n": s2["n"],
                             "vs_market": (s2.get("vs_market") or {}).get("estimate")}
                         for m, s2 in wf.summary.items()},
        "strategy": asdict(strat), "gate": gate,
        "reasons": ["candidate: holdout not yet evaluated (run `predlab certify`)"],
    }
    now = utcnow()
    with sim_engine.begin() as c:
        c.execute(model_registry.insert().values(
            model_version=dv, model_name=model, category=str(ds.meta["category"].iloc[0]),
            params_hash=probe.version.rsplit("-", 1)[-1],
            params_json={"model": model, "model_version": probe.version, "series": series,
                         "horizon_h": horizon_h, "params": probe.params()},
            trained_ts=now, validation_json=_plain(validation), status="candidate",
            status_ts=now))
    log_event(sim_engine, "promotion", "info", f"{dv} registered as candidate",
              {"gate": gate})
    return Certification(dv, "candidate", validation["reasons"], gate)


def _plain(o):
    """JSON-safe copy (numpy scalars, timestamps, NaN)."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return o
