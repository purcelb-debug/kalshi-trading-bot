"""Walk-forward evaluation of several models on the same folds, plus the one-time holdout.

Every model sees identical train/validation rows, so comparisons are paired. Results are
reported with cluster-bootstrap CIs (clusters = events, so correlated same-event markets
are not treated as independent evidence) and per-fold stability.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from predlab.evaluation.metrics import brier, log_loss, paired_bootstrap_diff, summarize
from predlab.evaluation.splits import Fold, HoldoutGuard, holdout_split, walk_forward
from predlab.features.pointintime import Dataset
from predlab.logs import get_logger
from predlab.models.base import Forecaster

log = get_logger(__name__)
REFERENCE = "market_implied"


@dataclass
class EvalResult:
    predictions: pd.DataFrame          # one row per (model, validation market)
    summary: dict[str, dict]           # model -> metrics
    folds: list[Fold]
    holdout_hash: str
    holdout_rows: np.ndarray
    dev_rows: np.ndarray
    versions: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    #: share of validated markets with a real market price; below 1.0 the "market"
    #: reference partly falls back to the training base rate.
    market_coverage: float = float("nan")
    #: share of validated markets priced <= 0.03 or >= 0.97 at decision time, i.e. the
    #: outcome was effectively already known; high values mean the horizon is too short.
    decided_share: float = float("nan")

    def table(self) -> pd.DataFrame:
        rows = []
        for name, s in self.summary.items():
            rows.append({
                "model": name, "n": s["n"], "brier": s["brier"]["estimate"],
                "brier_lo": s["brier"]["low"], "brier_hi": s["brier"]["high"],
                "log_loss": s["log_loss"]["estimate"], "ece": s["ece"]["estimate"],
                "fold_brier_sd": s["fold_brier_sd"],
                "vs_market_brier_diff": s.get("vs_market", {}).get("estimate"),
                "vs_market_lo": s.get("vs_market", {}).get("low"),
                "vs_market_hi": s.get("vs_market", {}).get("high"),
            })
        return pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)


def run_walk_forward(ds: Dataset, factories: dict[str, Callable[[], Forecaster]], *,
                     n_folds: int = 5, holdout_frac: float = 0.2, min_train: int = 50,
                     embargo: timedelta = timedelta(0), n_boot: int = 500) -> EvalResult:
    dev, hold, h = holdout_split(ds, holdout_frac)
    folds = walk_forward(ds, dev, n_folds=n_folds, min_train=min_train, embargo=embargo)
    if not folds:
        raise ValueError("not enough history for any walk-forward fold")
    recs, versions, errors = [], {}, {}
    for fold in folds:
        tr, va = ds.subset(np.isin(np.arange(len(ds)), fold.train)), \
            ds.subset(np.isin(np.arange(len(ds)), fold.valid))
        for name, make in factories.items():
            if name in errors:
                continue
            try:
                model = make().fit(tr)
                pr = model.predict(va)
            except Exception as e:  # noqa: BLE001 - one failing model must not hide the others
                errors[name] = f"{type(e).__name__}: {e}"
                log.warning("model_failed", model=name, fold=fold.index, error=errors[name])
                continue
            versions[name] = model.version
            recs.append(pd.DataFrame({
                "model": name, "fold": fold.index, "ticker": va.meta["ticker"],
                "category": va.meta["category"], "cluster": va.meta["cluster"],
                "decision_ts": va.meta["decision_ts"], "y": va.y, "p": pr.p,
                "ci_low": pr.ci_low if pr.ci_low is not None else np.nan,
                "ci_high": pr.ci_high if pr.ci_high is not None else np.nan,
            }))
    preds = pd.concat(recs, ignore_index=True)
    for name in errors:
        preds = preds[preds["model"] != name]
    summary = _summarize(preds, n_boot)
    validated = ds.meta[ds.meta["ticker"].isin(set(preds["ticker"]))]
    coverage = float(validated["market_mid"].notna().mean())
    mid = validated["market_mid"].dropna()
    decided = float(((mid <= 0.03) | (mid >= 0.97)).mean()) if len(mid) else float("nan")
    return EvalResult(preds, summary, folds, h, hold, dev, versions, errors, coverage, decided)


def _summarize(preds: pd.DataFrame, n_boot: int) -> dict:
    summary = {}
    ref = preds[preds["model"] == REFERENCE].set_index("ticker")
    for name, g in preds.groupby("model"):
        s = summarize(g["p"], g["y"], clusters=g["cluster"], n_boot=n_boot)
        fold_b = g.groupby("fold").apply(lambda d: brier(d["p"], d["y"]), include_groups=False)
        s["fold_brier"] = fold_b.round(5).tolist()
        s["fold_brier_sd"] = float(fold_b.std(ddof=0))
        if name != REFERENCE and len(ref):
            r = ref.loc[g["ticker"]]
            s["vs_market"] = paired_bootstrap_diff(
                brier, g["p"].to_numpy(), r["p"].to_numpy(), g["y"].to_numpy(),
                clusters=g["cluster"].to_numpy(), n_boot=n_boot).as_dict()
            s["vs_market_log_loss"] = paired_bootstrap_diff(
                log_loss, g["p"].to_numpy(), r["p"].to_numpy(), g["y"].to_numpy(),
                clusters=g["cluster"].to_numpy(), n_boot=n_boot).as_dict()
        summary[name] = s
    return summary


def evaluate_holdout(ds: Dataset, result: EvalResult, name: str,
                     make: Callable[[], Forecaster], guard: HoldoutGuard, *,
                     embargo: timedelta = timedelta(0), n_boot: int = 1000) -> dict:
    """Fit once on all development rows whose labels were public before the holdout,
    score the holdout once, and record it. A second call for the same version raises."""
    hold = result.holdout_rows
    start = ds.meta["decision_ts"].iloc[hold].min()
    dev_mask = np.isin(np.arange(len(ds)), result.dev_rows) & (
        ds.meta["label_ts"] <= start - embargo).to_numpy()
    model = make().fit(ds.subset(dev_mask))
    if guard.used(model.version, result.holdout_hash):
        from predlab.evaluation.splits import HoldoutAlreadyUsed
        raise HoldoutAlreadyUsed(f"{model.version} already scored on this holdout")
    hs = ds.subset(np.isin(np.arange(len(ds)), hold))
    pr = model.predict(hs)
    metrics = summarize(pr.p, hs.y, clusters=hs.meta["cluster"], n_boot=n_boot)
    metrics["model_version"] = model.version
    metrics["train_rows"] = int(dev_mask.sum())
    guard.record(model.version, result.holdout_hash, metrics)
    return metrics
