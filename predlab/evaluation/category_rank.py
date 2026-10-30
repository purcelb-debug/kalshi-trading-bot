"""Compare event categories on forecasting quality and reliability.

This ranks how well and how reliably each category can be *forecast* from the data on
hand. It is not a recommendation to bet on any category.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from predlab.backtest.fees import FeeSchedule
from predlab.evaluation.metrics import brier
from predlab.evaluation.walkforward import REFERENCE, run_walk_forward
from predlab.features.pointintime import MARKET_FEATURES, build_dataset
from predlab.timeutil import utcnow

DISCLAIMER = ("Ranking of forecasting quality and reliability only. Not a recommendation to "
              "trade or bet on any category.")

#: criterion -> True if higher is better
CRITERIA = {
    "n": True,
    "brier": False,
    "log_loss": False,
    "ece": False,
    "fold_brier_sd": False,
    "freshness_days": False,
    "independent_info": True,
    "market_coverage": True,
    "est_cost_per_contract": False,
    "oos_win_share": True,
}


def _category_row(cat: str, ds, res) -> dict:
    table = res.table()
    cand = table[table["model"] != REFERENCE]
    best = cand.iloc[0] if len(cand) else table.iloc[0]
    preds = res.predictions
    validated = ds.meta[ds.meta["ticker"].isin(set(preds["ticker"]))]
    vX = ds.X.loc[validated.index]
    non_mkt = [c for c in ds.X.columns if c not in MARKET_FEATURES]
    indep = float(vX[non_mkt].notna().any(axis=1).mean()) if non_mkt else 0.0
    mid = validated["market_mid"]
    spread = vX["mkt_spread"] if "mkt_spread" in vX else pd.Series(np.nan, index=vX.index)
    fees = FeeSchedule()
    cost = (spread / 2 + mid.map(lambda m: fees.per_contract(m) if 0 < m < 1 else np.nan))
    # Out-of-sample stability: share of folds in which the best model beat the reference
    # (the market where priced, otherwise the frequency baseline is the natural floor).
    ref_name = REFERENCE if res.market_coverage >= 0.95 else "frequency_bayes"
    wins = []
    for _, g in preds.groupby("fold"):
        b = g[g["model"] == best["model"]]
        r = g[g["model"] == ref_name]
        if len(b) and len(r):
            wins.append(brier(b["p"], b["y"]) < brier(r["p"], r["y"]))
    return {
        "category": cat, "n": len(ds), "best_model": best["model"],
        "brier": float(best["brier"]), "log_loss": float(best["log_loss"]),
        "ece": float(best["ece"]), "fold_brier_sd": float(best["fold_brier_sd"]),
        "vs_market_brier_diff": best.get("vs_market_brier_diff"),
        "vs_market_ci": (best.get("vs_market_lo"), best.get("vs_market_hi")),
        "freshness_days": float((utcnow() - ds.meta["decision_ts"].max()).total_seconds() / 86400),
        "independent_info": indep,
        "market_coverage": float(res.market_coverage),
        "decided_share": float(res.decided_share),
        "est_cost_per_contract": float(cost.mean()) if cost.notna().any() else np.nan,
        "oos_win_share": float(np.mean(wins)) if wins else np.nan,
        "oos_reference": ref_name,
    }


def compare_categories(engine: Engine, factories: dict, *, source=None, categories=None,
                       series=None, horizon: timedelta = timedelta(hours=6), n_folds: int = 5,
                       holdout_frac: float = 0.2, min_train: int = 50, n_boot: int = 300,
                       min_rows: int = 100, since=None) -> tuple[pd.DataFrame, dict]:
    """Walk-forward every category independently; return (ranked table, skipped reasons)."""
    from predlab.categories import CATEGORIES

    rows, skipped = [], {}
    for cat in categories or CATEGORIES:
        try:
            ds = build_dataset(engine, source=source, category=cat, series=series,
                               horizon=horizon, since=since)
        except ValueError as e:
            skipped[cat] = str(e)
            continue
        if len(ds) < min_rows:
            skipped[cat] = f"only {len(ds)} settled markets (< {min_rows})"
            continue
        try:
            res = run_walk_forward(ds, factories, n_folds=n_folds, holdout_frac=holdout_frac,
                                   min_train=min_train, n_boot=n_boot)
        except ValueError as e:
            skipped[cat] = str(e)
            continue
        rows.append(_category_row(cat, ds, res))
    table = pd.DataFrame(rows)
    if table.empty:
        return table, skipped
    ranks = []
    for crit, higher in CRITERIA.items():
        ranks.append(table[crit].rank(ascending=not higher, na_option="bottom"))
    table["composite_rank_score"] = np.mean(ranks, axis=0)
    table = table.sort_values("composite_rank_score").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table, skipped
