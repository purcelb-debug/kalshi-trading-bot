"""Overfitting and leakage probes.

* shuffled_label_probe: retrain on permuted labels. With no real signal available, a model
  must not beat a constant forecast out of sample; if it does, something leaks (a feature
  encodes the label, or splits overlap).
* multiple_testing_check: when k models are compared against the market, the chance that
  the best one looks better by luck grows with k. Recompute each paired CI at the
  Bonferroni level 1 - alpha/k and report which edges survive.
"""

import numpy as np

from predlab.evaluation.metrics import brier, paired_bootstrap_diff
from predlab.evaluation.walkforward import REFERENCE, run_walk_forward


def shuffled_label_probe(ds, factories: dict, *, n_perm: int = 3, n_folds: int = 3,
                         seed: int = 0, tolerance: float = 0.005) -> dict:
    """Per model: mean Brier advantage over a constant forecast on permuted labels.
    ``suspicious`` = the model beat the constant by more than ``tolerance``."""
    rng = np.random.default_rng(seed)
    gains: dict[str, list[float]] = {}
    for _ in range(n_perm):
        shuffled = ds.subset(np.ones(len(ds), bool))
        shuffled.y = rng.permutation(ds.y)
        res = run_walk_forward(shuffled, factories, n_folds=n_folds, n_boot=50)
        for name, g in res.predictions.groupby("model"):
            const = np.full(len(g), g["y"].mean())
            gains.setdefault(name, []).append(brier(const, g["y"]) - brier(g["p"], g["y"]))
    return {n: {"mean_gain_vs_constant": float(np.mean(v)),
                "suspicious": bool(np.mean(v) > tolerance)} for n, v in gains.items()}


def multiple_testing_check(result, alpha: float = 0.05, n_boot: int = 1000) -> dict:
    preds = result.predictions
    ref = preds[preds["model"] == REFERENCE].set_index("ticker")
    models = [m for m in preds["model"].unique() if m != REFERENCE]
    if ref.empty or not models:
        return {}
    level = 1 - alpha / len(models)
    out = {}
    for m in models:
        g = preds[preds["model"] == m]
        r = ref.loc[g["ticker"]]
        ci = paired_bootstrap_diff(brier, g["p"].to_numpy(), r["p"].to_numpy(),
                                   g["y"].to_numpy(), clusters=g["cluster"].to_numpy(),
                                   n_boot=n_boot, level=level)
        out[m] = {"diff": ci.estimate, "low": ci.low, "high": ci.high,
                  "survives": bool(ci.high < 0), "level": round(level, 5)}
    return out
