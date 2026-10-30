"""Learn how much to trust a model's disagreement with the market, from its own record.

For each walk-forward fold k, fit on earlier out-of-sample predictions whose outcomes were
public before fold k began:

    logit P(y=1) = a + b * logit(m) + kappa * (logit(p) - logit(m))

kappa (clamped to [0, 1]) is how much of the model's disagreement with the market price
actually predicted outcomes. Signals in fold k are then shrunk:

    logit(p') = logit(m) + kappa_k * (logit(p) - logit(m))

A model whose disagreements are noise gets kappa ~ 0 and stops trading. Fold 0 has no
track record and gets kappa = 0 (trust the market). No fold uses its own outcomes.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MIN_ROWS = 100


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def learned_shrink(preds: pd.DataFrame, meta: pd.DataFrame, folds) -> tuple[pd.Series, dict]:
    """preds: one model's walk-forward rows (ticker, fold, p, y).
    Returns (shrunk p indexed like ``preds``, {fold: kappa})."""
    m = meta.set_index("ticker")
    df = preds.copy()
    df["mid"] = m.loc[df["ticker"], "market_mid"].to_numpy()
    df["label_ts"] = m.loc[df["ticker"], "label_ts"].to_numpy()
    df["d"] = _logit(df["p"]) - _logit(df["mid"])
    kappas, out = {}, pd.Series(df["p"].to_numpy(), index=df.index, dtype=float)
    for f in folds:
        start = f.valid_start
        hist = df[(df["fold"] < f.index) & (pd.to_datetime(df["label_ts"], utc=True) <= start)
                  & df["mid"].notna()]
        k = 0.0
        if len(hist) >= MIN_ROWS and hist["y"].nunique() == 2:
            X = np.column_stack([_logit(hist["mid"]), hist["d"]])
            k = float(np.clip(LogisticRegression(C=1e4, max_iter=2000)
                              .fit(X, hist["y"]).coef_[0][1], 0.0, 1.0))
        kappas[f.index] = round(k, 4)
        rows = (df["fold"] == f.index) & df["mid"].notna()
        lm = _logit(df.loc[rows, "mid"])
        out.loc[rows] = 1 / (1 + np.exp(-(lm + k * df.loc[rows, "d"])))
    return out, kappas
