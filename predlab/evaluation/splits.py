"""Chronological walk-forward splits and a frozen, single-use holdout.

* The holdout is the most recent ``holdout_frac`` of rows by decision time. It is carved
  off first and never enters walk-forward folds. Its identity is a SHA-256 over its
  tickers, so the same holdout is recognisable across runs.
* Each walk-forward fold validates on a contiguous time window. Training rows must have
  a decision time before the window *and* a label known (``label_ts``) at least
  ``embargo`` before it, so no fold trains on an outcome that was not yet public.
* Every split is checked by ``assert_no_leak``.
"""

import hashlib
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import Engine, select

from predlab.db.schema import holdout_evaluations
from predlab.features.pointintime import Dataset, LookAheadError
from predlab.timeutil import utcnow


@dataclass(frozen=True)
class Fold:
    index: int
    train: np.ndarray       # integer row positions
    valid: np.ndarray
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def holdout_split(ds: Dataset, holdout_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (development rows, holdout rows, holdout hash). Rows must be time-sorted."""
    if not 0 < holdout_frac < 1:
        raise ValueError("holdout_frac must be in (0, 1)")
    ts = ds.meta["decision_ts"]
    if not ts.is_monotonic_increasing:
        raise ValueError("dataset must be sorted by decision_ts")
    cut_ts = ts.iloc[int(np.floor(len(ts) * (1 - holdout_frac)))]
    hold = np.flatnonzero(ts >= cut_ts)          # whole timestamps stay on one side
    dev = np.flatnonzero(ts < cut_ts)
    h = hashlib.sha256("\n".join(sorted(ds.meta["ticker"].iloc[hold])).encode()).hexdigest()
    return dev, hold, h


def walk_forward(ds: Dataset, rows: np.ndarray, *, n_folds: int = 5,
                 min_train: int = 50, embargo: timedelta = timedelta(0)) -> list[Fold]:
    """Expanding-window folds over ``rows`` (the development set, time-sorted).

    ``rows`` is cut into ``n_folds + 1`` equal-count time blocks. Fold k validates on block
    k (k = 1..n_folds) and trains on everything earlier whose label was already public.
    """
    meta = ds.meta.iloc[rows]
    ts = meta["decision_ts"].to_numpy()
    label_ts = meta["label_ts"].to_numpy()
    bounds = np.linspace(0, len(rows), n_folds + 2).astype(int)
    folds = []
    for k in range(1, n_folds + 1):
        start = ts[bounds[k]]
        end = ts[bounds[k + 1]] if bounds[k + 1] < len(rows) else ts[-1] + np.timedelta64(1, "s")
        valid = rows[(ts >= start) & (ts < end)]
        cutoff = start - np.timedelta64(int(embargo.total_seconds()), "s")
        train = rows[(ts < start) & (label_ts <= cutoff)]
        if len(train) < min_train or len(valid) == 0:
            continue
        f = Fold(len(folds), train, valid, pd.Timestamp(start), pd.Timestamp(end))
        assert_no_leak(ds, f, embargo)
        folds.append(f)
    return folds


def assert_no_leak(ds: Dataset, fold: Fold, embargo: timedelta = timedelta(0)) -> None:
    m = ds.meta
    tr, va = m.iloc[fold.train], m.iloc[fold.valid]
    if len(set(tr["ticker"]) & set(va["ticker"])):
        raise LookAheadError("train and validation share markets")
    if tr["decision_ts"].max() >= va["decision_ts"].min():
        raise LookAheadError("training rows are not strictly before validation")
    if tr["label_ts"].max() > va["decision_ts"].min() - embargo:
        raise LookAheadError("a training label was not public before validation began")


class HoldoutAlreadyUsed(RuntimeError):
    pass


class HoldoutGuard:
    """Allows each model version to be scored on a given holdout exactly once.

    The record lives in the simulation database so the rule survives restarts. Looking at
    holdout results and then retuning would turn the holdout into another validation set.
    """

    def __init__(self, sim_engine: Engine):
        self.engine = sim_engine

    def used(self, model_version: str, holdout_hash: str) -> bool:
        with self.engine.connect() as c:
            return c.execute(select(holdout_evaluations.c.model_version).where(
                holdout_evaluations.c.model_version == model_version,
                holdout_evaluations.c.holdout_hash == holdout_hash)).first() is not None

    def record(self, model_version: str, holdout_hash: str, metrics: dict) -> None:
        if self.used(model_version, holdout_hash):
            raise HoldoutAlreadyUsed(
                f"{model_version} was already evaluated on holdout {holdout_hash[:12]}")
        with self.engine.begin() as c:
            c.execute(holdout_evaluations.insert().values(
                model_version=model_version, holdout_hash=holdout_hash, evaluated_ts=utcnow(),
                metrics=metrics))
