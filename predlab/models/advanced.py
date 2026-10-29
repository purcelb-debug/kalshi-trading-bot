"""Phase 4 models.

* MarketTrajectory - time-series model of the market's own implied probability:
  logistic regression on logit(mid now) and its 6h / 24h momentum. Tests whether price
  *paths* carry information the current price does not.
* Calibrated       - wraps any forecaster with Platt or isotonic recalibration. The base
  model is fit on the earlier part of the training window and the calibrator on the later
  part, so the calibrator never sees data the base model was fit on.
* Stacked          - non-negative, sum-to-one weights over member forecasts, chosen by
  minimizing log loss on the later part of the training window; members are then refit
  on the whole window.

All three only ever see the training Dataset passed to ``fit``.
"""

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from predlab.features.pointintime import Dataset
from predlab.models.base import Forecaster, Predictions, clip_p

TRAJ_FEATURES = ("mkt_mid", "mkt_mom6", "mkt_mom24")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 0.005, 0.995)
    return np.log(p / (1 - p))


def _time_split(ds: Dataset, frac: float) -> tuple[Dataset, Dataset]:
    """Chronological split of a (time-sorted) dataset: first (1-frac) / last frac."""
    n = len(ds)
    cut = round(n * (1 - frac))
    idx = np.arange(n)
    return ds.subset(idx < cut), ds.subset(idx >= cut)


class MarketTrajectory(Forecaster):
    name = "market_trajectory"

    def __init__(self, C: float = 1.0):
        self.C = C

    def params(self):
        return {"C": self.C}

    def _design(self, ds: Dataset) -> np.ndarray:
        cols = []
        for f in TRAJ_FEATURES:
            v = ds.X[f].to_numpy(dtype=float) if f in ds.X else np.full(len(ds), np.nan)
            cols.append(_logit(v) if f == "mkt_mid" else v)
        X = np.column_stack(cols)
        return np.where(np.isnan(X), 0.0, X)   # missing momentum = no change

    def fit(self, ds: Dataset):
        if "mkt_mid" not in ds.X:
            raise ValueError("market_trajectory needs market features (mkt_mid)")
        self.base_rate_ = float(ds.y.mean())
        self.model_ = LogisticRegression(C=self.C, max_iter=2000).fit(self._design(ds), ds.y)
        return self

    def predict(self, ds: Dataset) -> Predictions:
        Z = self._design(ds)
        p = self.model_.predict_proba(Z)[:, 1]
        missing = ds.X["mkt_mid"].isna().to_numpy() if "mkt_mid" in ds.X else np.ones(len(ds), bool)
        p = np.where(missing, self.base_rate_, p)
        coef = self.model_.coef_[0]
        ex = [{"logit_mid_coef": round(float(coef[0]), 4), "mom6_coef": round(float(coef[1]), 4),
               "mom24_coef": round(float(coef[2]), 4)}] * len(ds)
        return Predictions(clip_p(p), explanations=ex,
                           flags=[["fallback_base_rate"] if m else [] for m in missing])


class Calibrated(Forecaster):
    def __init__(self, base: Callable[[], Forecaster], method: str = "platt",
                 cal_frac: float = 0.3):
        if method not in ("platt", "isotonic"):
            raise ValueError("method must be platt or isotonic")
        self.base_factory, self.method, self.cal_frac = base, method, cal_frac
        self.base_name = base().name
        self.name = f"{self.base_name}+{method}"

    def params(self):
        return {"base": self.base_factory().version, "method": self.method,
                "cal_frac": self.cal_frac}

    def fit(self, ds: Dataset):
        fit_part, cal_part = _time_split(ds, self.cal_frac)
        if len(np.unique(cal_part.y)) < 2 or len(fit_part) < 20:
            raise ValueError("calibration window too small or single-class")
        self.base_ = self.base_factory().fit(fit_part)
        raw = self.base_.predict(cal_part).p
        if self.method == "platt":
            self.cal_ = LogisticRegression(C=1e4, max_iter=2000).fit(
                _logit(raw).reshape(-1, 1), cal_part.y)
        else:
            self.cal_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
                raw, cal_part.y)
        return self

    def predict(self, ds: Dataset) -> Predictions:
        pr = self.base_.predict(ds)
        if self.method == "platt":
            p = self.cal_.predict_proba(_logit(pr.p).reshape(-1, 1))[:, 1]
        else:
            p = self.cal_.predict(pr.p)
        ex = [{**(e or {}), "calibration": self.method, "raw_p": round(float(r), 4)}
              for e, r in zip(pr.explanations or [{}] * len(ds), pr.p)]
        return Predictions(clip_p(p), explanations=ex, flags=pr.flags)


class Stacked(Forecaster):
    name = "stacked"

    def __init__(self, members: dict[str, Callable[[], Forecaster]], weight_frac: float = 0.3):
        self.members, self.weight_frac = members, weight_frac

    def params(self):
        return {"members": sorted(self.members), "weight_frac": self.weight_frac}

    def fit(self, ds: Dataset):
        fit_part, w_part = _time_split(ds, self.weight_frac)
        preds, names = [], []
        for name, make in self.members.items():
            try:
                preds.append(make().fit(fit_part).predict(w_part).p)
                names.append(name)
            except ValueError:
                continue  # a member that cannot fit this window is skipped, not fatal
        if not preds:
            raise ValueError("no ensemble member could be fit")
        P = np.column_stack(preds)
        y = w_part.y

        def loss(w):
            p = np.clip(P @ w, 1e-6, 1 - 1e-6)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        k = P.shape[1]
        res = minimize(loss, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
        w = np.clip(res.x, 0, None)
        self.weights_ = dict(zip(names, (w / w.sum()).round(6)))
        self.fitted_ = {n: self.members[n]().fit(ds) for n in names}
        return self

    def predict(self, ds: Dataset) -> Predictions:
        p = sum(self.weights_[n] * m.predict(ds).p for n, m in self.fitted_.items())
        return Predictions(clip_p(p), explanations=[{"weights": self.weights_}] * len(ds))
