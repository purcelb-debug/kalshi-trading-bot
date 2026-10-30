"""Baseline forecasters.

* MarketImplied      - the market's own mid price at decision time (the reference to beat)
* FrequencyBayes     - Beta-Binomial base rate per group; credible interval from the posterior
* Logistic           - plain / L2 / L1 logistic regression; CI from a cluster bootstrap
* HistGBM, LightGBM  - gradient-boosted trees; no CI (not defensible without more work)
"""

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from predlab.features.pointintime import Dataset
from predlab.models.base import Forecaster, Predictions, clip_p, top_contributions


class MarketImplied(Forecaster):
    """p = market mid at decision time. Falls back to the training base rate (flagged)."""

    name = "market_implied"

    def fit(self, ds: Dataset):
        self.base_rate_ = float(ds.y.mean()) if len(ds) else 0.5
        return self

    def predict(self, ds: Dataset) -> Predictions:
        mid = ds.meta["market_mid"].to_numpy(dtype=float)
        missing = np.isnan(mid)
        p = np.where(missing, self.base_rate_, mid)
        return Predictions(clip_p(p), flags=[["fallback_base_rate"] if m else [] for m in missing],
                           explanations=[{"source": "base_rate" if m else "market_mid"}
                                         for m in missing])


class FrequencyBayes(Forecaster):
    """Posterior mean of Beta(a + yes, b + no) per group; unseen groups use the pooled rate."""

    name = "frequency_bayes"
    provides_ci = True

    def __init__(self, group_by: str = "series_ticker", prior=(1.0, 1.0), level: float = 0.9):
        self.group_by, self.prior, self.level = group_by, tuple(prior), level

    def params(self):
        return {"group_by": self.group_by, "prior": self.prior, "level": self.level}

    def fit(self, ds: Dataset):
        a, b = self.prior
        g = ds.meta[self.group_by].fillna("_").to_numpy()
        self.groups_ = {}
        for key in np.unique(g):
            yk = ds.y[g == key]
            self.groups_[key] = (a + yk.sum(), b + len(yk) - yk.sum(), len(yk))
        self.pooled_ = (a + ds.y.sum(), b + len(ds.y) - ds.y.sum(), len(ds.y))
        return self

    def predict(self, ds: Dataset) -> Predictions:
        q = (1 - self.level) / 2
        g = ds.meta[self.group_by].fillna("_").to_numpy()
        p, lo, hi, ex, fl = [], [], [], [], []
        for key in g:
            aa, bb, n = self.groups_.get(key, self.pooled_)
            p.append(aa / (aa + bb))
            lo.append(beta_dist.ppf(q, aa, bb))
            hi.append(beta_dist.ppf(1 - q, aa, bb))
            ex.append({"group": str(key), "n_train": int(n), "alpha": float(aa), "beta": float(bb)})
            fl.append([] if key in self.groups_ else ["unseen_group_pooled"])
        return Predictions(clip_p(p), np.array(lo), np.array(hi), ex, fl)


def _linear_pipeline(estimator):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                       keep_empty_features=True),
                         StandardScaler(), estimator)


class Logistic(Forecaster):
    """Logistic regression. ``penalty`` None = unregularized; 'l2'/'l1' choose C by
    time-ordered CV inside the training window. CI: percentile interval over refits on
    cluster-bootstrap resamples of the training set (clusters = events)."""

    provides_ci = True

    def __init__(self, penalty: str | None = None, n_boot: int = 30, level: float = 0.9,
                 seed: int = 0):
        self.penalty, self.n_boot, self.level, self.seed = penalty, n_boot, level, seed
        self.name = {None: "logistic", "l2": "logistic_l2", "l1": "logistic_l1"}[penalty]

    def params(self):
        return {"penalty": self.penalty, "n_boot": self.n_boot, "level": self.level,
                "seed": self.seed}

    def _make(self, n):
        if self.penalty is None:
            est = LogisticRegression(C=1e6, max_iter=2000)
        else:
            est = LogisticRegressionCV(
                Cs=np.logspace(-3, 2, 12), cv=TimeSeriesSplit(n_splits=3 if n >= 60 else 2),
                l1_ratios=(1.0 if self.penalty == "l1" else 0.0,),
                solver="saga" if self.penalty == "l1" else "lbfgs",
                scoring="neg_log_loss", max_iter=5000, use_legacy_attributes=False)
        return _linear_pipeline(est)

    def fit(self, ds: Dataset):
        if len(np.unique(ds.y)) < 2:
            raise ValueError("training window contains a single outcome class")
        self.features_ = list(ds.X.columns)
        self.model_ = self._make(len(ds)).fit(ds.X, ds.y)
        self.boot_ = []
        rng = np.random.default_rng(self.seed)
        clusters = ds.meta["cluster"].to_numpy()
        uniq = np.unique(clusters)
        members = {c: np.flatnonzero(clusters == c) for c in uniq}
        for _ in range(self.n_boot):
            ix = np.concatenate([members[c] for c in rng.choice(uniq, len(uniq))])
            if len(np.unique(ds.y[ix])) < 2:
                continue
            # Bootstrap refits reuse the plain estimator at the selected C (fast, stable).
            C = self._chosen_C()
            est = _linear_pipeline(LogisticRegression(
                C=C, max_iter=2000, **({"l1_ratio": 1.0, "solver": "saga"}
                                       if self.penalty == "l1" else {})))
            self.boot_.append(est.fit(ds.X.iloc[ix], ds.y[ix]))
        return self

    def _chosen_C(self):
        est = self.model_[-1]
        return float(np.ravel(est.C_)[0]) if hasattr(est, "C_") else float(est.C)

    def predict(self, ds: Dataset) -> Predictions:
        X = ds.X[self.features_]
        p = self.model_.predict_proba(X)[:, 1]
        lo = hi = None
        if len(self.boot_) >= 10:
            q = (1 - self.level) / 2
            bs = np.stack([m.predict_proba(X)[:, 1] for m in self.boot_])
            lo, hi = np.quantile(bs, q, axis=0), np.quantile(bs, 1 - q, axis=0)
        # Contributions in log-odds units on the standardized scale (exact for linear models).
        Z = self.model_[:-1].transform(X)
        coef = self.model_[-1].coef_[0]
        names = list(self.model_[0].get_feature_names_out(self.features_))
        contrib = Z * coef
        ex = [top_contributions(names, contrib[i]) for i in range(len(X))]
        return Predictions(clip_p(p), lo, hi, ex)


class HistGBM(Forecaster):
    name = "hist_gbm"

    def __init__(self, max_iter=100, learning_rate=0.05, max_depth=2, min_samples_leaf=30,
                 l2=1.0, seed=0):
        self.kw = {"max_iter": max_iter, "learning_rate": learning_rate, "max_depth": max_depth,
                       "min_samples_leaf": min_samples_leaf, "l2_regularization": l2,
                       "random_state": seed, "early_stopping": False}

    def params(self):
        return self.kw

    def fit(self, ds: Dataset):
        self.features_ = list(ds.X.columns)
        self.model_ = HistGradientBoostingClassifier(**self.kw).fit(ds.X, ds.y)
        return self

    def predict(self, ds: Dataset) -> Predictions:
        p = self.model_.predict_proba(ds.X[self.features_])[:, 1]
        return Predictions(clip_p(p), explanations=[{"note": "see global importances"}] * len(ds))


class LightGBM(Forecaster):
    """LightGBM; per-row explanations are exact TreeSHAP values (``pred_contrib``)."""

    name = "lightgbm"

    def __init__(self, n_estimators=150, learning_rate=0.03, num_leaves=7,
                 min_child_samples=30, reg_lambda=1.0, seed=0):
        self.kw = {"n_estimators": n_estimators, "learning_rate": learning_rate,
                       "num_leaves": num_leaves, "min_child_samples": min_child_samples,
                       "reg_lambda": reg_lambda, "random_state": seed, "verbose": -1}

    def params(self):
        return self.kw

    def fit(self, ds: Dataset):
        from lightgbm import LGBMClassifier
        if ds.X.shape[1] == 0:
            raise ValueError("lightgbm needs at least one feature")
        self.features_ = list(ds.X.columns)
        self.model_ = LGBMClassifier(**self.kw).fit(ds.X, ds.y)
        return self

    def predict(self, ds: Dataset) -> Predictions:
        X = ds.X[self.features_]
        p = self.model_.predict_proba(X)[:, 1]
        shap = self.model_.predict(X, pred_contrib=True)[:, :-1]  # drop bias column
        return Predictions(clip_p(p), explanations=[top_contributions(self.features_, shap[i])
                                                    for i in range(len(X))])


MODEL_FACTORIES = {
    "market_implied": MarketImplied,
    "frequency_bayes": FrequencyBayes,
    "logistic": lambda: Logistic(None),
    "logistic_l2": lambda: Logistic("l2"),
    "logistic_l1": lambda: Logistic("l1"),
    "hist_gbm": HistGBM,
    "lightgbm": LightGBM,
}
