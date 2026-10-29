"""Probabilistic forecast metrics with bootstrap confidence intervals.

All functions take predicted probabilities ``p`` for the YES outcome and binary outcomes
``y`` (1 = yes). Lower is better for Brier, log loss and ECE.
"""

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


def _arr(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("p and y must be 1-D arrays of equal length")
    if len(p) == 0:
        raise ValueError("empty input")
    if np.any((p < 0) | (p > 1)) or not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("p must lie in [0, 1] and y must be 0/1")
    return p, y


def brier(p, y) -> float:
    p, y = _arr(p, y)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps: float = EPS) -> float:
    """Mean negative log-likelihood; p is clipped to [eps, 1-eps] so certainty is finite."""
    p, y = _arr(p, y)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _bin_index(p, n_bins):
    return np.minimum((p * n_bins).astype(int), n_bins - 1)


def reliability_table(p, y, n_bins: int = 10) -> list[dict]:
    """Equal-width bins: count, mean forecast, observed frequency."""
    p, y = _arr(p, y)
    idx = _bin_index(p, n_bins)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append({"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins, "n": int(m.sum()),
                        "mean_p": float(p[m].mean()), "obs_freq": float(y[m].mean())})
    return out


def ece(p, y, n_bins: int = 10) -> float:
    """Expected calibration error: count-weighted |mean forecast - observed frequency|."""
    p, _ = _arr(p, y)
    rows = reliability_table(p, y, n_bins)
    return float(sum(r["n"] * abs(r["mean_p"] - r["obs_freq"]) for r in rows) / len(p))


def murphy_decomposition(p, y, n_bins: int = 10) -> dict:
    """Brier = reliability - resolution + uncertainty (+ within-bin term).

    The identity is exact only when forecasts are constant within each bin; the residual
    is reported as ``within_bin`` so the parts always sum to the Brier score.
    """
    p, y = _arr(p, y)
    n, base = len(p), y.mean()
    rel = res = 0.0
    for r in reliability_table(p, y, n_bins):
        rel += r["n"] * (r["mean_p"] - r["obs_freq"]) ** 2
        res += r["n"] * (r["obs_freq"] - base) ** 2
    rel, res, unc = rel / n, res / n, base * (1 - base)
    bs = brier(p, y)
    return {"brier": bs, "reliability": rel, "resolution": res, "uncertainty": unc,
            "within_bin": bs - (rel - res + unc)}


def skill_score(score: float, reference: float) -> float:
    """1 - score/reference (e.g. Brier skill score). >0 means better than the reference."""
    return float("nan") if reference == 0 else 1.0 - score / reference


METRICS = {"brier": brier, "log_loss": log_loss, "ece": ece}


@dataclass(frozen=True)
class CI:
    estimate: float
    low: float
    high: float
    level: float
    n_boot: int

    def as_dict(self):
        return {"estimate": self.estimate, "low": self.low, "high": self.high}


def _resample_index(rng, n, clusters):
    """Plain bootstrap, or cluster bootstrap when ``clusters`` is given (whole clusters
    are resampled, so correlated outcomes such as same-day events are not double-counted
    as independent evidence)."""
    if clusters is None:
        return rng.integers(0, n, n)
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    picks = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([members[k] for k in picks])


def bootstrap_ci(metric, p, y, *, clusters=None, n_boot: int = 1000, level: float = 0.95,
                 seed: int = 0) -> CI:
    p, y = _arr(p, y)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        ix = _resample_index(rng, len(p), clusters)
        stats[b] = metric(p[ix], y[ix])
    a = (1 - level) / 2
    return CI(metric(p, y), float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a)),
              level, n_boot)


def paired_bootstrap_diff(metric, p_a, p_b, y, *, clusters=None, n_boot: int = 1000,
                          level: float = 0.95, seed: int = 0) -> CI:
    """CI for metric(a) - metric(b) on the same outcomes. Negative = a is better
    for loss metrics. If the interval excludes 0 the difference is distinguishable."""
    p_a, y = _arr(p_a, y)
    p_b, _ = _arr(p_b, y)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        ix = _resample_index(rng, len(y), clusters)
        d[b] = metric(p_a[ix], y[ix]) - metric(p_b[ix], y[ix])
    a = (1 - level) / 2
    return CI(metric(p_a, y) - metric(p_b, y), float(np.quantile(d, a)),
              float(np.quantile(d, 1 - a)), level, n_boot)


def summarize(p, y, *, clusters=None, n_boot: int = 500, seed: int = 0) -> dict:
    """Point metrics, bootstrap CIs, and calibration diagnostics in one dict."""
    p, y = _arr(p, y)
    out = {"n": len(p), "base_rate": float(y.mean())}
    for name, fn in METRICS.items():
        out[name] = bootstrap_ci(fn, p, y, clusters=clusters, n_boot=n_boot, seed=seed).as_dict()
    out["murphy"] = murphy_decomposition(p, y)
    out["brier_skill_vs_climatology"] = skill_score(brier(p, y), y.mean() * (1 - y.mean()))
    return out
