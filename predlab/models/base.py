"""Forecaster interface and the per-forecast record every model must produce."""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from predlab.features.pointintime import Dataset

P_FLOOR, P_CEIL = 0.001, 0.999


@dataclass
class Predictions:
    """Batch output aligned with the dataset rows it was made for."""
    p: np.ndarray
    ci_low: np.ndarray | None = None
    ci_high: np.ndarray | None = None
    explanations: list[dict] = field(default_factory=list)
    flags: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class Forecast:
    ticker: str
    model_version: str
    p: float
    ci_low: float | None
    ci_high: float | None
    input_max_ts: pd.Timestamp
    decision_ts: pd.Timestamp
    dq_flags: tuple[str, ...]
    explanation: dict


class Forecaster(ABC):
    name: str = "base"
    #: True when ci_low/ci_high come from a defensible procedure (posterior or bootstrap).
    provides_ci: bool = False

    @abstractmethod
    def fit(self, ds: Dataset) -> "Forecaster": ...

    @abstractmethod
    def predict(self, ds: Dataset) -> Predictions: ...

    def params(self) -> dict:
        return {}

    @property
    def version(self) -> str:
        blob = json.dumps({"name": self.name, "params": self.params()}, sort_keys=True,
                          default=str)
        return f"{self.name}-{hashlib.sha256(blob.encode()).hexdigest()[:10]}"

    def forecasts(self, ds: Dataset) -> list[Forecast]:
        pr = self.predict(ds)
        out = []
        for i in range(len(ds)):
            m = ds.meta.iloc[i]
            flags = list(m["dq_flags"]) + (pr.flags[i] if pr.flags else [])
            if pr.ci_low is None:
                flags.append("no_ci")
            out.append(Forecast(
                ticker=m["ticker"], model_version=self.version, p=float(pr.p[i]),
                ci_low=None if pr.ci_low is None else float(pr.ci_low[i]),
                ci_high=None if pr.ci_high is None else float(pr.ci_high[i]),
                input_max_ts=m["input_max_ts"], decision_ts=m["decision_ts"],
                dq_flags=tuple(dict.fromkeys(flags)),
                explanation=pr.explanations[i] if pr.explanations else {}))
        return out


def clip_p(p) -> np.ndarray:
    """Keep probabilities off exact 0/1 so a single miss cannot produce infinite log loss."""
    return np.clip(np.asarray(p, dtype=float), P_FLOOR, P_CEIL)


def top_contributions(names, values, k: int = 3) -> dict:
    order = np.argsort(-np.abs(values))[:k]
    return {"top_features": [{"feature": names[j], "contribution": round(float(values[j]), 4)}
                             for j in order]}
