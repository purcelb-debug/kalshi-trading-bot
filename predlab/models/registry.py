"""Every model the CLI and research loop can name."""

from predlab.models.advanced import Calibrated, MarketTrajectory, Stacked
from predlab.models.baselines import MODEL_FACTORIES as BASELINES
from predlab.models.baselines import LightGBM, Logistic, MarketImplied

ADVANCED = {
    "market_trajectory": MarketTrajectory,
    "market_platt": lambda: Calibrated(MarketImplied, "platt"),
    "market_isotonic": lambda: Calibrated(MarketImplied, "isotonic"),
    "logistic_l2_platt": lambda: Calibrated(lambda: Logistic("l2", n_boot=0), "platt"),
    "lightgbm_isotonic": lambda: Calibrated(LightGBM, "isotonic"),
    "stacked": lambda: Stacked({
        "market_implied": MarketImplied,
        "logistic_l2": lambda: Logistic("l2", n_boot=0),
        "lightgbm": LightGBM,
        "market_trajectory": MarketTrajectory,
    }),
}

ALL_MODELS = {**BASELINES, **ADVANCED}
#: models that need market-price features to be meaningful
NEEDS_MARKET = {"market_trajectory", "market_platt", "market_isotonic", "stacked"}
