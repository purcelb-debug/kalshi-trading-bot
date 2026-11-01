"""Desk settings, from ``desk/.env`` (git-ignored) with the DESK_ prefix."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent


class DeskSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DESK_", env_file=str(PKG / ".env"),
                                      env_file_encoding="utf-8", extra="ignore",
                                      env_ignore_empty=True)

    model: str = "claude-opus-5-5"
    skeptic_model: str = "claude-opus-5-5"   # the Skeptic can get its own model
    effort: str = "medium"
    skeptic_effort: str = "high"
    search_max_uses: int = Field(default=5, ge=1)
    max_meetings_per_day: int = Field(default=4, ge=1)   # hard cost cap
    max_picks: int = Field(default=3, ge=1)               # markets the Scanner may pick
    max_buys: int = Field(default=3, ge=0)                # BUYs kept per meeting
    p_floor: float = Field(default=0.02, gt=0, lt=0.5)    # desk p_yes clipped to [floor, 1-floor]
    #: Pull desk probabilities toward the market mid (as for fitted models). There is no dev
    #: data to fit kappa on, so this is a fixed, conservative setting.
    shrink: float = Field(default=0.5, ge=0.0, le=1.0)
    #: A fitted deployment whose latest forecasts are shown to the desk as one more input.
    quant_deployment: str = ""
    prompts_path: str = str(PKG / "prompts.json")
    report_dir: str = str(ROOT / "reports")
