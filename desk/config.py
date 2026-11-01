"""Desk settings, with the DESK_ prefix, from the repository ``.env`` then ``desk/.env``
(both git-ignored; the later file wins)."""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent


class DeskSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DESK_",
                                      env_file=(str(ROOT / ".env"), str(PKG / ".env")),
                                      env_file_encoding="utf-8", extra="ignore",
                                      env_ignore_empty=True, populate_by_name=True)

    #: Which LLM runs the six roles. Set the model names to match when switching.
    provider: Literal["claude", "gemini"] = "claude"
    model: str = "claude-opus-5-5"
    skeptic_model: str = "claude-opus-5-5"   # the Skeptic can get its own model
    effort: str = "medium"           # Claude effort / Gemini thinking_level
    skeptic_effort: str = "high"
    gemini_api_key: str = Field(default="", validation_alias=AliasChoices(
        "GEMINI_API_KEY", "gemini_api_key"))
    search_max_uses: int = Field(default=5, ge=1)        # Claude only
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
