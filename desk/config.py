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

    #: Scanner, Pricer, Risk and Desk Lead run on ``provider`` (no web search needed, so a
    #: free Gemini key is enough). News and Skeptic need web search and run on
    #: ``search_provider``. Set the model names to match each provider.
    provider: Literal["claude", "gemini"] = "gemini"
    model: str = "gemini-3.8-flash"
    search_provider: Literal["claude", "gemini"] = "claude"
    search_model: str = "claude-sonnet-5-5"    # News Checker
    skeptic_model: str = "claude-sonnet-5-5"   # Skeptic, also on search_provider
    effort: str = "medium"           # Claude effort / Gemini thinking_level
    skeptic_effort: str = "high"
    gemini_api_key: str = Field(default="", validation_alias=AliasChoices(
        "GEMINI_API_KEY", "gemini_api_key"))
    search_max_uses: int = Field(default=5, ge=1)        # Claude only
    #: Hard caps per UTC day across ALL desk deployments combined: meetings, and estimated
    #: LLM spend in dollars (measured token use x ``prices``, plus web searches).
    max_meetings_per_day: int = Field(default=24, ge=1)
    daily_budget_usd: float = Field(default=1.0, gt=0)
    #: $ per million tokens (input, output). Unknown models are priced as Opus to be safe.
    prices: dict[str, tuple[float, float]] = {
        "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
        "gemini-3.8-flash": (0.0, 0.0)}      # free-tier key
    web_search_usd: float = 0.01
    #: Broad scan (desk@ALL): markets closing within ``scan_horizon_h`` hours.
    scan_horizon_h: float = Field(default=48.0, gt=0)
    scan_min_hours: float = 1.0      # skips 15-minute coin flips a meeting can't keep up with
    scan_min_mid: float = 0.05
    scan_max_mid: float = 0.95
    scan_max_spread: float = 0.06
    scan_min_volume: float = 100.0
    scan_per_event: int = Field(default=2, ge=1)
    scan_max_rows: int = Field(default=40, ge=1)
    #: Exit levels from the desk are clipped to these price bounds (price of the side held).
    exit_min_gap: float = Field(default=0.03, gt=0)
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
