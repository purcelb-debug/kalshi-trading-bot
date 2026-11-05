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

    #: Scanner, Pricer, Risk and Desk Lead run on ``provider``. News and Skeptic need web
    #: search and run on ``search_provider``. Default: all free. Gemini answers, and with a
    #: TAVILY_API_KEY the searches are run by code (desk.search, free tier). Set
    #: ``search_provider=claude`` and Claude model names to use paid Claude search instead.
    provider: Literal["claude", "gemini"] = "gemini"
    model: str = "gemini-3.8-flash"
    #: Tried in order when a Gemini model's free daily quota is used up (quotas are per model).
    gemini_models: list[str] = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite",
                                "gemini-3.1-flash-lite"]
    #: Free backup after every Gemini model is out for the day (Groq: ~1,000 requests a day,
    #: 8,000 tokens a minute). Used only when GROQ_API_KEY is set.
    groq_models: list[str] = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    #: Groq's strongest model is tried ahead of these Gemini models (weaker than gpt-oss-120b).
    groq_before: list[str] = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    groq_api_key: str = Field(default="", validation_alias=AliasChoices(
        "GROQ_API_KEY", "groq_api_key"))
    search_provider: Literal["claude", "gemini"] = "gemini"
    search_model: str = "gemini-3.8-flash"     # News Checker
    skeptic_model: str = "gemini-3.8-flash"    # Skeptic, also on search_provider
    effort: str = "medium"           # Claude effort / Gemini thinking_level
    skeptic_effort: str = "high"
    gemini_api_key: str = Field(default="", validation_alias=AliasChoices(
        "GEMINI_API_KEY", "gemini_api_key"))
    search_max_uses: int = Field(default=3, ge=1)        # Claude only
    #: Free search (Tavily, 1,000 searches a month): the key, queries per searching role,
    #: and a daily cap so the month's credits last (30 x 30 days = 900).
    tavily_api_key: str = Field(default="", validation_alias=AliasChoices(
        "TAVILY_API_KEY", "tavily_api_key"))
    searches_per_role: int = Field(default=3, ge=1)
    search_daily_cap: int = Field(default=30, ge=0)
    #: Tavily's free plan (calendar month assumed).
    search_monthly_cap: int = Field(default=1000, ge=0)
    search_usage_path: str = str(ROOT / "data" / "desk_search_usage.json")
    #: Time limits so one slow call can't eat the job's step: per LLM request, and per meeting
    #: (the meeting stops after the role that crosses it and is recorded as aborted).
    call_timeout_s: float = Field(default=180.0, gt=0)
    meeting_deadline_s: float = Field(default=600.0, gt=0)
    role_timeout_s: float = Field(default=240.0, gt=0)   # hard limit per role, all retries
    max_meetings_per_cycle: int = Field(default=1, ge=1)
    #: Hard caps per UTC day across ALL desk deployments combined: meetings, and estimated
    #: LLM spend in dollars (measured token use x ``prices``, plus web searches).
    max_meetings_per_day: int = Field(default=24, ge=1)
    daily_budget_usd: float = Field(default=1.0, gt=0)
    #: $ per million tokens (input, output). Unknown models are priced as Opus to be safe.
    prices: dict[str, tuple[float, float]] = {
        "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
        "gemini-3.8-flash": (0.0, 0.0), "gemini-3.6-flash": (0.0, 0.0),   # free-tier key
        "gemini-3.5-flash-lite": (0.0, 0.0), "gemini-3.1-flash-lite": (0.0, 0.0),
        "openai/gpt-oss-120b": (0.0, 0.0), "qwen/qwen3.8-27b": (0.0, 0.0)}
    web_search_usd: float = 0.01
    #: Broad scan (desk@ALL): markets closing within ``scan_horizon_h`` hours.
    scan_horizon_h: float = Field(default=48.0, gt=0)
    scan_min_hours: float = 1.0      # skips 15-minute coin flips a meeting can't keep up with
    scan_min_mid: float = 0.05
    scan_max_mid: float = 0.95
    scan_max_spread: float = 0.06
    scan_min_volume: float = 100.0
    scan_per_event: int = Field(default=2, ge=1)
    scan_max_rows: int = Field(default=20, ge=1)
    #: Prefixes excluded from broad scan (sports/esports where LLM lacks edge and triggers clustering).
    scan_exclude_prefixes: list[str] = [
        "KXLOL", "KXCS2", "KXDOTA", "KXBRASIL", "KXAPFD", "KXUCL", "KXNCAAMB",
        "KXNCAAWB", "KXNBB", "KXDIMAYOR", "KXECULP", "KXNHL", "KXNBA", "KXNFL", "KXMLB",
        "KXLIGA", "KXEFL", "KXEPL", "KXLALIGA", "KXSERIEA", "KXBUNDESLIGA", "KXLIGUE1",
        "KXMLS", "KXUEL", "KXUECL", "KXAFCCL", "KXEUROCUP", "KXNCAAF", "KXFIBA", "KXKHL"
    ]
    #: The broad desk sees a market again this many hours after it was last shown (never
    #: one it bought), so a market first seen without data gets another look.
    scan_reshow_hours: float = Field(default=4.0, gt=0)
    #: Annotate scanned markets with live spot values from Coinbase/Yahoo/NWS.
    scan_spot: bool = Field(default=True, description="Annotate scanned markets with live spot values")
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
