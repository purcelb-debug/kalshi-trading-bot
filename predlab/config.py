"""Settings loaded from environment / .env. No exchange credentials exist here by design."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PREDLAB_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    market_db_url: str = "sqlite:///data/market.db"
    sim_db_url: str = "sqlite:///data/sim.db"
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_reads_per_sec: float = 10.0
    ncei_base_url: str = "https://www.ncei.noaa.gov/access/services/data/v1"
    fred_base_url: str = "https://api.stlouisfed.org/fred"
    fred_api_key: str = ""
    user_agent: str = "predlab-research"
    log_level: str = "INFO"
    http_timeout_s: float = 30.0
    http_max_retries: int = 5
    # Read via its conventional name, without the PREDLAB_ prefix.
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")

    def ensure_sqlite_dirs(self) -> None:
        for url in (self.market_db_url, self.sim_db_url):
            if url.startswith("sqlite:///"):
                Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
