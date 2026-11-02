"""Notifier settings, read from ``notify/.env`` (git-ignored). Blank token = notifier off."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent


class NotifySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PKG / ".env"), env_file_encoding="utf-8",
                                      extra="ignore", env_ignore_empty=True)

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    stale_after_hours: float = 2.0          # warn when the loop report is older than this
    report_dir: str = str(ROOT / "reports")
    #: Owner's success baseline (2026-11-02): the account must make at least this many
    #: dollars per day. Reported daily and weekly; it never changes how the bot trades.
    daily_target_usd: float = 8.0
    executor_data_dir: str = str(ROOT / "executor" / "data")
    state_path: str = str(ROOT / "data" / "notify_state.json")

    @property
    def enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)
