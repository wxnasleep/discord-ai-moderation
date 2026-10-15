"""Application configuration loaded from environment variables."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    bot_token: str
    model_path: Path
    data_path: Path
    moderation_channel_id: int | None
    guild_id: int | None
    warning_threshold: int
    severe_threshold: int
    warnings_before_timeout: int
    warning_timeout_minutes: int
    severe_timeout_minutes: int
    log_level: str


def _optional_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    return int(value) if value else None


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")

    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN is missing. Copy .env.example to .env.")

    warning_threshold = int(os.getenv("WARNING_THRESHOLD", "6"))
    severe_threshold = int(os.getenv("SEVERE_THRESHOLD", "10"))
    if not 1 <= warning_threshold < severe_threshold <= 10:
        raise ValueError("Thresholds must satisfy 1 <= WARNING_THRESHOLD < SEVERE_THRESHOLD <= 10.")

    model_value = os.getenv("MODEL_PATH", "models/toxicity_model.joblib")
    data_value = os.getenv("DATA_PATH", "data/moderation.sqlite3")
    model_path = Path(model_value)
    data_path = Path(data_value)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    warning_timeout_minutes = int(os.getenv("WARNING_TIMEOUT_MINUTES", "60"))
    severe_timeout_minutes = int(os.getenv("SEVERE_TIMEOUT_MINUTES", "180"))
    warnings_before_timeout = int(os.getenv("WARNINGS_BEFORE_TIMEOUT", "3"))
    if not 1 <= warning_timeout_minutes <= 10080 or not 1 <= severe_timeout_minutes <= 10080:
        raise ValueError("Timeout duration must be between 1 minute and 7 days.")
    if warnings_before_timeout < 1:
        raise ValueError("WARNINGS_BEFORE_TIMEOUT must be at least 1.")

    return Settings(
        bot_token=token,
        model_path=model_path,
        data_path=data_path,
        moderation_channel_id=_optional_int("MODERATION_CHANNEL_ID"),
        guild_id=_optional_int("DISCORD_GUILD_ID"),
        warning_threshold=warning_threshold,
        severe_threshold=severe_threshold,
        warnings_before_timeout=warnings_before_timeout,
        warning_timeout_minutes=warning_timeout_minutes,
        severe_timeout_minutes=severe_timeout_minutes,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
