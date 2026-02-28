"""StreamChat AI — Configuration & Settings."""

from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    app_name: str = "StreamChat AI"
    app_version: str = "1.0.0"
    debug: bool = False

    # LLM Providers
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = Field(default="", description="Gemini API key")
    gemini_model: str = "gemini-1.5-flash"

    # Default model
    default_model: str = "gpt-4o-mini"

    # Auth
    api_keys: str = Field(
        default="demo-key-123",
        description="Comma-separated list of valid API keys",
    )

    # Rate limiting
    rate_limit_requests: int = 60  # per minute
    rate_limit_tokens: int = 100000  # per minute

    # History
    db_path: str = "./data/chat_history.db"

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def valid_api_keys(self) -> set[str]:
        return set(k.strip() for k in self.api_keys.split(",") if k.strip())


@lru_cache()
def get_settings() -> Settings:
    return Settings()
