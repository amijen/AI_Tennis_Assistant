"""
Central configuration — reads .env once, validates types, used everywhere.
"""

from pydantic import Field, AliasChoices, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All environment variables are loaded automatically from .env.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM (Groq) ────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"

    # ── Embeddings (Local) ────────────────────────────────
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    # ── Database ───────────────────────────────────────────
    database_url: str = Field(
        validation_alias=AliasChoices("DATABASE_URL", "DB_URL")
    )

    # ── API / CORS ────────────────────────────────────────
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    @computed_field
    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # ── App ───────────────────────────────────────────────
    app_env: str = "development"

    @computed_field
    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"


# Create ONE instance that the whole app imports
settings = Settings()