"""Configuration helpers for DocJangler."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Optional

from dotenv import load_dotenv
from firecrawl import FirecrawlApp

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Represents configuration required by the application."""

    firecrawl_api_key: Optional[str]
    openrouter_api_key: Optional[str]
    firecrawl_api_url: str
    openrouter_endpoint: str
    openrouter_model: str
    redis_url: str
    ollama_base_url: str
    ollama_embedding_model: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from environment variables (cached)."""

    return Settings(
        firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        firecrawl_api_url=os.getenv("FIRECRAWL_API_URL", "https://firecrawl.delo.sh"),
        openrouter_endpoint=os.getenv(
            "OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions"
        ),
        openrouter_model=os.getenv("DOCJANGLER_MODEL", "moonshotai/kimi-k2"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_embedding_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    )


@lru_cache(maxsize=1)
def _get_cached_firecrawl_app() -> FirecrawlApp:
    settings = get_settings()
    if not settings.firecrawl_api_key:
        raise RuntimeError("FIRECRAWL_API_KEY is not configured.")
    return FirecrawlApp(api_key=settings.firecrawl_api_key, api_url=settings.firecrawl_api_url)


def get_firecrawl_app(settings: Optional[Settings] = None) -> FirecrawlApp:
    """Return a FirecrawlApp instance using provided or cached settings."""

    if settings is None:
        return _get_cached_firecrawl_app()

    if not settings.firecrawl_api_key:
        raise RuntimeError("FIRECRAWL_API_KEY is not configured.")

    return FirecrawlApp(api_key=settings.firecrawl_api_key, api_url=settings.firecrawl_api_url)


__all__ = ["Settings", "get_firecrawl_app", "get_settings"]
