"""FastMCP tool definitions for DocJangler."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import httpx
from fastmcp.tool import Tool
from firecrawl import FirecrawlApp

from .config import Settings, get_firecrawl_app, get_settings
from .core import (
    MappingResult,
    MissingConfigurationError,
    ObjectiveFinder,
    ObjectiveFinderError,
    ObjectiveResult,
)


class DocJanglerTools:
    """Expose DocJangler capabilities as FastMCP tools."""

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        firecrawl_app: Optional[FirecrawlApp] = None,
    ) -> None:
        self._settings = settings
        self._firecrawl_app = firecrawl_app

    def _resolve_dependencies(self) -> Tuple[FirecrawlApp, Settings]:
        settings = self._settings or get_settings()
        app = self._firecrawl_app or get_firecrawl_app(settings)
        return app, settings

    @Tool(
        name="scrape_and_find_objective",
        description="Scrapes a URL and extracts information relevant to the provided objective.",
    )
    def scrape_and_find_objective(self, url: str, objective: str) -> Dict[str, Any]:
        """Find objective-specific information from a site."""

        try:
            firecrawl_app, settings = self._resolve_dependencies()
            finder = ObjectiveFinder(firecrawl_app=firecrawl_app, settings=settings)
            finder.ensure_model_available()
            mapping: MappingResult = finder.find_relevant_pages(objective, url)

            if not mapping.links:
                return {"status": "error", "message": "No relevant pages found."}

            result: Optional[ObjectiveResult] = finder.find_objective_in_pages(
                mapping.links,
                objective,
                limit=3,
            )

            if not result:
                return {"status": "error", "message": "Objective could not be fulfilled."}

            return {
                "status": "success",
                "source_url": result.source_url,
                "data": result.data,
            }
        except RuntimeError as exc:
            return {"status": "error", "message": str(exc)}
        except (MissingConfigurationError, ObjectiveFinderError) as exc:
            return {"status": "error", "message": str(exc)}
        except httpx.HTTPError as exc:
            return {"status": "error", "message": f"HTTP error: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": f"Unexpected error: {exc}"}


__all__ = ["DocJanglerTools"]
