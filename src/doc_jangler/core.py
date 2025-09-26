"""Core business logic for the DocJangler application."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import httpx
from firecrawl import FirecrawlApp

from .config import Settings


class ObjectiveFinderError(Exception):
    """Base exception for objective finding errors."""


class MissingConfigurationError(ObjectiveFinderError):
    """Raised when required configuration is not present."""


@dataclass(frozen=True)
class MappingResult:
    """Represents the outcome of mapping a website for relevant pages."""

    search_parameter: str
    links: List[str]


@dataclass(frozen=True)
class ObjectiveResult:
    """Represents the extracted data that satisfies the objective."""

    source_url: str
    data: Dict[str, Any]


ProgressCallback = Callable[[str, str], None]


class ObjectiveFinder:
    """Encapsulates the logic required to discover objective-related content."""

    def __init__(self, firecrawl_app: FirecrawlApp, settings: Settings) -> None:
        self._firecrawl_app = firecrawl_app
        self._settings = settings

    def ensure_model_available(self) -> None:
        """Checks that the OpenRouter model is reachable."""

        self._call_chat_completion([
            {"role": "user", "content": "test"},
        ])

    def find_relevant_pages(self, objective: str, url: str) -> MappingResult:
        """Identify relevant pages on a site for the provided objective."""

        search_parameter = self._suggest_search_parameter(objective)
        map_response = self._firecrawl_app.map(url, search=search_parameter)
        links = [link.url for link in getattr(map_response, "links", []) if getattr(link, "url", None)]
        return MappingResult(search_parameter=search_parameter, links=links)

    def _process_pages(
        self,
        pages: Sequence[str],
        objective: str,
        *,
        limit: Optional[int] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> Iterable[ObjectiveResult]:
        """Scrape and analyse pages, yielding results as they are found."""

        selected_pages = list(pages)
        if limit is not None:
            selected_pages = selected_pages[:limit]

        for link in selected_pages:
            if progress:
                progress("scrape", link)
            scrape_result = self._firecrawl_app.scrape(url=link)
            completion = self._analyse_scraped_content(objective, scrape_result.markdown)
            if progress:
                progress("model_response", completion)
            try:
                parsed = self._extract_structured_data(completion)
            except ValueError:
                if progress:
                    progress("parse_error", completion)
                continue

            if parsed is None:
                if progress:
                    progress("objective_not_met", link)
                continue

            if progress:
                progress("objective_met", link)
            yield ObjectiveResult(source_url=link, data=parsed)

    def find_objective_in_pages(
        self,
        pages: Sequence[str],
        objective: str,
        *,
        limit: int = 3,
        progress: Optional[ProgressCallback] = None,
    ) -> Optional[ObjectiveResult]:
        """Attempt to fulfil the objective by analysing the provided pages."""

        return next(
            self._process_pages(pages, objective, limit=limit, progress=progress),
            None,
        )

    def extract_metadata_from_pages(
        self,
        pages: Sequence[str],
        objective: str,
        *,
        limit: Optional[int] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> List[ObjectiveResult]:
        """Collect objective-aligned metadata for each provided page."""

        return list(self._process_pages(pages, objective, limit=limit, progress=progress))

    def _suggest_search_parameter(self, objective: str) -> str:
        map_prompt = (
            "The map function generates a list of URLs from a website and it accepts "
            "a search parameter. Based on the objective of: "
            f"{objective}, come up with a 1-2 word search parameter that will help "
            "us find the information we need. Only respond with 1-2 words nothing else."
        )

        suggestion = self._call_chat_completion([
            {"role": "user", "content": map_prompt},
        ])
        return suggestion.strip()

    def _analyse_scraped_content(self, objective: str, content: str) -> str:
        prompt = (
            "Given the following scraped content and objective, determine if the objective is met.\n"
            "If it is, extract the relevant information in a simple JSON format. \n"
            "If the objective is not met, respond with exactly 'Objective not met'.\n\n"
            "The JSON format should be:\n"
            "{\n"
            "    \"found\": true,\n"
            "    \"data\": {\n"
            "        // extracted information here\n"
            "    }\n"
            "}\n\n"
            "Important: Do not wrap the JSON in markdown code blocks. Just return the raw JSON. "
            "Make sure the JSON is valid and does not contain '...'.\n\n"
            f"Objective: {objective}\n"
            f"Scraped content: {content}"
        )

        return self._call_chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that extracts information from web pages. "
                        "Always respond in valid JSON format when information is found. Do not wrap "
                        "the JSON in markdown code blocks."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )

    def _extract_structured_data(self, model_response: str) -> Optional[Dict[str, Any]]:
        if model_response == "Objective not met":
            return None

        cleaned = model_response.strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            if len(parts) > 1:
                cleaned = parts[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON returned from model") from exc

        if isinstance(parsed, dict) and parsed.get("found"):
            data = parsed.get("data")
            if isinstance(data, dict):
                return data
            return {"data": data}

        raise ValueError("Model response missing expected structure")

    def _call_chat_completion(self, messages: Iterable[Dict[str, str]]) -> str:
        if not self._settings.openrouter_api_key:
            raise MissingConfigurationError("OPENROUTER_API_KEY is not configured.")

        response = httpx.post(
            self._settings.openrouter_endpoint,
            headers={
                "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._settings.openrouter_model,
                "messages": list(messages),
            },
            timeout=60.0,
        )

        response.raise_for_status()

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ObjectiveFinderError("Unexpected response format from OpenRouter") from exc


__all__ = [
    "MappingResult",
    "MissingConfigurationError",
    "ObjectiveFinder",
    "ObjectiveFinderError",
    "ObjectiveResult",
    "ProgressCallback",
]
