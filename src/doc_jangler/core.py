"""Core business logic for the DocJangler application."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urljoin, urlparse

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

        self._call_chat_completion(
            [
                {"role": "user", "content": "test"},
            ]
        )

    def find_relevant_pages(
        self,
        objective: str,
        url: str,
        *,
        topic: Optional[str] = None,
        debug: Optional[Callable[[str], None]] = None,
    ) -> MappingResult:
        """Identify relevant pages on a site for the provided objective."""

        def add_candidate(value: Optional[str], bucket: List[Optional[str]]) -> None:
            if value is None:
                if None not in bucket:
                    bucket.append(None)
                return

            trimmed = value.strip()
            if trimmed:
                if trimmed not in bucket:
                    bucket.append(trimmed)
            else:
                if "" not in bucket:
                    bucket.append("")

        suggested_search = self._suggest_search_parameter(objective)
        attempts: List[Optional[str]] = []
        add_candidate(suggested_search, attempts)

        if topic and topic.strip() and topic.strip() != objective.strip():
            topic_suggestion = self._suggest_search_parameter(topic)
            add_candidate(topic_suggestion, attempts)
            add_candidate(topic, attempts)

        add_candidate(objective, attempts)
        add_candidate("documentation", attempts)
        add_candidate("docs", attempts)
        add_candidate("", attempts)
        add_candidate(None, attempts)

        for attempt in attempts:
            if debug:
                label = (
                    attempt
                    if attempt not in (None, "")
                    else ("<none>" if attempt is None else "<empty>")
                )
                debug(f"firecrawl.map search candidate: {label}")
            map_response = self._map_site(url, attempt, debug=debug)
            links = self._extract_links_from_map_response(map_response)
            if debug:
                debug(f"  -> {len(links)} link(s) returned")
            if links:
                search_value = (
                    attempt if isinstance(attempt, str) else (suggested_search or "")
                )
                return MappingResult(search_parameter=search_value, links=links)

        if debug:
            debug("No links discovered across all search candidates")
        return MappingResult(search_parameter=suggested_search or "", links=[])

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
            completion = self._analyse_scraped_content(
                objective, scrape_result.markdown
            )
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

        return list(
            self._process_pages(pages, objective, limit=limit, progress=progress)
        )

    def collect_document_links(
        self,
        url: str,
        *,
        topic: Optional[str] = None,
        debug: Optional[Callable[[str], None]] = None,
    ) -> List[str]:
        """Return a normalised list of documentation links for the site."""

        scraped = self._scrape_links(url, debug=debug)
        if scraped:
            links = self._extract_links_from_map_response(scraped)
            filtered = self._filter_links(url, links)
            if filtered:
                return filtered

        mapping = self.find_relevant_pages(
            topic or "documentation",
            url,
            topic=topic,
            debug=debug,
        )
        return self._filter_links(url, mapping.links)

    def scrape_page_markdown(
        self,
        url: str,
        *,
        debug: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """Scrape a page and return markdown-only content."""

        payload: Dict[str, Any] = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True,
            "removeBase64Images": True,
            "waitFor": 2000,
            "timeout": 30000,
        }

        response = self._scrape_request(payload, debug=debug)
        if not response:
            return None

        if isinstance(response, dict):
            if isinstance(response.get("markdown"), str):
                return response["markdown"].strip()
            nested = response.get("data")
            if isinstance(nested, dict):
                markdown = nested.get("markdown") or nested.get("content")
                if isinstance(markdown, str):
                    return markdown.strip()

        return None

    def _map_site(
        self,
        url: str,
        search: Optional[str],
        *,
        debug: Optional[Callable[[str], None]] = None,
    ) -> Any:
        payload: Dict[str, Any] = {
            "url": url,
            "formats": ["links"],
            "onlyMainContent": False,
            "removeBase64Images": False,
            "waitFor": 2000,
            "timeout": 30000,
        }

        if search is not None:
            payload["search"] = search

        if not self._settings.firecrawl_api_key:
            raise MissingConfigurationError("FIRECRAWL_API_KEY is not configured.")

        base_url = self._settings.firecrawl_api_url.rstrip("/")
        candidate_paths = [
            ("/v1/map", "v1"),
            ("/v0/map", "legacy"),
            ("/map", "legacy"),
        ]
        last_exception: Optional[Exception] = None

        def build_payload(version: str) -> Dict[str, Any]:
            base: Dict[str, Any] = {"url": url, "timeout": 30000}
            if search:
                base["search"] = search

            if version == "legacy":
                base.update(
                    {
                        "formats": ["links"],
                        "onlyMainContent": False,
                        "removeBase64Images": False,
                        "waitFor": 2000,
                    }
                )

            return base

        for path, version in candidate_paths:
            endpoint = f"{base_url}{path}"
            try:
                payload = build_payload(version)
                if debug:
                    debug(
                        "  -> "
                        + f"Attempting Firecrawl {endpoint} with payload keys: {sorted(payload.keys())}"
                    )
                response = httpx.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self._settings.firecrawl_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60.0,
                )

                if response.status_code in {400, 404}:
                    if debug:
                        debug(
                            f"  -> Firecrawl responded with {response.status_code} for {endpoint}:"
                        )
                        debug(f"     {response.text[:200] or '<empty response>'}")
                    last_exception = httpx.HTTPStatusError(
                        response.reason_phrase,
                        request=response.request,
                        response=response,
                    )
                    continue

                response.raise_for_status()

                try:
                    data = response.json()
                except ValueError as exc:  # pragma: no cover - guard against malformed data
                    raise ObjectiveFinderError(
                        "Unexpected response from Firecrawl map endpoint"
                    ) from exc

                links = self._extract_links_from_map_response(data)
                if links:
                    if debug:
                        debug(f"  -> Using response from {endpoint}")
                    return data

                if debug:
                    debug("  -> No links returned; attempting scrape fallback")
                fallback = self._scrape_links(url, debug=debug)
                if fallback:
                    return fallback
                if debug:
                    debug("  -> Scrape fallback produced no links")
                last_exception = ObjectiveFinderError(
                    "No links returned from Firecrawl map or scrape"
                )
                continue
            except httpx.HTTPStatusError as exc:
                if (
                    exc.response is not None
                    and exc.response.status_code in {400, 404}
                ):
                    if debug:
                        body = exc.response.text[:200] if exc.response.text else "<empty response>"
                        debug(
                            f"  -> Firecrawl responded with {exc.response.status_code} for {endpoint}:"
                        )
                        debug(f"     {body}")
                    last_exception = exc
                    continue
                raise
            except httpx.HTTPError as exc:  # pragma: no cover - network anomalies
                if debug:
                    debug(f"  -> Firecrawl request failed for {endpoint}: {exc}")
                last_exception = exc
                continue

        if last_exception:
            raise last_exception
        raise ObjectiveFinderError("Unable to reach Firecrawl map endpoint")

    def _scrape_links(
        self,
        url: str,
        *,
        debug: Optional[Callable[[str], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "url": url,
            "formats": ["links"],
            "onlyMainContent": False,
            "removeBase64Images": False,
            "waitFor": 2000,
            "timeout": 30000,
        }

        return self._scrape_request(payload, debug=debug)

    def _scrape_request(
        self,
        payload: Dict[str, Any],
        *,
        debug: Optional[Callable[[str], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        base_url = self._settings.firecrawl_api_url.rstrip("/")
        candidate_paths = ["/v1/scrape", "/api/v1/scrape"]

        for path in candidate_paths:
            endpoint = f"{base_url}{path}"
            try:
                if debug:
                    debug(
                        "  -> "
                        + f"Attempting Firecrawl scrape {endpoint} with payload keys: {sorted(payload.keys())}"
                    )
                response = httpx.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self._settings.firecrawl_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60.0,
                )

                if response.status_code in {400, 404}:
                    if debug:
                        debug(
                            f"  -> Firecrawl responded with {response.status_code} for {endpoint}:"
                        )
                        debug(f"     {response.text[:200] or '<empty response>'}")
                    continue

                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:  # pragma: no cover - network anomalies
                if debug:
                    debug(f"  -> Firecrawl scrape request failed for {endpoint}: {exc}")
                continue

        return None

    def _extract_links_from_map_response(self, map_response: Any) -> List[str]:
        """Normalise the Firecrawl map payload into a list of URLs."""

        raw_links: Any = None

        if isinstance(map_response, dict):
            raw_links = map_response.get("links")
            if not raw_links:
                data = map_response.get("data")
                if isinstance(data, dict):
                    raw_links = data.get("links") or data.get("data")
        else:
            raw_links = getattr(map_response, "links", None)

        urls: List[str] = []
        seen = set()

        for entry in raw_links or []:
            url: Optional[str] = None
            if isinstance(entry, str):
                url = entry
            elif isinstance(entry, dict):
                url = entry.get("url") or entry.get("href")
            else:
                url = getattr(entry, "url", None)

            if url and url not in seen:
                seen.add(url)
                urls.append(url)

        return urls

    def _filter_links(self, base_url: str, links: Iterable[str]) -> List[str]:
        parsed_base = urlparse(base_url)
        filtered: List[str] = []
        seen: set[str] = set()

        for link in links:
            absolute = urljoin(base_url, link)
            parsed = urlparse(absolute)
            if parsed.scheme not in {"http", "https"}:
                continue
            if parsed.netloc != parsed_base.netloc:
                continue
            normalised = absolute.split("#", 1)[0].rstrip("/") or absolute
            if normalised not in seen:
                seen.add(normalised)
                filtered.append(normalised)

        return filtered

    def _suggest_search_parameter(self, objective: str) -> str:
        map_prompt = (
            "The map function generates a list of URLs from a website and it accepts "
            "a search parameter. Based on the objective of: "
            f"{objective}, come up with a 1-2 word search parameter that will help "
            "us find the information we need. Only respond with 1-2 words nothing else."
        )

        suggestion = self._call_chat_completion(
            [
                {"role": "user", "content": map_prompt},
            ]
        )
        return suggestion.strip()

    def _analyse_scraped_content(self, objective: str, content: str) -> str:
        prompt = (
            "Given the following scraped content and objective, determine if the objective is met.\n"
            "If it is, extract the relevant information in a simple JSON format. \n"
            "If the objective is not met, respond with exactly 'Objective not met'.\n\n"
            "The JSON format should be:\n"
            "{\n"
            '    "found": true,\n'
            '    "data": {\n'
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
            raise ObjectiveFinderError(
                "Unexpected response format from OpenRouter"
            ) from exc


__all__ = [
    "MappingResult",
    "MissingConfigurationError",
    "ObjectiveFinder",
    "ObjectiveFinderError",
    "ObjectiveResult",
    "ProgressCallback",
]
