from __future__ import annotations

from types import SimpleNamespace

import pytest

from doc_jangler import mcp_tools
from doc_jangler.config import Settings
from doc_jangler.core import MappingResult, ObjectiveResult


class SuccessFinder:
    def __init__(self, firecrawl_app, settings):  # noqa: D401 - signature dictated by production code
        self._mapping = MappingResult(
            search_parameter="docs",
            links=["https://example.com/docs"],
        )
        self._result = ObjectiveResult(
            source_url="https://example.com/docs",
            data={"result": "ok"},
        )

    def ensure_model_available(self) -> None:  # pragma: no cover - nothing to do
        return None

    def find_relevant_pages(self, objective: str, url: str) -> MappingResult:
        return self._mapping

    def find_objective_in_pages(self, pages, objective, *, limit=3):
        return self._result


class EmptyFinder(SuccessFinder):
    def find_relevant_pages(self, objective: str, url: str) -> MappingResult:
        return MappingResult(search_parameter="docs", links=[])


class ErrorFinder(SuccessFinder):
    def ensure_model_available(self) -> None:
        raise ValueError("boom")


def _make_tools() -> mcp_tools.DocJanglerTools:
    settings = Settings(
        firecrawl_api_key="fake-firecrawl",
        openrouter_api_key="fake-openrouter",
        firecrawl_api_url="https://firecrawl.example.com",
        openrouter_endpoint="https://openrouter.example.com",
        openrouter_model="fake-model",
    )
    return mcp_tools.DocJanglerTools(settings=settings, firecrawl_app=SimpleNamespace())


def test_scrape_and_find_objective_success(monkeypatch) -> None:
    monkeypatch.setattr(mcp_tools, "ObjectiveFinder", SuccessFinder)

    tools = _make_tools()
    payload = tools.scrape_and_find_objective("https://example.com", "topic")

    assert payload == {
        "status": "success",
        "source_url": "https://example.com/docs",
        "data": {"result": "ok"},
    }


def test_scrape_and_find_objective_no_links(monkeypatch) -> None:
    monkeypatch.setattr(mcp_tools, "ObjectiveFinder", EmptyFinder)

    tools = _make_tools()
    payload = tools.scrape_and_find_objective("https://example.com", "topic")

    assert payload == {"status": "error", "message": "No relevant pages found."}


def test_scrape_and_find_objective_handles_exception(monkeypatch) -> None:
    monkeypatch.setattr(mcp_tools, "ObjectiveFinder", ErrorFinder)

    tools = _make_tools()
    payload = tools.scrape_and_find_objective("https://example.com", "topic")

    assert payload["status"] == "error"
    assert "boom" in payload["message"]
