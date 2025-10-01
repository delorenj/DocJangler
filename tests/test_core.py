from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, List, Optional

import pytest

from doc_jangler.config import Settings
from doc_jangler.core import ObjectiveFinder

from .conftest import FakeFirecrawlApp


TEST_SETTINGS = Settings(
    firecrawl_api_key="fake",
    openrouter_api_key="fake",
    firecrawl_api_url="https://firecrawl.example.com",
    openrouter_endpoint="https://openrouter.example.com",
    openrouter_model="fake-model",
    redis_url="redis://localhost:6379/0",
    ollama_base_url="http://localhost:11434",
    ollama_embedding_model="nomic-embed-text",
)


def _build_finder(markdown_map: dict[str, str]) -> ObjectiveFinder:
    finder = ObjectiveFinder(
        firecrawl_app=FakeFirecrawlApp(markdown_map.keys(), markdown_map),
        settings=TEST_SETTINGS,
    )

    def _stub_analyse(self: ObjectiveFinder, objective: str, content: str) -> str:
        return content

    finder._analyse_scraped_content = MethodType(_stub_analyse, finder)

    finder._map_site = MethodType(_create_map_stub(finder._firecrawl_app), finder)
    return finder


def _create_map_stub(app: FakeFirecrawlApp):
    def _stub(
        self: ObjectiveFinder,
        url: str,
        search: Optional[str],
        *,
        debug=None,
    ) -> Any:
        data = app.map(url, search=search)
        if debug:
            count = len(self._extract_links_from_map_response(data))
            debug(f"  -> {count} link(s) returned")
        return data

    return _stub


class DictFirecrawlApp(FakeFirecrawlApp):
    def __init__(self, responses: dict[str | None, dict], markdown_map: dict[str, str] | None = None) -> None:
        super().__init__([], markdown_map or {})
        self._responses = responses
        self.calls: list[str | None] = []

    def map(self, url: str, search: str | None = None):  # type: ignore[override]
        self.calls.append(search)
        return self._responses.get(search, {"links": []})


def test_find_objective_in_pages_returns_first_result() -> None:
    finder = _build_finder(
        {
            "https://example.com/a": '{"found": true, "data": {"title": "A"}}',
            "https://example.com/b": '{"found": true, "data": {"title": "B"}}',
        }
    )

    result = finder.find_objective_in_pages(
        ["https://example.com/a", "https://example.com/b"],
        "sample objective",
    )

    assert result is not None
    assert result.source_url == "https://example.com/a"
    assert result.data == {"title": "A"}


def test_extract_metadata_from_pages_returns_all_results() -> None:
    finder = _build_finder(
        {
            "https://example.com/a": '{"found": true, "data": {"title": "A"}}',
            "https://example.com/b": '{"found": true, "data": {"title": "B"}}',
        }
    )

    results = finder.extract_metadata_from_pages(
        ["https://example.com/a", "https://example.com/b"],
        "sample objective",
    )

    assert [r.source_url for r in results] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_extract_metadata_skips_invalid_and_reports_progress() -> None:
    finder = _build_finder(
        {
            "https://example.com/a": "not json",
            "https://example.com/b": '{"found": true, "data": {"title": "B"}}',
            "https://example.com/c": "Objective not met",
        }
    )

    events: List[tuple[str, str]] = []

    def progress(event: str, payload: str) -> None:
        events.append((event, payload))

    results = finder.extract_metadata_from_pages(
        [
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
        ],
        "sample objective",
        progress=progress,
    )

    assert len(results) == 1
    assert results[0].source_url == "https://example.com/b"
    assert ("parse_error", "not json") in events
    assert ("objective_not_met", "https://example.com/c") in events


@pytest.mark.parametrize(
    "model_response, expected",
    [
        ('{"found": true, "data": {"x": 1}}', {"x": 1}),
        ("Objective not met", None),
        ('```json\n{"found": true, "data": {"x": 2}}\n```', {"x": 2}),
    ],
)
def test_extract_structured_data_behaviour(model_response: str, expected: dict | None) -> None:
    finder = _build_finder({})

    if expected is None:
        assert finder._extract_structured_data(model_response) is None
    else:
        assert finder._extract_structured_data(model_response) == expected


def test_extract_structured_data_invalid_json_raises() -> None:
    finder = _build_finder({})

    with pytest.raises(ValueError, match="Invalid JSON"):
        finder._extract_structured_data('{"found": true, "data": invalid}')


def test_find_relevant_pages_handles_dict_payload() -> None:
    firecrawl = DictFirecrawlApp(
        {
            "docs": {"links": [{"url": "https://example.com/a"}, {"href": "https://example.com/b"}]},
        }
    )
    finder = ObjectiveFinder(firecrawl, TEST_SETTINGS)
    finder._map_site = MethodType(_create_map_stub(firecrawl), finder)

    finder._suggest_search_parameter = MethodType(lambda self, obj: "docs", finder)  # type: ignore

    mapping = finder.find_relevant_pages("objective", "https://example.com")

    assert mapping.search_parameter == "docs"
    assert mapping.links == ["https://example.com/a", "https://example.com/b"]


def test_find_relevant_pages_falls_back_when_search_empty() -> None:
    responses = {
        "docs": {"links": []},
        "objective": {"links": []},
        "documentation": {"links": []},
        "": {"links": [{"url": "https://example.com/fallback"}]},
    }
    firecrawl = DictFirecrawlApp(responses)
    finder = ObjectiveFinder(firecrawl, TEST_SETTINGS)
    finder._map_site = MethodType(_create_map_stub(firecrawl), finder)

    finder._suggest_search_parameter = MethodType(lambda self, obj: "docs", finder)  # type: ignore

    mapping = finder.find_relevant_pages("objective", "https://example.com")

    assert mapping.links == ["https://example.com/fallback"]
    assert mapping.search_parameter == ""
    assert firecrawl.calls == ["docs", "objective", "documentation", ""]


def test_find_relevant_pages_uses_topic_hint() -> None:
    responses = {
        "docs": {"links": []},
        "templater": {"links": []},
        "Templater obsidian plugin": {
            "links": [{"url": "https://example.com/templater"}],
        },
    }
    firecrawl = DictFirecrawlApp(responses)
    finder = ObjectiveFinder(firecrawl, TEST_SETTINGS)
    finder._map_site = MethodType(_create_map_stub(firecrawl), finder)

    def stub(self: ObjectiveFinder, text: str) -> str:
        return "templater" if text == "Templater obsidian plugin" else "docs"

    finder._suggest_search_parameter = MethodType(stub, finder)  # type: ignore

    mapping = finder.find_relevant_pages(
        "objective",
        "https://example.com",
        topic="Templater obsidian plugin",
    )

    assert mapping.links == ["https://example.com/templater"]
    assert mapping.search_parameter == "Templater obsidian plugin"
    assert firecrawl.calls == ["docs", "templater", "Templater obsidian plugin"]


def test_find_relevant_pages_emits_debug_messages() -> None:
    responses = {
        "docs": {"links": []},
        "objective": {"links": [{"url": "https://example.com/hit"}]},
    }
    firecrawl = DictFirecrawlApp(responses)
    finder = ObjectiveFinder(firecrawl, TEST_SETTINGS)
    finder._map_site = MethodType(_create_map_stub(firecrawl), finder)

    finder._suggest_search_parameter = MethodType(lambda self, obj: "docs", finder)  # type: ignore

    messages: List[str] = []

    mapping = finder.find_relevant_pages(
        "objective",
        "https://example.com",
        debug=messages.append,
    )

    assert mapping.links == ["https://example.com/hit"]
    assert any("firecrawl.map search candidate:" in msg for msg in messages)
    assert any("->" in msg for msg in messages)


def test_map_site_uses_scrape_fallback(monkeypatch) -> None:
    finder = ObjectiveFinder(FakeFirecrawlApp([], {}), TEST_SETTINGS)

    def fake_post(url, headers, json, timeout):
        class Response:
            status_code = 200
            text = ""
            reason_phrase = "OK"
            request = SimpleNamespace(url=url)

            def raise_for_status(self) -> None:
                return None

            def json(self) -> Dict[str, Any]:
                return {"success": True}

        return Response()

    monkeypatch.setattr("doc_jangler.core.httpx.post", fake_post)

    fallback_data = {"data": {"links": [{"url": "https://example.com/fallback"}]}}
    calls: List[str] = []

    def fake_scrape_request(self, payload: Dict[str, Any], *, debug=None):
        calls.append(payload["url"])
        return fallback_data

    monkeypatch.setattr(ObjectiveFinder, "_scrape_request", fake_scrape_request)

    result = finder._map_site("https://example.com", search=None, debug=None)

    assert result == fallback_data
    assert calls == ["https://example.com"]
