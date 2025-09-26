from __future__ import annotations

from types import MethodType
from typing import List

import pytest

from doc_jangler.config import Settings
from doc_jangler.core import ObjectiveFinder

from .conftest import FakeFirecrawlApp


def _build_finder(markdown_map: dict[str, str]) -> ObjectiveFinder:
    settings = Settings(
        firecrawl_api_key="fake",
        openrouter_api_key="fake",
        firecrawl_api_url="https://firecrawl.example.com",
        openrouter_endpoint="https://openrouter.example.com",
        openrouter_model="fake-model",
    )
    finder = ObjectiveFinder(
        firecrawl_app=FakeFirecrawlApp(markdown_map.keys(), markdown_map),
        settings=settings,
    )

    def _stub_analyse(self: ObjectiveFinder, objective: str, content: str) -> str:
        return content

    finder._analyse_scraped_content = MethodType(_stub_analyse, finder)
    return finder


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
