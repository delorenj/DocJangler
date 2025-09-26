from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

import respx
from httpx import Response
from typer.testing import CliRunner

from doc_jangler import cli as cli_module
from doc_jangler.cli import typer_app
from doc_jangler.config import Settings


class FakeFirecrawlApp:
    def __init__(self) -> None:
        self._links = ["https://example.com/docs"]

    def map(self, url: str, search: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(links=[SimpleNamespace(url=link) for link in self._links])

    def scrape(self, url: str) -> SimpleNamespace:
        return SimpleNamespace(markdown="Example documentation content")


runner = CliRunner()


@respx.mock
def test_jangle_docs_success(monkeypatch) -> None:
    fake_settings = Settings(
        firecrawl_api_key="fake-firecrawl",
        openrouter_api_key="fake-openrouter",
        firecrawl_api_url="https://firecrawl.example.com",
        openrouter_endpoint="https://openrouter.example.com",
        openrouter_model="fake-model",
    )

    monkeypatch.setattr(cli_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(cli_module, "get_firecrawl_app", lambda settings: FakeFirecrawlApp())
    monkeypatch.setattr(cli_module.ObjectiveFinder, "ensure_model_available", lambda self: None)

    openrouter_route = respx.post(fake_settings.openrouter_endpoint)
    openrouter_route.mock(
        side_effect=[
            Response(
                200,
                json={"choices": [{"message": {"content": "docs"}}]},
                request=httpx.Request("POST", fake_settings.openrouter_endpoint),
            ),
            Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '{"found": true, "data": {"result": "success"}}'
                            }
                        }
                    ]
                },
                request=httpx.Request("POST", fake_settings.openrouter_endpoint),
            ),
        ]
    )

    result = runner.invoke(
        typer_app,
        [
            "jangle-docs",
            "my-topic",
            "--site",
            "https://example.com",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Objective successfully found" in result.stdout

    json_start = result.stdout.index("{")
    payload = json.loads(result.stdout[json_start:])
    assert payload["data"]["result"] == "success"
    assert payload["source_url"] == "https://example.com/docs"

    assert openrouter_route.call_count == 2
