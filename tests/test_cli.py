from __future__ import annotations

import json
from types import SimpleNamespace

from typer.testing import CliRunner

from doc_jangler import cli as cli_module
from doc_jangler.cli import typer_app
from doc_jangler.config import Settings


class FakeFirecrawlApp:
    def map(self, url: str, search: str | None = None) -> SimpleNamespace:  # pragma: no cover - unused in test
        return SimpleNamespace(links=[SimpleNamespace(url="https://example.com/docs/intro")])

    def scrape(self, url: str) -> SimpleNamespace:  # pragma: no cover - unused in test
        return SimpleNamespace(markdown="Example documentation content")


runner = CliRunner()


def test_jangle_docs_success(monkeypatch) -> None:
    fake_settings = Settings(
        firecrawl_api_key="fake-firecrawl",
        openrouter_api_key="fake-openrouter",
        firecrawl_api_url="https://firecrawl.example.com",
        openrouter_endpoint="https://openrouter.example.com",
        openrouter_model="fake-model",
        redis_url="redis://localhost:6379/0",
        ollama_base_url="http://localhost:11434",
        ollama_embedding_model="nomic-embed-text",
    )

    monkeypatch.setattr(cli_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(cli_module, "get_firecrawl_app", lambda settings: FakeFirecrawlApp())
    monkeypatch.setattr(
        cli_module,
        "_build_embedder",
        lambda settings: SimpleNamespace(embed=lambda text: [0.0]),
    )

    class DummyRedis:
        def __init__(self) -> None:
            self.hset_calls = []
            self.set_calls = []

        def ping(self) -> bool:
            return True

        def hset(self, name: str, mapping):
            self.hset_calls.append((name, mapping))
            return 1

        def set(self, name: str, value: str) -> bool:
            self.set_calls.append((name, value))
            return True

    dummy_redis = DummyRedis()

    class DummyIngestor:
        def __init__(self, finder, redis_client, *, embedder, chunk_size, chunk_overlap):
            self.finder = finder
            self.redis_client = redis_client
            self.embedder = embedder
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def ingest(self, site, topic, *, batch_id, debug=None):
            return {
                "batch_id": batch_id,
                "link_count": 2,
                "chunk_count": 4,
                "stored": True,
                "links": [
                    "https://example.com/docs/intro",
                    "https://example.com/docs/setup",
                ],
            }

    monkeypatch.setattr(cli_module.redis, "from_url", lambda url: dummy_redis)
    monkeypatch.setattr(cli_module, "DocumentationIngestor", DummyIngestor)

    result = runner.invoke(
        typer_app,
        [
            "jangle-docs",
            "my-topic",
            "--site",
            "https://example.com",
            "--chunk-size",
            "300",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Stored 4 chunks" in result.stdout

    json_start = result.stdout.index("{")
    payload = json.loads(result.stdout[json_start:])
    assert payload["chunk_count"] == 4
    assert payload["link_count"] == 2
    assert payload["stored"] is True
