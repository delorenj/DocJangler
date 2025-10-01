from __future__ import annotations

import json
from typing import Dict, List

import pytest

from doc_jangler.document_pipeline import (
    DeterministicEmbedder,
    DocumentationIngestor,
    chunk_markdown,
)


class FakeFinder:
    def __init__(self) -> None:
        self.links: List[str] = []
        self.contents: Dict[str, str] = {}
        self.scrape_calls: List[str] = []

    def collect_document_links(self, site: str, *, topic: str | None = None, debug=None) -> List[str]:
        return self.links

    def scrape_page_markdown(self, url: str, *, debug=None) -> str | None:
        self.scrape_calls.append(url)
        return self.contents.get(url)


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: Dict[str, Dict[str, str]] = {}
        self.values: Dict[str, str] = {}

    def hset(self, name: str, mapping: Dict[str, str]) -> int:
        bucket = self.hashes.setdefault(name, {})
        bucket.update(mapping)
        return len(mapping)

    def set(self, name: str, value: str) -> bool:
        self.values[name] = value
        return True

    def ping(self) -> bool:
        return True


def test_chunk_markdown_produces_overlapping_chunks() -> None:
    text = "Paragraph one." + "\n\n" + "Paragraph two." + "\n\n" + "Paragraph three."
    chunks = chunk_markdown(text, max_chars=20, overlap=5)
    assert len(chunks) >= 2
    assert "Paragraph one" in chunks[0]


@pytest.mark.parametrize("chunk_size", [200, 400])
def test_documentation_ingestor_stores_chunks(chunk_size: int) -> None:
    finder = FakeFinder()
    finder.links = [
        "https://example.com/docs/intro",
        "https://example.com/docs/setup",
    ]
    finder.contents = {
        "https://example.com/docs/intro": "Introduction to docs.\n\nMore content.",
        "https://example.com/docs/setup": "Setup guide details.\n\nExtras.",
    }

    redis_client = FakeRedis()
    embedder = DeterministicEmbedder(dimension=8)

    ingestor = DocumentationIngestor(
        finder,
        redis_client,
        embedder=embedder,
        chunk_size=chunk_size,
        chunk_overlap=0,
    )

    summary = ingestor.ingest(
        "https://example.com",
        topic="Example Docs",
        batch_id="batch-123",
    )

    assert summary["link_count"] == 2
    assert summary["chunk_count"] >= 2
    assert summary["stored"] is True
    assert redis_client.hashes  # ensure something stored

    for payload in redis_client.hashes.values():
        for raw_value in payload.values():
            data = json.loads(raw_value)
            assert data["batch_id"] == "batch-123"
            assert isinstance(data["embedding"], list)
            assert len(data["embedding"]) == 8
