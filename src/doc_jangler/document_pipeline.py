from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence

from .core import ObjectiveFinder


class Embedder(Protocol):
    def embed(self, text: str) -> Sequence[float]:  # pragma: no cover - interface definition
        ...


def chunk_markdown(
    markdown: str,
    *,
    max_chars: int = 1200,
    overlap: int = 200,
) -> List[str]:
    """Split markdown content into overlapping chunks."""

    paragraphs = [para.strip() for para in markdown.split("\n\n") if para.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_length = 0

    for paragraph in paragraphs:
        para_length = len(paragraph)
        if current and current_length + para_length + 2 > max_chars:
            chunks.append("\n\n".join(current))
            if overlap > 0:
                overlap_text = "\n\n".join(current)[-overlap:]
                current = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)
            else:
                current = []
                current_length = 0

        current.append(paragraph)
        current_length += para_length + (2 if current_length else 0)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


class DeterministicEmbedder:
    """Produce deterministic pseudo-embeddings for testing/local workflows."""

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    def embed(self, text: str) -> List[float]:
        import hashlib

        if not text:
            return [0.0] * self._dimension

        digest = b""
        counter = 0
        while len(digest) < self._dimension * 8:
            counter_bytes = f"{counter}".encode("utf-8")
            digest += hashlib.sha256(text.encode("utf-8") + counter_bytes).digest()
            counter += 1

        vector: List[float] = []
        for index in range(self._dimension):
            chunk = digest[index * 8 : (index + 1) * 8]
            integer = int.from_bytes(chunk, "big", signed=False)
            vector.append(((integer % 2000) / 1000.0) - 1.0)

        return vector


@dataclass
class ChunkRecord:
    chunk_id: str
    batch_id: str
    source_url: str
    index: int
    text: str
    embedding: Sequence[float]

    def to_json(self) -> str:
        return json.dumps(
            {
                "chunk_id": self.chunk_id,
                "batch_id": self.batch_id,
                "source_url": self.source_url,
                "index": self.index,
                "text": self.text,
                "embedding": list(self.embedding),
            }
        )


class DocumentationIngestor:
    """Pipeline that collects, chunks, embeds, and stores documentation pages."""

    def __init__(
        self,
        finder: ObjectiveFinder,
        redis_client: "RedisProtocol",
        *,
        embedder: Optional[Embedder] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> None:
        self._finder = finder
        self._redis = redis_client
        self._embedder = embedder or DeterministicEmbedder()
        self._chunk_size = max(200, chunk_size)
        self._chunk_overlap = max(0, min(chunk_overlap, self._chunk_size // 2))

    def ingest(
        self,
        site: str,
        topic: str,
        *,
        batch_id: str,
        debug: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, object]:
        links = self._finder.collect_document_links(site, topic=topic, debug=debug)
        if debug:
            debug(f"Collected {len(links)} candidate documentation links")

        chunks: List[ChunkRecord] = []

        for link in links:
            markdown = self._finder.scrape_page_markdown(link, debug=debug)
            if not markdown:
                if debug:
                    debug(f"  -> Skipping {link}: no content returned")
                continue

            for index, chunk_text in enumerate(
                chunk_markdown(
                    markdown,
                    max_chars=self._chunk_size,
                    overlap=self._chunk_overlap,
                )
            ):
                chunk_id = str(uuid.uuid4())
                embedding = self._embedder.embed(chunk_text)
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        batch_id=batch_id,
                        source_url=link,
                        index=index,
                        text=chunk_text,
                        embedding=embedding,
                    )
                )

        stored = self._store(batch_id, links, chunks)

        return {
            "batch_id": batch_id,
            "link_count": len(links),
            "chunk_count": len(chunks),
            "stored": stored,
            "links": links,
        }

    def _store(self, batch_id: str, links: Sequence[str], chunks: Sequence[ChunkRecord]) -> bool:
        if not chunks:
            return False

        batch_key = f"docjangler:batch:{batch_id}"
        mapping = {chunk.chunk_id: chunk.to_json() for chunk in chunks}

        self._redis.hset(batch_key, mapping=mapping)
        self._redis.set(f"{batch_key}:links", json.dumps(list(links)))
        return True


class RedisProtocol(Protocol):  # pragma: no cover - protocol for typing only
    def hset(self, name: str, mapping: Dict[str, str]) -> int:
        ...

    def set(self, name: str, value: str) -> bool:
        ...

    def ping(self) -> bool:
        ...
