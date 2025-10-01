"""Embedding utilities for DocJangler."""

from __future__ import annotations

from typing import List, Sequence

import httpx


class OllamaEmbeddingError(RuntimeError):
    """Raised when embedding generation via Ollama fails."""


class OllamaEmbedder:
    """Embed text by delegating to an Ollama server."""

    def __init__(self, base_url: str, model: str, *, timeout: float = 30.0) -> None:
        self._model = model
        self._timeout = timeout
        self._endpoint = base_url.rstrip("/") + "/api/embeddings"

    def embed(self, text: str) -> Sequence[float]:
        payload = {"model": self._model, "prompt": text}

        try:
            response = httpx.post(self._endpoint, json=payload, timeout=self._timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network operations
            raise OllamaEmbeddingError(f"Failed to generate embedding: {exc}") from exc

        data = response.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise OllamaEmbeddingError("Ollama response did not include an embedding vector.")

        try:
            return [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
            raise OllamaEmbeddingError("Embedding vector contained non-numeric values.") from exc


__all__ = ["OllamaEmbedder", "OllamaEmbeddingError"]
