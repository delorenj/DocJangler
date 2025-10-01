from __future__ import annotations

import json
import uuid
from typing import Callable, List, Optional, Sequence

import httpx
import redis
import typer

from .config import Settings, get_firecrawl_app, get_settings
from .core import (
    MappingResult,
    MissingConfigurationError,
    ObjectiveFinder,
    ObjectiveFinderError,
    ObjectiveResult,
)
from .document_pipeline import DocumentationIngestor
from .embeddings import OllamaEmbedder, OllamaEmbeddingError


class Colors:
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


typer_app = typer.Typer(help="DocJangler command line interface.")
ingest_app = typer.Typer(help="Commands that orchestrate the ingestion pipeline.")
typer_app.add_typer(ingest_app, name="ingest")


def _echo(message: str, color: str) -> None:
    typer.echo(f"{color}{message}{Colors.RESET}")


def _build_finder(settings: Optional[Settings] = None) -> ObjectiveFinder:
    settings = settings or get_settings()
    try:
        firecrawl_app = get_firecrawl_app(settings)
    except RuntimeError as exc:  # Raised when FIRECRAWL_API_KEY is missing.
        _echo(str(exc), Colors.RED)
        raise typer.Exit(code=1) from exc
    return ObjectiveFinder(firecrawl_app=firecrawl_app, settings=settings)


def _build_embedder(settings: Settings) -> OllamaEmbedder:
    return OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )


def _connect_redis(url: str):
    try:
        client = redis.from_url(url)
        client.ping()
    except redis.RedisError as exc:
        _echo(f"Redis connection error: {exc}", Colors.RED)
        raise typer.Exit(code=1) from exc
    return client


def _ensure_model_ready(finder: ObjectiveFinder) -> None:
    try:
        finder.ensure_model_available()
    except MissingConfigurationError as exc:
        _echo(f"Error: Could not connect to the language model. {exc}", Colors.RED)
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        _echo("Error: Could not connect to the language model. Please try again later.", Colors.RED)
        _echo(f"Details: {exc}", Colors.RED)
        raise typer.Exit(code=1) from exc
    except ObjectiveFinderError as exc:
        _echo(f"Unexpected model response: {exc}", Colors.RED)
        raise typer.Exit(code=1) from exc


def _progress_reporter(show_output: bool):
    def report(event: str, payload: str) -> None:
        if not show_output:
            return
        if event == "scrape":
            _echo(f"Scraping page: {payload}", Colors.YELLOW)
        elif event == "model_response":
            _echo(f"Model response: {payload}", Colors.CYAN)
        elif event == "parse_error":
            _echo(f"Error parsing JSON response: {payload}", Colors.RED)
        elif event == "objective_not_met":
            _echo("Objective not met in this page, continuing search...", Colors.YELLOW)
        elif event == "objective_met":
            _echo(f"Objective met at page: {payload}", Colors.GREEN)

    return report


def _safe_find_relevant_pages(
    finder: ObjectiveFinder,
    objective: str,
    site: str,
    *,
    topic_hint: Optional[str] = None,
    debug: Optional[Callable[[str], None]] = None,
) -> MappingResult:
    try:
        return finder.find_relevant_pages(objective, site, topic=topic_hint, debug=debug)
    except (ObjectiveFinderError, MissingConfigurationError) as exc:
        _echo(str(exc), Colors.RED)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        _echo(f"Error encountered: {exc}", Colors.RED)
        raise typer.Exit(code=1) from exc


def _safe_extract_metadata(
    finder: ObjectiveFinder,
    links: Sequence[str],
    objective: str,
    *,
    limit: Optional[int] = None,
    show_progress: bool = False,
) -> List[ObjectiveResult]:
    try:
        return finder.extract_metadata_from_pages(
            links,
            objective,
            limit=limit,
            progress=_progress_reporter(show_progress) if show_progress else None,
        )
    except (ObjectiveFinderError, MissingConfigurationError) as exc:
        _echo(str(exc), Colors.RED)
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        _echo(f"Error encountered during scraping: {exc}", Colors.RED)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        _echo(f"Error encountered: {exc}", Colors.RED)
        raise typer.Exit(code=1) from exc


@typer_app.command("jangle-docs")
def jangle_docs(
    topic: str = typer.Argument(..., help="Topic to explore."),
    site: str = typer.Option(..., "--site", help="Base site to inspect."),
    objective: Optional[str] = typer.Option(None, "--objective", help="Deprecated; retained for compatibility."),
    chunk_size: int = typer.Option(1200, "--chunk-size", min=200, help="Maximum characters per chunk."),
    chunk_overlap: int = typer.Option(200, "--chunk-overlap", min=0, help="Character overlap between chunks."),
    redis_url: Optional[str] = typer.Option(None, "--redis-url", help="Redis connection URL."),
    batch_id: Optional[str] = typer.Option(None, "--batch-id", help="Optional batch identifier."),
    debug: bool = typer.Option(False, "--debug/--no-debug", help="Print detailed crawl diagnostics."),
) -> None:
    """Collect, chunk, embed, and persist documentation for the site."""

    settings = get_settings()
    finder = _build_finder(settings)
    debug_logger = _debug_logger(debug)
    embedder = _build_embedder(settings)

    _echo("Initiating documentation ingestion...", Colors.YELLOW)
    _echo(f"Target topic: {topic}", Colors.CYAN)
    _echo(f"Target site: {site}", Colors.CYAN)

    redis_client = _connect_redis(redis_url or settings.redis_url)
    batch_identifier = batch_id or str(uuid.uuid4())

    ingestor = DocumentationIngestor(
        finder,
        redis_client,
        embedder=embedder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    try:
        summary = ingestor.ingest(
            site,
            topic,
            batch_id=batch_identifier,
            debug=debug_logger,
        )
    except OllamaEmbeddingError as exc:
        _echo(f"Embedding generation failed: {exc}", Colors.RED)
        raise typer.Exit(code=1) from exc

    if summary["chunk_count"] == 0:
        _echo("No content was ingested; verify the target site contains documentation.", Colors.RED)
        raise typer.Exit(code=1)

    _echo(
        (
            f"Stored {summary['chunk_count']} chunks from {summary['link_count']} pages "
            f"into Redis batch {batch_identifier}."
        ),
        Colors.GREEN,
    )
    typer.echo(json.dumps(summary, indent=2))


@ingest_app.command("find-domains")
def find_doc_domains(
    topic: str = typer.Argument(..., help="Topic to look up documentation domains for."),
    limit: int = typer.Option(5, "--limit", min=1, help="Maximum number of domains to return."),
) -> None:
    """Placeholder for domain discovery. To be implemented with Tavily search."""

    _echo(
        (
            "Domain discovery is not implemented yet. Provide a domain via --site or "
            "help implement Tavily-backed lookup."
        ),
        Colors.YELLOW,
    )
    raise typer.Exit(code=1)


@ingest_app.command("find-links")
def find_doc_links(
    topic: str = typer.Argument(..., help="Topic to map inside the domain."),
    domain: str = typer.Option(..., "--domain", help="Documentation domain to inspect."),
    max_links: int = typer.Option(10, "--max-links", min=1, help="Limit number of links returned."),
    debug: bool = typer.Option(False, "--debug/--no-debug", help="Print detailed crawl diagnostics."),
) -> None:
    """Return candidate documentation links within a domain for the given topic."""

    finder = _build_finder()
    _ensure_model_ready(finder)

    links = finder.collect_document_links(
        domain,
        topic=topic,
        debug=_debug_logger(debug),
    )

    if not links:
        _echo("No relevant links found.", Colors.RED)
        raise typer.Exit(code=1)

    payload = {"links": links[:max_links]}
    typer.echo(json.dumps(payload, indent=2))


@ingest_app.command("generate-metadata")
def generate_doc_link_metadata(
    topic: str = typer.Argument(..., help="Topic used when extracting metadata."),
    links: List[str] = typer.Argument(..., metavar="LINKS...", help="Links to inspect."),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Optional cap on processed links."),
    show_progress: bool = typer.Option(False, "--progress/--no-progress", help="Toggle verbose crawl output."),
) -> None:
    """Generate structured metadata for provided documentation links."""

    if not links:
        _echo("At least one link must be provided.", Colors.RED)
        raise typer.Exit(code=1)

    finder = _build_finder()
    _ensure_model_ready(finder)

    results = _safe_extract_metadata(
        finder,
        links,
        topic,
        limit=limit,
        show_progress=show_progress,
    )

    if not results:
        _echo("No metadata could be extracted for the supplied links.", Colors.RED)
        raise typer.Exit(code=1)

    payload = [
        {
            "source_url": result.source_url,
            "data": result.data,
        }
        for result in results
    ]
    typer.echo(json.dumps(payload, indent=2))


@ingest_app.command("find-example-code")
def find_and_scrape_example_code() -> None:
    """Placeholder for future example-code scraping implementation."""

    _echo(
        "Example code scraping is not implemented yet. Track progress in future iterations.",
        Colors.YELLOW,
    )
    raise typer.Exit(code=1)


@ingest_app.command("generate-embeddings")
def generate_embeddings() -> None:
    """Placeholder for embedding generation step."""

    _echo(
        "Embedding generation is not implemented yet. Hook into Qdrant once the pipeline is ready.",
        Colors.YELLOW,
    )
    raise typer.Exit(code=1)


@ingest_app.command("store-embeddings")
def store_embeddings() -> None:
    """Placeholder for storing embeddings in the vector database."""

    _echo(
        "Embedding storage is not implemented yet. Implement Qdrant integration to proceed.",
        Colors.YELLOW,
    )
    raise typer.Exit(code=1)


def main() -> None:
    """Execute the Typer CLI application."""

    typer_app()


__all__ = ["typer_app", "main"]
def _debug_logger(enabled: bool):
    if not enabled:
        return None

    def log(message: str) -> None:
        _echo(message, Colors.MAGENTA)

    return log
