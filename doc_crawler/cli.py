# This file will contain the CLI interface for the doc_crawler project using Click.
import click
import asyncio
import json # For pretty printing dicts/lists if needed

from .main import crawl_and_embed
from .utils import get_urls as util_get_urls # Alias to avoid naming conflict
from .utils import find_docs as util_find_docs # Alias

# Default values
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_QDRANT_URL = "http://localhost:6333" # Qdrant default
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large" # A common Ollama model, user should have it
DEFAULT_QDRANT_COLLECTION = "ai_docs"

@click.group()
def cli():
    """
    A CLI tool to crawl documentation websites, process content,
    embed it, and store it in a Qdrant vector database.
    """
    pass

@cli.command()
@click.argument("url", type=str)
@click.option(
    "--qdrant-url",
    default=DEFAULT_QDRANT_URL,
    show_default=True,
    help="URL of the Qdrant instance."
)
@click.option(
    "--qdrant-collection",
    default=DEFAULT_QDRANT_COLLECTION,
    show_default=True,
    help="Name of the Qdrant collection to use/create."
)
@click.option(
    "--embedding-model",
    default=DEFAULT_EMBEDDING_MODEL,
    show_default=True,
    help="Name of the embedding model to use with Ollama (e.g., mxbai-embed-large, nomic-embed-text)."
)
@click.option(
    "--ollama-base-url",
    default=DEFAULT_OLLAMA_BASE_URL,
    show_default=True,
    help="Base URL of the Ollama API."
)
@click.option(
    "--max-pages",
    default=1,
    show_default=True,
    type=int,
    help="Maximum number of pages to crawl (for deep crawling, if applicable)."
)
@click.option(
    "--output-type",
    default="markdown",
    show_default=True,
    type=click.Choice(['markdown', 'text', 'html'], case_sensitive=False), # Reflecting crawl4ai common outputs
    help="Preferred output type from crawl4ai for processing."
)
@click.option(
    "--chunking-strategy",
    default="paragraph",
    show_default=True,
    type=click.Choice(['paragraph', 'sentence', 'fixed'], case_sensitive=False), # 'fixed' can map to default in utils
    help="Strategy for chunking content."
)
@click.option(
    "--chunk-size",
    default=512,
    show_default=True,
    type=int,
    help="Target size for content chunks (meaning varies by strategy)."
)
@click.option(
    "--chunk-overlap",
    default=50,
    show_default=True,
    type=int,
    help="Overlap size between chunks (primarily for fixed strategy)."
)
@click.option(
    "--is-doc-filter",
    is_flag=True, # Changed from doc-type-filter to is-doc-filter for clarity
    default=True, # Defaulting to True as it's a doc crawler
    show_default=True,
    help="Enable filter to attempt to only process documentation-like pages."
)
def crawl(
    url: str,
    qdrant_url: str,
    qdrant_collection: str,
    embedding_model: str,
    ollama_base_url: str,
    max_pages: int,
    output_type: str,
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    is_doc_filter: bool
):
    """
    Crawls a given URL, extracts documentation content, chunks it,
    generates embeddings, and stores them in a Qdrant collection.
    """
    click.echo(f"Starting crawl for URL: {url}")
    click.echo(f"Qdrant: {qdrant_url}/{qdrant_collection}, Model: {embedding_model} (via {ollama_base_url})")
    click.echo(f"Crawl Params: MaxPages={max_pages}, OutputType={output_type}, IsDocFilterActive={is_doc_filter}")
    click.echo(f"Chunking Params: Strategy={chunking_strategy}, Size={chunk_size}, Overlap={chunk_overlap}")

    try:
        asyncio.run(crawl_and_embed(
            initial_url=url, # Parameter name changed in crawl_and_embed
            qdrant_url=qdrant_url,
            qdrant_collection=qdrant_collection,
            embedding_model_name=embedding_model,
            ollama_base_url=ollama_base_url,
            max_pages_to_crawl=max_pages,
            output_type=output_type,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_doc_filter_active=is_doc_filter
        ))
        click.echo(f"Successfully processed initial URL: {url} and any discovered pages.")
    except Exception as e:
        click.echo(f"An error occurred: {e}", err=True)
        # Consider more specific error handling or re-raising if needed
        # For now, just printing the error to the CLI user

@cli.command("geturls")
@click.argument("start_url", type=str)
@click.option(
    "--max-pages",
    default=10,
    show_default=True,
    type=int,
    help="Maximum number of pages/links to discover."
)
def geturls_command(start_url: str, max_pages: int):
    """
    Fetches a list of relevant URLs starting from a given URL.
    (Currently uses placeholder implementation)
    """
    click.echo(f"Fetching URLs from: {start_url}, max pages: {max_pages}")
    try:
        urls = asyncio.run(util_get_urls(start_url=start_url, max_pages=max_pages))
        if urls:
            click.echo("Found URLs:")
            for url in urls:
                click.echo(f"- {url}")
        else:
            click.echo("No URLs found or an error occurred.")
    except Exception as e:
        click.echo(f"An error occurred: {e}", err=True)

@cli.command("finddocs")
@click.argument("project_identifier", type=str)
def finddocs_command(project_identifier: str):
    """
    Analyzes a project (local path or PRD URL) to find documentation URLs.
    (Currently uses placeholder implementation)
    """
    click.echo(f"Analyzing project/identifier: {project_identifier} to find documentation URLs.")
    try:
        doc_urls = asyncio.run(util_find_docs(project_path_or_prd_url=project_identifier))
        if doc_urls:
            click.echo("Potential documentation URLs found:")
            for url in doc_urls:
                click.echo(f"- {url}")
        else:
            click.echo("No documentation URLs found or an error occurred.")
    except Exception as e:
        click.echo(f"An error occurred: {e}", err=True)

if __name__ == "__main__":
    cli()
