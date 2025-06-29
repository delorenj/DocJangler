# Document Crawler & Embedder (`doc-crawler`)

`doc-crawler` is a Python-based tool designed to crawl documentation websites, extract relevant content, chunk it into manageable pieces, generate embeddings using local LLMs (via Ollama), and store these embeddings along with the text chunks in a Qdrant vector database. This enables semantic search and RAG (Retrieval Augmented Generation) capabilities on your documentation.

The tool provides both a Command Line Interface (CLI) and a FastAPI-based web API for integration into various workflows.

## Features

*   **Web Crawling:** Leverages the `crawl4ai` library to fetch and parse web content, focusing on documentation.
*   **Content Extraction:** Primarily aims to extract clean Markdown from web pages.
*   **Text Chunking:** Various strategies for splitting large documents into smaller chunks suitable for embedding.
*   **Local Embeddings:** Uses Ollama to generate text embeddings with locally hosted models (e.g., `mxbai-embed-large`, `nomic-embed-text`).
*   **Vector Storage:** Stores text chunks and their corresponding embeddings in a Qdrant vector database.
*   **CLI Interface:** Easy-to-use commands via `click` for initiating crawls and other operations.
*   **FastAPI Endpoints:** Exposes core functionalities through a web API for programmatic access.
*   **Documentation Discovery (Experimental):** Includes a feature to attempt to find documentation URLs for a given project.

## Project Structure

```
.
├── doc_crawler/
│   ├── __init__.py
│   ├── main.py         # Core orchestration logic for crawling and embedding
│   ├── utils.py        # Utility functions (is_doc, chunking, embedding, qdrant interaction)
│   ├── cli.py          # Click-based CLI application
│   └── api.py          # FastAPI application
├── tests/
│   ├── __init__.py
│   ├── test_utils.py   # Unit tests for utility functions
│   ├── test_cli.py     # Tests for CLI commands
│   └── test_api.py     # Tests for FastAPI endpoints
├── requirements.txt    # Python package dependencies
├── README.md           # This file
└── AGENTS.md           # Instructions for AI development agents
```

## Prerequisites

*   Python 3.9+
*   [Ollama](https://ollama.ai/) installed and running with desired embedding models pulled.
    *   Example: `ollama pull mxbai-embed-large`
*   [Qdrant](https://qdrant.tech/documentation/overview/) installed and running.
    *   E.g., using Docker: `docker pull qdrant/qdrant` and `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd doc-crawler
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    You might also need to run `crawl4ai-setup` if it's the first time using `crawl4ai` to ensure browser components are installed:
    ```bash
    crawl4ai-setup
    # or, if playwright issues occur:
    # python -m playwright install --with-deps chromium
    ```

## Usage

### CLI Interface

The main entry point for the CLI is `doc_crawler.cli`. You can run it as a module:
`python -m doc_crawler.cli [COMMAND] [ARGS]...`

**Available Commands:**

*   **`crawl`**: Crawls a URL, processes, embeds, and stores its content.
    ```bash
    python -m doc_crawler.cli crawl <URL> [OPTIONS]
    ```
    Example:
    ```bash
    python -m doc_crawler.cli crawl https://click.palletsprojects.com/en/8.1.x/ \
        --qdrant-collection click_docs \
        --embedding-model nomic-embed-text \
        --max-pages 10
    ```
    Common Options:
    *   `--qdrant-url TEXT`: URL of the Qdrant instance (default: `http://localhost:6333`).
    *   `--qdrant-collection TEXT`: Qdrant collection name (default: `ai_docs`).
    *   `--embedding-model TEXT`: Ollama embedding model name (default: `mxbai-embed-large`).
    *   `--ollama-base-url TEXT`: Ollama API base URL (default: `http://localhost:11434`).
    *   `--max-pages INTEGER`: Maximum number of pages to crawl (default: `1`).
    *   `--output-type [markdown|text|html]`: Preferred output from crawl4ai (default: `markdown`).
    *   `--chunking-strategy [paragraph|sentence|fixed]`: Strategy for chunking content (default: `paragraph`).
    *   `--chunk-size INTEGER`: Target size for content chunks (default: `512`).
    *   `--chunk-overlap INTEGER`: Overlap size between chunks (default: `50`).
    *   `--is-doc-filter / --no-is-doc-filter`: Enable/disable filter for documentation-like pages (default: enabled).

*   **`geturls`**: Fetches a list of relevant URLs from a starting URL (placeholder implementation).
    ```bash
    python -m doc_crawler.cli geturls <START_URL> [--max-pages INTEGER]
    ```

*   **`finddocs`**: Analyzes a project identifier to find documentation URLs (placeholder implementation).
    ```bash
    python -m doc_crawler.cli finddocs <PROJECT_IDENTIFIER>
    ```

Run `python -m doc_crawler.cli [COMMAND] --help` for more details on specific command options.

### FastAPI Web API

To run the FastAPI application:
```bash
# Ensure you are in the directory *above* doc_crawler, i.e., the project root
uvicorn doc_crawler.api:app --reload --host 0.0.0.0 --port 8000
```
The API documentation (Swagger UI) will be available at `http://localhost:8000/docs`.

**API Endpoints:**

*   `POST /crawl/`: Initiates crawling and embedding for a given URL.
    *   Request Body: JSON with parameters including:
        *   `url: HttpUrl` (required)
        *   `qdrant_url: str` (optional, defaults provided)
        *   `qdrant_collection: str` (optional, defaults provided)
        *   `embedding_model_name: str` (optional, defaults provided)
        *   `ollama_base_url: str` (optional, defaults provided)
        *   `max_pages_to_crawl: int` (optional, default: 1)
        *   `output_type: str` (optional, default: "markdown", enum: ["markdown", "text", "html"])
        *   `chunking_strategy: str` (optional, default: "paragraph", enum: ["paragraph", "sentence", "fixed"])
        *   `chunk_size: int` (optional, default: 512)
        *   `chunk_overlap: int` (optional, default: 50)
        *   `is_doc_filter_active: bool` (optional, default: true)
*   `POST /get-urls/`: Fetches relevant URLs from a start URL.
*   `POST /find-docs/`: Finds documentation URLs for a project identifier.

Refer to the `/docs` endpoint for detailed request/response schemas.

## Development

### Running Tests

Ensure `pytest` and `pytest-mock` are installed (they are in `requirements.txt`).
```bash
pytest
# or
python -m pytest
```

### Code Style & Linting

(Instructions for linters/formatters like Black or Ruff would go here if they were set up).

### `AGENTS.md`

For AI agents contributing to this project, please refer to the `AGENTS.md` file for specific development guidelines and instructions.

## Contributing

(Details on how to contribute to the project, if open for contributions).

## License

(Specify license information, e.g., MIT, Apache 2.0).
