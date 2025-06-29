## Agent Instructions for Doc Crawler Project

This document provides guidance for AI agents working on the `doc-crawler` project.

### Project Overview

The `doc-crawler` is a Python tool designed to:
1.  Crawl websites, specifically targeting documentation pages.
2.  Extract content from these pages (primarily in Markdown format).
3.  Chunk the extracted content into manageable pieces.
4.  Generate embeddings for these chunks using an Ollama-hosted model.
5.  Store the chunks and their embeddings in a Qdrant vector database.

The tool provides both a CLI interface (using Click) and a FastAPI web interface.

### Key Technologies
*   **Python 3.9+**
*   **crawl4ai:** For web crawling and content extraction.
*   **Click:** For the CLI interface.
*   **FastAPI:** For the web API.
*   **Ollama:** For serving local embedding models.
*   **Qdrant:** For the vector database.
*   **pytest:** For testing.

### Development Guidelines

1.  **Modularity:** Keep functionalities well-separated, especially in `doc_crawler/utils.py`.
2.  **Asynchronous Operations:** Use `async` and `await` for I/O-bound operations, particularly in crawling (`crawl4ai`), Ollama calls, and potentially Qdrant interactions if client libraries support it.
3.  **Configuration:**
    *   Default configurations (like Ollama URL, Qdrant URL, default model names, default chunking parameters, default filter status) should be clearly defined. These are currently at the top of `cli.py` and in Pydantic models in `api.py`.
    *   Allow overriding defaults through CLI options and API request parameters. The `crawl_and_embed` function in `main.py` now accepts these parameters.
4.  **Error Handling:** Implement robust error handling. Provide informative messages to the user (CLI) or appropriate HTTP error responses (API).
5.  **Logging:** Basic logging is implemented in `utils.py` and `main.py`. Ensure consistent use.
6.  **Testing:**
    *   Write unit tests for utility functions (`test_utils.py`). Mock external dependencies like Ollama, Qdrant, and `crawl4ai`'s web interactions.
    *   Write integration tests for the CLI (`test_cli.py`) and API (`test_api.py`).
    *   Aim for good test coverage.
7.  **Dependencies:** Keep `requirements.txt` up-to-date. Specify version ranges for stability where appropriate.
8.  **`is_doc` function:** The `is_doc` function in `utils.py` is critical for filtering relevant content. This function will likely need iterative improvements. Start with heuristics, but consider more advanced methods (e.g., a small classification model, or leveraging LLM capabilities of `crawl4ai` if it has any for content classification) if heuristics are insufficient.
9.  **Chunking Strategy:** The `chunk_content` function should also be flexible. While paragraph and basic sentence splitting are good starts, explore more advanced text splitting techniques (e.g., semantic chunking, token-based splitting with overlap) if needed for better embedding quality.
10. **Embedding Model & Vector Size:** The choice of embedding model (via Ollama) determines the vector size for Qdrant. This needs to be handled consistently. The Qdrant collection should be created with the correct vector size. If a collection exists with a different size, the tool should error out gracefully. The vector size should ideally be fetched from the Ollama model's metadata if possible, or be a known characteristic of the chosen model.
11. **`find_docs` Functionality:** This is a more exploratory feature. Initial implementation can be basic. Future enhancements could involve:
    *   Parsing dependency files (`requirements.txt`, `pyproject.toml`, `package.json`, etc.).
    *   Using search engine APIs (if permissible and API keys are manageable) to find official documentation.
    *   Heuristics for common project documentation URL patterns.
12. **Code Style:** Follow PEP 8 guidelines. Use a formatter like Black and a linter like Ruff/Flake8 if possible (though not strictly required for agent actions unless specified).

### Running the Application

*   **CLI:**
    ```bash
    # Example (ensure Ollama and Qdrant are running)
    # python -m doc_crawler.cli crawl http://example.com/docs
    ```
*   **API:**
    ```bash
    # Ensure you are in the directory *above* doc_crawler
    # uvicorn doc_crawler.api:app --reload
    ```
    Access OpenAPI docs at `http://localhost:8000/docs`.

### Future Considerations (Agent may be asked to implement these)
*   Support for other embedding providers (e.g., OpenAI, Hugging Face Hub models directly).
*   More sophisticated scheduling and state management for large crawling jobs.
*   A web UI for initiating crawls and viewing status.
*   Authentication for the FastAPI endpoints.

When making changes, update this `AGENTS.md` if the instructions become outdated or new conventions are established.Tool output for `create_file_with_block`:
