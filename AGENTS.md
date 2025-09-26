# Repository Guidelines

## Project Structure & Module Organization
DocJangler ships as a Typer-based CLI. All runtime code lives under `src/doc_jangler/`: `cli.py` wires the command interface, `core.py` contains the crawling and extraction logic, and `config.py` loads Firecrawl/OpenRouter settings. Package exports are surfaced through `__init__.py` for the `doc-jangler` entry point. Reference materials and meeting artifacts sit in `docs/threads/`; keep generated assets out of `src/` so the CLI stays lean.

## Build, Test, and Development Commands
Install dependencies with `uv sync` (creates the managed `.venv`). Run the CLI locally via `uv run doc-jangler https://example.com "Summarize the API"`. For ad-hoc debugging, `uv run python -m doc_jangler.cli --help` lists the current parameters. When adding features, lint with `uv run ruff check` if you introduce Ruff to the tree, and document any new management commands in this section.

## Coding Style & Naming Conventions
Target Python 3.12, four-space indentation, and PEP 8 defaults. Keep the heavy use of type hints and frozen dataclasses consistent with existing modules. CLI feedback should stay ANSI-colored, but limit new color constants to `Colors`. Favor descriptive snake_case identifiers (`objective_result`, `mapping_links`), and keep Typer command names kebab-cased when exposed via the entry point.

## Testing Guidelines
An automated suite is not yet committed; new work should introduce `tests/` with pytest modules named `test_<feature>.py`. Cover both the happy path for `ObjectiveFinder` and error cases like missing config or malformed JSON. Use `uv run pytest` locally before requesting review, and include fixtures to stub external HTTP calls so tests do not hit Firecrawl/OpenRouter.

## Commit & Pull Request Guidelines
Follow the existing Conventional Commit style (`refactor: simplify CLI`) for history clarity. Each PR should describe the objective, highlight affected modules, and call out any required environment variables (`FIRECRAWL_API_KEY`, `OPENROUTER_API_KEY`). Link relevant issues, attach CLI output or transcripts when behaviour changes, and tick off testing notes (`pytest`, manual CLI run) in the PR template.

## Configuration & Security Notes
Secrets load from `.env` through `python-dotenv`; never commit `.env` files or raw keys. Prefer referencing `FIRECRAWL_API_URL` and `OPENROUTER_ENDPOINT` via the `Settings` dataclass instead of hardcoding URLs. When sharing logs, redact API responses and tokens captured by the CLI progress reporter.
