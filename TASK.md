It sounds like you're aiming for a robust and well-structured Python application, which is excellent. Your current directory structure is already quite good for a `uv`-managed Python package, following the recommended `src/` layout. Let's refine it for idiomatic practices, especially with your CLI and FastMCP integration.

---

## Current Structure Review

Your existing structure:

```
.
├── pyproject.toml
├── PLAN.md
├── .gitignore
├── README.md
├── .python-version
├── src/
│   └── doc_jangler/
│       ├── __init__.py
│       └── cli.py
```

This setup is generally solid. The `src/` directory for your package `doc_jangler` is a modern and recommended approach, preventing common namespace issues.

---

## Idiomatic Refinements

The primary area for refinement is how your `doc-jangler` script entry point (`doc_jangler:main`) connects to your Typer CLI application defined in `src/doc_jangler/cli.py`.

### 1\. **Connecting the CLI Entry Point**

Currently, your `pyproject.toml` points to `doc_jangler:main`, which refers to the `main` function in `src/doc_jangler/__init__.py`. This `main` function simply prints "Hello from doc-jangler\!". Your actual Typer CLI logic is in `src/doc_jangler/cli.py` under `typer_app()`.

To make this idiomatic:

- **Option A (Recommended): Modify `src/doc_jangler/__init__.py`**
  Make `src/doc_jangler/__init__.py` responsible for invoking your Typer app. This keeps `pyproject.toml` pointing to a single, clear entry point for your package.

  ```python
  # src/doc_jangler/__init__.py
  from doc_jangler.cli import typer_app # Import your typer app

  def main() -> None:
      typer_app() # This will run your CLI

  # You might also want to expose other core functions here if your package is meant to be imported
  # e.g., from .core import some_function
  ```

  This way, `doc-jangler` on the command line will execute `doc_jangler.cli.typer_app()`.

- **Option B: Directly Reference `cli.py` in `pyproject.toml`**
  You could change `pyproject.toml` to directly point to the `typer_app` in `cli.py`.

  ```toml
  # pyproject.toml
  [project.scripts]
  doc-jangler = "doc_jangler.cli:typer_app" # Points directly to your Typer app
  ```

  While this works, Option A is slightly more common as `__init__.py` often serves as the public interface for a package's top-level entry points.

### 2\. **FastMCP Integration**

When integrating FastMCP to serve your CLI commands as MCP tools, you'll want a clear separation of concerns.

- **`src/doc_jangler/mcp_tools.py` (or similar)**
  Consider creating a dedicated module for your FastMCP definitions. This module would house the classes or functions that FastMCP uses to expose your commands as tools. For example:

  ```python
  # src/doc_jangler/mcp_tools.py
  import typer
  from fastmcp.tool import Tool

  # You might import functions directly from cli.py or have wrapper functions here
  from .cli import main as cli_main_command, find_relevant_page_via_map, find_objective_in_top_pages

  class DocJanglerTools:
      @Tool(name="scrape_and_find_objective", description="Scrapes a URL and finds specific information based on an objective.")
      def scrape_and_find_objective_tool(self, url: str, objective: str) -> dict:
          """
          Scrapes content from the given URL and attempts to extract information
          relevant to the specified objective.
          """
          # You would adapt the logic from your cli.py's main function here
          # or call sub-functions from cli.py.
          relevant_pages = find_relevant_page_via_map(objective, url, app_instance_here) # Need to pass app
          if not relevant_pages:
              return {"status": "error", "message": "No relevant pages found."}

          result = find_objective_in_top_pages(relevant_pages, objective, app_instance_here) # Need to pass app
          if result:
              return {"status": "success", "data": result}
          else:
              return {"status": "error", "message": "Objective could not be fulfilled."}

      # Add other specific tools if your CLI has distinct sub-commands
      # @Tool(...)
      # def some_other_command_tool(...):
      #    ...

  ```

  This approach keeps your CLI implementation (Typer) separate from your tool definitions (FastMCP), promoting modularity and reusability, which aligns with your saved information about **Strict separation of concerns** and **Single Responsibility Principle**. You would then initialize and register `DocJanglerTools` with FastMCP in your main FastMCP application.

### 3\. **Environmental Variables and Global Objects**

In `src/doc_jangler/cli.py`, you're initializing `FirecrawlApp` and loading `dotenv` directly within the module. For a more testable and flexible application, especially when using dependency injection, consider:

- **Centralized Configuration:** Have a dedicated `config.py` or configuration loading mechanism.

- **Lazy Initialization/Dependency Injection:** Pass `app` (FirecrawlApp instance) and API keys as arguments to functions where they are needed, rather than relying on global module-level variables. Typer's dependency injection features can be very helpful here.

  ```python
  # src/doc_jangler/config.py
  import os
  from dotenv import load_dotenv

  load_dotenv()

  FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
  OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

  # src/doc_jangler/cli.py (simplified example)
  import typer
  from firecrawl import FirecrawlApp
  from .config import FIRECRAWL_API_KEY, OPENROUTER_API_KEY

  app = FirecrawlApp(api_key=FIRECRAWL_API_KEY, api_url="https://firecrawl.delo.sh")

  def get_firecrawl_app() -> FirecrawlApp:
      return app # Or create a new instance if stateful

  @typer_app.command()
  def scrape(url: str, objective: str, firecrawl_app: FirecrawlApp = typer.Depends(get_firecrawl_app)):
      # Use firecrawl_app here
      ...
  ```

  This makes your components more **plug-and-play** and easier to test, adhering to **Favor interface-first Design for plug-and-play component replacement** and **Use dependency injection liberally**.

---

## Revised Directory Structure Suggestion

```
.
├── pyproject.toml
├── PLAN.md
├── .gitignore
├── README.md
├── .python-version
├── src/
│   └── doc_jangler/
│       ├── __init__.py         # Imports typer_app, possibly config, and other core components
│       ├── cli.py              # Defines Typer commands
│       ├── config.py           # Handles environment variables and configuration
│       ├── core.py             # Contains business logic, e.g., scraping and analysis functions
│       └── mcp_tools.py        # Defines FastMCP tools
```

This structure further separates concerns:

- `__init__.py`: Package initialization and primary entry point.
- `cli.py`: Specifically for your command-line interface.
- `config.py`: Centralized configuration management.
- `core.py`: Where your main scraping, chunking, and Qdrant interaction logic would live, independent of how it's invoked (CLI or MCP).
- `mcp_tools.py`: Dedicated to exposing your application's capabilities as FastMCP tools.

By implementing these suggestions, your `doc-jangler` app will be more idiomatic, maintainable, and flexible for both CLI and LLM-driven interactions.
