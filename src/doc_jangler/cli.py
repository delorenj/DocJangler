from __future__ import annotations

import json
from typing import Optional

import httpx
import typer
from firecrawl import FirecrawlApp

from .config import Settings, get_firecrawl_app, get_settings
from .core import (
    MappingResult,
    MissingConfigurationError,
    ObjectiveFinder,
    ObjectiveFinderError,
    ObjectiveResult,
)


class Colors:
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


typer_app = typer.Typer()


@typer_app.command()
def main(url: str, objective: str) -> None:
    settings: Settings = get_settings()
    try:
        firecrawl_app: FirecrawlApp = get_firecrawl_app(settings)
    except RuntimeError as exc:
        print(f"{Colors.RED}{exc}{Colors.RESET}")
        raise typer.Exit(code=1) from exc

    finder = ObjectiveFinder(firecrawl_app=firecrawl_app, settings=settings)

    try:
        finder.ensure_model_available()
    except MissingConfigurationError as exc:
        print(
            f"{Colors.RED}Error: Could not connect to the language model. {exc}{Colors.RESET}"
        )
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        print(
            f"{Colors.RED}Error: Could not connect to the language model. Please try again later.{Colors.RESET}"
        )
        print(f"{Colors.RED}Details: {str(exc)}{Colors.RESET}")
        raise typer.Exit(code=1) from exc
    except ObjectiveFinderError as exc:
        print(f"{Colors.RED}Unexpected model response: {exc}{Colors.RESET}")
        raise typer.Exit(code=1) from exc

    print(f"{Colors.YELLOW}Initiating web crawling process...{Colors.RESET}")
    print(f"{Colors.CYAN}Understood. Objective: {objective}{Colors.RESET}")
    print(f"{Colors.CYAN}Searching website: {url}{Colors.RESET}")

    try:
        mapping: MappingResult = finder.find_relevant_pages(objective, url)
    except (ObjectiveFinderError, MissingConfigurationError) as exc:
        print(f"{Colors.RED}{exc}{Colors.RESET}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"{Colors.RED}Error encountered: {str(exc)}{Colors.RESET}")
        raise typer.Exit(code=1) from exc

    if not mapping.links:
        print(f"{Colors.RED}No relevant pages found. Exiting...{Colors.RESET}")
        raise typer.Exit(code=0)

    if mapping.search_parameter:
        print(
            f"{Colors.GREEN}Optimal search parameter identified: {mapping.search_parameter}{Colors.RESET}"
        )
    print(f"{Colors.GREEN}Website mapping completed successfully.{Colors.RESET}")

    def report(event: str, payload: str) -> None:
        if event == "scrape":
            print(f"{Colors.YELLOW}Scraping page: {payload}{Colors.RESET}")
        elif event == "model_response":
            print(f"{Colors.CYAN}Model response: {payload}{Colors.RESET}")
        elif event == "parse_error":
            print(
                f"{Colors.RED}Error parsing JSON response: {payload}{Colors.RESET}"
            )
        elif event == "objective_not_met":
            print(
                f"{Colors.YELLOW}Objective not met in this page, continuing search...{Colors.RESET}"
            )
        elif event == "objective_met":
            print(f"{Colors.GREEN}Objective met at page: {payload}{Colors.RESET}")

    try:
        result: Optional[ObjectiveResult] = finder.find_objective_in_pages(
            mapping.links,
            objective,
            limit=3,
            progress=report,
        )
    except (ObjectiveFinderError, MissingConfigurationError) as exc:
        print(f"{Colors.RED}{exc}{Colors.RESET}")
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        print(f"{Colors.RED}Error encountered during scraping: {str(exc)}{Colors.RESET}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"{Colors.RED}Error encountered: {str(exc)}{Colors.RESET}")
        raise typer.Exit(code=1) from exc

    if result:
        print(
            f"{Colors.GREEN}Objective successfully found! Extracted information:{Colors.RESET}"
        )
        output = {"source_url": result.source_url, "data": result.data}
        print(json.dumps(output, indent=2))
    else:
        print(f"{Colors.RED}Objective could not be fulfilled.{Colors.RESET}")


__all__ = ["typer_app", "main"]
