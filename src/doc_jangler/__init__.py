"""DocJangler package entry point."""

from .cli import typer_app


def main() -> None:
    """Execute the Typer CLI application."""

    typer_app()


__all__ = ["main", "typer_app"]
