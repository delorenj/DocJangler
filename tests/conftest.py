from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable

import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if "firecrawl" not in sys.modules:
    firecrawl_module = types.ModuleType("firecrawl")

    class _FakeFirecrawlApp:  # pragma: no cover - testing scaffold
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    firecrawl_module.FirecrawlApp = _FakeFirecrawlApp
    sys.modules["firecrawl"] = firecrawl_module

if "fastmcp" not in sys.modules:
    fastmcp_module = types.ModuleType("fastmcp")
    tool_module = types.ModuleType("fastmcp.tool")

    def _tool_decorator(*args, **kwargs):  # pragma: no cover - testing scaffold
        def wrapper(func):
            return func

        return wrapper

    tool_module.Tool = _tool_decorator
    fastmcp_module.tool = tool_module
    sys.modules["fastmcp"] = fastmcp_module
    sys.modules["fastmcp.tool"] = tool_module

if "respx" not in sys.modules:
    import httpx

    class _Route:
        def __init__(self, method: str, url: str) -> None:
            self.method = method
            self.url = url
            self._responses: list = []
            self.call_count = 0

        def mock(self, side_effect=None, return_value=None):
            if side_effect is not None:
                if isinstance(side_effect, list):
                    self._responses = list(side_effect)
                else:
                    self._responses = [side_effect]
            elif return_value is not None:
                self._responses = [return_value]
            else:
                self._responses = []
            return self

        def next_response(self):
            self.call_count += 1
            if not self._responses:
                raise AssertionError(f"No mocked response for {self.method} {self.url}")
            response = self._responses.pop(0)
            if callable(response):
                response = response()
            return response

    class _Router:
        def __init__(self) -> None:
            self._routes: dict[tuple[str, str], _Route] = {}
            self._original_post = None

        def post(self, url: str) -> _Route:
            route = _Route("POST", url)
            self._routes[("POST", url)] = route
            return route

        def __enter__(self):
            self._original_post = httpx.post

            def _patched_post(url, *args, **kwargs):
                route = self._routes.get(("POST", url))
                if route is None:
                    raise AssertionError(f"Unexpected POST {url}")
                return route.next_response()

            httpx.post = _patched_post
            return self

        def __exit__(self, exc_type, exc, tb):
            if self._original_post is not None:
                httpx.post = self._original_post
            self._routes.clear()
            self._original_post = None

        def mock(self, func=None):
            def decorator(test_func):
                @wraps(test_func)
                def wrapper(*args, **kwargs):
                    with self:
                        return test_func(*args, **kwargs)

                return wrapper

            return decorator(func) if func else decorator

    router = _Router()
    respx_module = types.ModuleType("respx")
    respx_module.post = router.post
    respx_module.mock = router.mock
    sys.modules["respx"] = respx_module

import pytest

from doc_jangler.config import Settings


@dataclass
class FakeFirecrawlLink:
    url: str


class FakeFirecrawlApp:
    """Minimal Firecrawl replacement for tests."""

    def __init__(self, links: Iterable[str], markdown_map: Dict[str, str] | None = None) -> None:
        self._links = list(links)
        self._markdown_map = markdown_map or {}

    def map(self, url: str, search: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(links=[FakeFirecrawlLink(link) for link in self._links])

    def scrape(self, url: str) -> SimpleNamespace:
        return SimpleNamespace(markdown=self._markdown_map.get(url, ""))


@pytest.fixture
def fake_settings() -> Settings:
    """Provide deterministic settings for tests."""

    return Settings(
        firecrawl_api_key="fake-firecrawl",
        openrouter_api_key="fake-openrouter",
        firecrawl_api_url="https://firecrawl.example.com",
        openrouter_endpoint="https://openrouter.example.com",
        openrouter_model="fake-model",
    )


@pytest.fixture
def firecrawl_factory() -> "FakeFirecrawlFactory":
    """Factory fixture for building fake Firecrawl apps with custom data."""

    def factory(links: Iterable[str], markdown_map: Dict[str, str] | None = None) -> FakeFirecrawlApp:
        return FakeFirecrawlApp(links, markdown_map)

    return factory

FakeFirecrawlFactory = Callable[[Iterable[str], Dict[str, str] | None], FakeFirecrawlApp]
