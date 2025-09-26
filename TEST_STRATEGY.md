Of course. Building on your excellent code structure and your own principle that "No task is complete without validation through testing," let's design a robust testing strategy for `doc-jangler`.

This strategy will focus on **isolation**, **automation**, and **confidence**, ensuring every part of your application is verifiable.

---

### \#\# 1. The Testing Toolkit 🧰

For a modern Python application like yours, these are the essential tools:

- **Test Runner**: **`pytest`** is the standard. Its fixture system, concise syntax, and powerful plugin ecosystem make it the best choice.
- **HTTP Mocking**: **`respx`** is perfect for this. Your application is heavily dependent on external APIs (Firecrawl, OpenRouter). `respx` allows you to intercept outgoing `httpx` requests and return predictable, fake responses. This makes your tests:
  - **Fast**: No real network latency.
  - **Reliable**: Not dependent on external API uptime.
  - **Free**: No API calls are actually made.
  - **Secure**: API keys are not needed during testing.
- **CLI Testing**: **`typer.testing.CliRunner`** is built into Typer and is designed specifically for invoking and asserting the behavior of your CLI commands.
- **Code Coverage**: **`pytest-cov`** integrates with `pytest` to measure what percentage of your codebase is executed by your tests. Aiming for high coverage ensures you aren't leaving parts of your logic untested.

You can install them all with `uv`:

```bash
uv pip install --dev pytest respx pytest-cov
```

---

### \#\# 2. Test Directory Structure

Your project structure should include a top-level `tests/` directory that mirrors your `src/` layout. This keeps tests separate from your application code but organized in a parallel way.

```
doc-jangler/
├── src/
│   └── doc_jangler/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── core.py
│       └── mcp_tools.py
├── tests/
│   ├── __init__.py
│   ├── test_cli.py
│   ├── test_core.py
│   └── test_mcp_tools.py
├── pyproject.toml
...
```

---

### \#\# 3. The Testing Pyramid Strategy

We'll apply the classic testing pyramid: a large base of fast unit tests, a smaller layer of integration tests, and a few key end-to-end tests.

#### \#\#\# Unit Tests (The Foundation)

**Goal**: Test a single function or class in complete isolation.

This is where you'll test your business logic in `core.py` and your tool definitions in `mcp_tools.py`.

**Example: Testing `_extract_structured_data` in `core.py`**

This is a perfect candidate for unit testing because it's a pure function: given an input string, it produces a predictable output.

````python
# tests/test_core.py
import pytest
from doc_jangler.core import ObjectiveFinder, Settings

# Dummy settings and app, not used in this specific test but needed for instantiation
dummy_settings = Settings(firecrawl_api_key="fake", openrouter_api_key="fake", ...)
finder = ObjectiveFinder(firecrawl_app=None, settings=dummy_settings)

@pytest.mark.parametrize(
    "model_response, expected_output",
    [
        # Test case 1: Perfect JSON
        ('{"found": true, "data": {"key": "value"}}', {"key": "value"}),
        # Test case 2: Objective not met
        ("Objective not met", None),
        # Test case 3: JSON wrapped in markdown
        ('```json\n{"found": true, "data": {"info": "details"}}\n```', {"info": "details"}),
        # Test case 4: Found is false
        ('{"found": false, "data": {}}', None),
    ],
)
def test_extract_structured_data(model_response, expected_output):
    """Verify that structured data is correctly extracted from various model responses."""
    if expected_output is None:
        assert finder._extract_structured_data(model_response) is None
    else:
        assert finder._extract_structured_data(model_response) == expected_output

def test_extract_structured_data_invalid_json():
    """Verify it raises a ValueError for malformed JSON."""
    with pytest.raises(ValueError, match="Invalid JSON returned from model"):
        finder._extract_structured_data('{"found": true, "data": malformed}')
````

#### \#\#\# Integration Tests (The Middle Layer)

**Goal**: Test how components work together. Here, we'll test the CLI's interaction with the `core` logic.

**Example: Testing the `jangle-docs` CLI command**

We will use `CliRunner` to run the command and `respx` to mock the API calls that `ObjectiveFinder` makes.

```python
# tests/test_cli.py
import json
import respx
from typer.testing import CliRunner
from httpx import Response

from doc_jangler.cli import typer_app

runner = CliRunner()

# Mock the OpenRouter and Firecrawl APIs
@respx.mock
def test_jangle_docs_success():
    """Verify the jangle-docs command works end-to-end with mocked APIs."""
    # 1. Mock the call to get a search parameter
    openrouter_route = respx.post("https://openrouter.ai/api/v1/chat/completions")
    openrouter_route.mock(return_value=Response(200, json={
        "choices": [{"message": {"content": "test parameter"}}]
    }))

    # 2. Mock the Firecrawl 'map' call
    firecrawl_map_route = respx.post("https://firecrawl.delo.sh/v0/map")
    firecrawl_map_route.mock(return_value=Response(200, json={
        "success": True, "links": [{"url": "https://example.com/docs"}]
    }))

    # 3. Mock the Firecrawl 'scrape' call
    firecrawl_scrape_route = respx.post("https://firecrawl.delo.sh/v0/scrape")
    firecrawl_scrape_route.mock(return_value=Response(200, json={
        "success": True, "data": {"markdown": "Some page content"}
    }))

    # 4. Mock the final analysis call to OpenRouter
    openrouter_route.mock(return_value=Response(200, json={
        "choices": [{"message": {"content": '{"found": true, "data": {"result": "success"}}'}}]
    }))

    # Invoke the CLI command
    result = runner.invoke(typer_app, ["jangle-docs", "my-topic", "--site", "https://example.com"])

    assert result.exit_code == 0
    assert "Objective successfully found!" in result.stdout

    # Verify the output is valid JSON and contains the expected data
    output_data = json.loads(result.stdout.splitlines()[-1])
    assert output_data["data"]["result"] == "success"
    assert output_data["source_url"] == "https://example.com/docs"
```

---

### \#\# 4. Running the Tests and CI/CD

To run your entire test suite and generate a coverage report, you'll execute this from your project's root directory:

```bash
pytest --cov=src/doc_jangler --cov-report=term-missing
```

The final step is to automate this. You should add a step to your CI/CD pipeline (e.g., GitHub Actions) that runs this command on every push or pull request. This enforces your rule of "Continuous validation throughout development lifecycle" and ensures that no code that breaks existing functionality can be merged.
