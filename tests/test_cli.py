# This file will contain tests for the Click CLI commands.
import pytest
from click.testing import CliRunner
from doc_crawler.cli import cli

# To mock asyncio.run and the underlying functions effectively,
# we might need to use pytest-mock or unittest.mock
from unittest import mock

@pytest.fixture
def runner():
    return CliRunner()

# Test for the main 'crawl' command
@mock.patch("doc_crawler.cli.crawl_and_embed") # Target where it's imported and used
def test_crawl_command_success(mock_crawl_and_embed, runner):
    """Test the crawl command with basic arguments."""
    # Mock asyncio.run to directly return the result of the coroutine
    # or to check if the coroutine is called with right args.
    # For simplicity, we'll assume crawl_and_embed is a synchronous mock here,
    # or that its async nature is handled by how we mock it.

    # If crawl_and_embed was async, we'd do:
    # mock_crawl_and_embed.return_value = asyncio.Future()
    # mock_crawl_and_embed.return_value.set_result(None) # Or some other return value

    result = runner.invoke(cli, [
        "crawl",
        "http://example.com",
        "--qdrant-url", "http://localhost:1234",
        "--qdrant-collection", "testcol",
        "--embedding-model", "testmodel",
        "--ollama-base-url", "http://localhost:5678",
        "--max-pages", "5",
        "--output-type", "markdown"
    ])

    assert result.exit_code == 0
    assert "Starting crawl for URL: http://example.com" in result.output
    # Check if the mocked function was called with the correct parameters
    mock_crawl_and_embed.assert_called_once_with(
        initial_url="http://example.com", # Changed from url to initial_url
        qdrant_url="http://localhost:1234",
        qdrant_collection="testcol",
        embedding_model_name="testmodel",
        ollama_base_url="http://localhost:5678",
        max_pages_to_crawl=5,
        output_type="markdown",
        # Default values for new params if not specified in invoke,
        # or specified values if they were included in the test invoke call.
        # The `crawl` command in cli.py has defaults for these:
        chunking_strategy="paragraph", # default in cli.crawl
        chunk_size=512,             # default in cli.crawl
        chunk_overlap=50,           # default in cli.crawl
        is_doc_filter_active=True   # default in cli.crawl
    )
    assert "Successfully processed initial URL: http://example.com" in result.output # Updated message

@mock.patch("doc_crawler.cli.crawl_and_embed")
def test_crawl_command_custom_chunk_filter_args(mock_crawl_and_embed, runner):
    """Test the crawl command with custom chunking and filter arguments."""
    result = runner.invoke(cli, [
        "crawl",
        "http://custom.example.com",
        "--chunking-strategy", "sentence",
        "--chunk-size", "256",
        "--chunk-overlap", "25",
        "--is-doc-filter" # This will set it to True, which is default. To test False: --no-is-doc-filter
    ])
    assert result.exit_code == 0
    mock_crawl_and_embed.assert_called_once_with(
        initial_url="http://custom.example.com",
        qdrant_url=mock.ANY, # Default values will be used, or pass them if specific test needed
        qdrant_collection=mock.ANY,
        embedding_model_name=mock.ANY,
        ollama_base_url=mock.ANY,
        max_pages_to_crawl=mock.ANY, # Default is 1
        output_type=mock.ANY, # Default is markdown
        chunking_strategy="sentence",
        chunk_size=256,
        chunk_overlap=25,
        is_doc_filter_active=True # Because --is-doc-filter is a flag that defaults to True in definition
    )

    # Test with --no-is-doc-filter
    mock_crawl_and_embed.reset_mock()
    result_no_filter = runner.invoke(cli, [
        "crawl",
        "http://custom.example.com",
        "--no-is-doc-filter"
    ])
    assert result_no_filter.exit_code == 0
    mock_crawl_and_embed.assert_called_once_with(
        initial_url="http://custom.example.com",
        qdrant_url=mock.ANY,
        qdrant_collection=mock.ANY,
        embedding_model_name=mock.ANY,
        ollama_base_url=mock.ANY,
        max_pages_to_crawl=mock.ANY,
        output_type=mock.ANY,
        chunking_strategy="paragraph", # Default
        chunk_size=512,             # Default
        chunk_overlap=50,           # Default
        is_doc_filter_active=False # Due to --no-is-doc-filter
    )


@mock.patch("doc_crawler.cli.crawl_and_embed")
def test_crawl_command_default_args(mock_crawl_and_embed, runner):
    """Test the crawl command with default arguments."""
    from doc_crawler.cli import DEFAULT_QDRANT_URL, DEFAULT_QDRANT_COLLECTION, \
                                DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_BASE_URL

    result = runner.invoke(cli, ["crawl", "http://another.example.com"])

    assert result.exit_code == 0
    mock_crawl_and_embed.assert_called_once_with(
        initial_url="http://another.example.com", # Changed from url to initial_url
        qdrant_url=DEFAULT_QDRANT_URL,
        qdrant_collection=DEFAULT_QDRANT_COLLECTION, # Default collection name from cli.py
        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
        ollama_base_url=DEFAULT_OLLAMA_BASE_URL,
        max_pages_to_crawl=1,
        output_type="markdown",
        chunking_strategy="paragraph", # Default from cli.py crawl command
        chunk_size=512,             # Default from cli.py crawl command
        chunk_overlap=50,           # Default from cli.py crawl command
        is_doc_filter_active=True   # Default from cli.py crawl command (is_flag=True, default=True)
    )

@mock.patch("doc_crawler.cli.crawl_and_embed")
def test_crawl_command_error(mock_crawl_and_embed, runner):
    """Test the crawl command when crawl_and_embed raises an exception."""
    mock_crawl_and_embed.side_effect = Exception("Test crawl error")

    result = runner.invoke(cli, ["crawl", "http://error.example.com"])

    assert result.exit_code == 0 # Click command itself doesn't fail, but prints error
    assert "An error occurred: Test crawl error" in result.output
    assert "Successfully processed URL" not in result.output


# Test for 'geturls' command
@mock.patch("doc_crawler.cli.util_get_urls") # Target where it's imported and used
async def test_geturls_command_success(mock_util_get_urls, runner):
    """Test the geturls command."""
    # util_get_urls is an async function.
    # We need to mock its behavior when called by asyncio.run().
    # The easiest way is to have the mock itself be a synchronous function
    # that returns what the async function would have returned.
    # Or, more accurately, mock asyncio.run to handle the async mock.

    # Simpler: if util_get_urls is mocked directly, and it's called via asyncio.run(util_get_urls(...)),
    # the mock should behave like a coroutine if needed, or just return value.
    # Let's assume util_get_urls mock will return a list directly for simplicity of this test setup.

    # To properly mock an async function called with asyncio.run,
    # you often mock 'asyncio.run' itself or the target function to return a future.
    # However, for CLI tests, it's often simpler to mock the direct business logic function.

    # Patching 'asyncio.run' to just execute the first arg if it's a mock
    # or to return a predefined result for the coroutine.

    mock_util_get_urls.return_value = ["http://example.com/page1", "http://example.com/page2"]

    # The above mock setup is for when util_get_urls is called directly (not via asyncio.run).
    # If cli.py does `asyncio.run(util_get_urls(...))`, then `mock_util_get_urls` should be an async mock.
    # For now, let's assume the mock is simple and the test runner handles it.
    # Click's test runner doesn't inherently manage asyncio loops.
    # A common pattern is:
    # with mock.patch('asyncio.run', new_callable=mock.MagicMock) as mock_asyncio_run:
    #   mock_asyncio_run.return_value = ["http://example.com/page1", "http://example.com/page2"]
    #   result = runner.invoke(cli, ["geturls", "http://example.com", "--max-pages", "2"])
    #   mock_util_get_urls.assert_called_once_with(start_url="http://example.com", max_pages=2)

    # For this structure, since util_get_urls is imported as `from .utils import get_urls as util_get_urls`
    # and called as `asyncio.run(util_get_urls(...))`, we need to ensure the mock handles this.
    # The patch on "doc_crawler.cli.util_get_urls" makes it a normal MagicMock.
    # We will make its return value awaitable if needed, or just a direct list.

    with mock.patch('asyncio.run', side_effect=lambda coro: coro) as mock_async_run:
        # Make the mock_util_get_urls return a value that would be the result of the await
        # Since asyncio.run is side_effected to just return the coro,
        # the coro (which is mock_util_get_urls) should just return the list.
        mock_util_get_urls.return_value = ["http://example.com/page1", "http://example.com/page2"]

        result = runner.invoke(cli, ["geturls", "http://example.com", "--max-pages", "2"])

        assert result.exit_code == 0
        assert "Fetching URLs from: http://example.com, max pages: 2" in result.output
        assert "http://example.com/page1" in result.output
        assert "http://example.com/page2" in result.output
        # The call to util_get_urls is wrapped by asyncio.run.
        # The mock_util_get_urls itself is called.
        # mock_async_run.assert_called_once() # Verifies asyncio.run was called
        # And inside that, util_get_urls was called:
        mock_util_get_urls.assert_called_once_with(start_url="http://example.com", max_pages=2)


# Test for 'finddocs' command
@mock.patch("doc_crawler.cli.util_find_docs") # Target where it's imported and used
async def test_finddocs_command_success(mock_util_find_docs, runner):
    """Test the finddocs command."""
    with mock.patch('asyncio.run', side_effect=lambda coro: coro) as mock_async_run:
        mock_util_find_docs.return_value = ["http://docs.example.com/project1"]

        result = runner.invoke(cli, ["finddocs", "project1_id"])

        assert result.exit_code == 0
        assert "Analyzing project/identifier: project1_id" in result.output
        assert "http://docs.example.com/project1" in result.output
        mock_util_find_docs.assert_called_once_with(project_path_or_prd_url="project1_id")

# Example of running:
# pytest tests/test_cli.py
