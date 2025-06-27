# This file will contain tests for the FastAPI application.
import pytest
from fastapi.testclient import TestClient
# Assuming your FastAPI app instance is named 'app' in 'doc_crawler.api'
from doc_crawler.api import app, DEFAULT_QDRANT_COLLECTION, DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_BASE_URL, DEFAULT_QDRANT_URL

from unittest import mock
import asyncio

client = TestClient(app)

# --- Test /crawl/ endpoint ---
@mock.patch("doc_crawler.api.crawl_and_embed") # Patch where it's used
def test_crawl_endpoint_success(mock_crawl_and_embed_api):
    # Mock crawl_and_embed to be an async function that does nothing or returns a mock
    async def mock_async_crawl_and_embed(*args, **kwargs):
        return None # Simulate successful completion
    mock_crawl_and_embed_api.side_effect = mock_async_crawl_and_embed

    test_url = "http://example.com/api_test"
    response = client.post(
        "/crawl/",
        json={
            "url": test_url,
            "qdrant_url": "http://test-qdrant:6333",
            "qdrant_collection": "test_api_collection",
            "embedding_model_name": "test-embed-model",
            "ollama_base_url": "http://test-ollama:11434",
            "max_pages_to_crawl": 2,
            "output_type": "markdown"
        },
    )
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["message"] == "Crawling and embedding process initiated successfully."
    assert json_response["processed_url"] == test_url
    assert json_response["qdrant_collection"] == "test_api_collection"

    mock_crawl_and_embed_api.assert_called_once_with(
        initial_url=test_url, # Changed from url to initial_url
        qdrant_url="http://test-qdrant:6333",
        qdrant_collection="test_api_collection",
        embedding_model_name="test-embed-model",
        ollama_base_url="http://test-ollama:11434",
        max_pages_to_crawl=2,
        output_type="markdown",
        # Values from the CrawlRequest model in api.py for new params
        chunking_strategy="sentence", # Assuming this was passed in JSON, see below
        chunk_size=256,          # Assuming this was passed in JSON
        chunk_overlap=30,        # Assuming this was passed in JSON
        is_doc_filter_active=False # Assuming this was passed in JSON
    )

# Updated test_crawl_endpoint_success to include new params in request
@mock.patch("doc_crawler.api.crawl_and_embed") # Patch where it's used
def test_crawl_endpoint_success_with_all_params(mock_crawl_and_embed_api):
    async def mock_async_crawl_and_embed(*args, **kwargs):
        return None
    mock_crawl_and_embed_api.side_effect = mock_async_crawl_and_embed

    test_url = "http://example.com/api_test_full"
    request_payload = {
        "url": test_url,
        "qdrant_url": "http://test-qdrant:6333",
        "qdrant_collection": "test_api_collection_full",
        "embedding_model_name": "test-embed-model-full",
        "ollama_base_url": "http://test-ollama:11434",
        "max_pages_to_crawl": 3,
        "output_type": "text",
        "chunking_strategy": "sentence",
        "chunk_size": 256,
        "chunk_overlap": 30,
        "is_doc_filter_active": False
    }
    response = client.post("/crawl/", json=request_payload)

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["message"] == "Crawling and embedding process initiated successfully."
    assert json_response["processed_url"] == test_url
    assert json_response["qdrant_collection"] == "test_api_collection_full"

    mock_crawl_and_embed_api.assert_called_once_with(
        initial_url=test_url,
        qdrant_url="http://test-qdrant:6333",
        qdrant_collection="test_api_collection_full",
        embedding_model_name="test-embed-model-full",
        ollama_base_url="http://test-ollama:11434",
        max_pages_to_crawl=3,
        output_type="text",
        chunking_strategy="sentence",
        chunk_size=256,
        chunk_overlap=30,
        is_doc_filter_active=False
    )


@mock.patch("doc_crawler.api.crawl_and_embed")
def test_crawl_endpoint_default_params(mock_crawl_and_embed_api):
    async def mock_async_crawl_and_embed(*args, **kwargs):
        return None
    mock_crawl_and_embed_api.side_effect = mock_async_crawl_and_embed

    test_url = "http://default.example.com"
    response = client.post("/crawl/", json={"url": test_url})

    assert response.status_code == 200
    # Check call to the mocked crawl_and_embed
    # Defaults for new fields are defined in the Pydantic model `CrawlRequest`
    mock_crawl_and_embed_api.assert_called_once_with(
        initial_url=test_url, # Changed from url to initial_url
        qdrant_url=DEFAULT_QDRANT_URL,
        qdrant_collection=DEFAULT_QDRANT_COLLECTION,
        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
        ollama_base_url=DEFAULT_OLLAMA_BASE_URL,
        max_pages_to_crawl=1, # Default from CrawlRequest Pydantic model
        output_type="markdown", # Default from CrawlRequest Pydantic model
        chunking_strategy="paragraph", # Default from CrawlRequest model
        chunk_size=512,             # Default from CrawlRequest model
        chunk_overlap=50,           # Default from CrawlRequest model
        is_doc_filter_active=True   # Default from CrawlRequest model
    )

@mock.patch("doc_crawler.api.crawl_and_embed")
def test_crawl_endpoint_internal_error(mock_crawl_and_embed_api):
    async def mock_async_crawl_and_embed_error(*args, **kwargs):
        raise ValueError("Internal processing error")
    mock_crawl_and_embed_api.side_effect = mock_async_crawl_and_embed_error

    response = client.post("/crawl/", json={"url": "http://error.example.com"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal processing error"


# --- Test /get-urls/ endpoint ---
@mock.patch("doc_crawler.api.util_get_urls") # Patch where it's used
def test_get_urls_endpoint_success(mock_util_get_urls_api):
    async def mock_async_get_urls(*args, **kwargs):
        return ["http://example.com/url1", "http://example.com/url2"]
    mock_util_get_urls_api.side_effect = mock_async_get_urls

    test_start_url = "http://example.com/start"
    response = client.post(
        "/get-urls/",
        json={"start_url": test_start_url, "max_pages": 5}
    )
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["start_url"] == test_start_url
    assert json_response["found_urls"] == ["http://example.com/url1", "http://example.com/url2"]
    mock_util_get_urls_api.assert_called_once_with(start_url=test_start_url, max_pages=5)

@mock.patch("doc_crawler.api.util_get_urls")
def test_get_urls_endpoint_internal_error(mock_util_get_urls_api):
    async def mock_async_get_urls_error(*args, **kwargs):
        raise ConnectionError("Failed to connect to website")
    mock_util_get_urls_api.side_effect = mock_async_get_urls_error

    response = client.post("/get-urls/", json={"start_url": "http://broken.example.com"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to connect to website"


# --- Test /find-docs/ endpoint ---
@mock.patch("doc_crawler.api.util_find_docs") # Patch where it's used
def test_find_docs_endpoint_success(mock_util_find_docs_api):
    async def mock_async_find_docs(*args, **kwargs):
        return ["http://docs.example.com/projA", "http://wiki.example.com/projA"]
    mock_util_find_docs_api.side_effect = mock_async_find_docs

    test_project_id = "projectA_local_path_or_url"
    response = client.post(
        "/find-docs/",
        json={"project_identifier": test_project_id}
    )
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["project_identifier"] == test_project_id
    assert json_response["potential_doc_urls"] == ["http://docs.example.com/projA", "http://wiki.example.com/projA"]
    mock_util_find_docs_api.assert_called_once_with(project_path_or_prd_url=test_project_id)

@mock.patch("doc_crawler.api.util_find_docs")
def test_find_docs_endpoint_internal_error(mock_util_find_docs_api):
    async def mock_async_find_docs_error(*args, **kwargs):
        raise FileNotFoundError("PRD file not found")
    mock_util_find_docs_api.side_effect = mock_async_find_docs_error

    response = client.post("/find-docs/", json={"project_identifier": "/path/to/nonexistent_prd.md"})
    assert response.status_code == 500
    assert response.json()["detail"] == "PRD file not found"

# To run these tests:
# Ensure pytest and httpx are installed: pip install pytest httpx
# From the project root: pytest tests/test_api.py
# Or: python -m pytest tests/test_api.py
