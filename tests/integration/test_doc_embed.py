import pytest
from fastapi.testclient import TestClient
from httpx import Response  # For type hinting
import os

# Adjust the import path to your FastAPI application instance
# This assumes your main FastAPI app instance is named 'app' in 'api.main'
from api.main import app

# Create a TestClient instance using your FastAPI app
client = TestClient(app)

# Define a common test query and collection name
TEST_QUERY = "What are FastAPI dependencies?"
TEST_COLLECTION_NAME = "test_doc_embed_collection"


# Helper function to check Qdrant (optional, can be expanded)
def check_qdrant_collection_exists(collection_name: str):
    # This is a placeholder. Actual implementation would require a Qdrant client.
    # For now, we're focusing on the API response.
    # from qdrant_client import QdrantClient
    # qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    # try:
    #     q_client = QdrantClient(url=qdrant_url)
    #     q_client.get_collection(collection_name)
    #     return True
    # except Exception:
    #     return False
    return True  # Assume exists for now for test flow


@pytest.fixture(scope="module", autouse=True)
def manage_test_environment():
    """
    Set up environment variables for testing.
    Clean up Qdrant collection after tests (if created).
    """
    # Set dummy OpenAI API key if not present, as embeddings might try to init
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    if not original_openai_key:
        os.environ["OPENAI_API_KEY"] = "sk-testdummykeyfortesting12345"

    # Set Qdrant vars for test if not present (core.settings will pick these up)
    original_qdrant_url = os.environ.get("QDRANT_URL")
    original_qdrant_coll = os.environ.get("QDRANT_DEFAULT_COLLECTION_NAME")

    # Use a different port for test Qdrant to avoid conflict with dev instance
    os.environ["QDRANT_URL"] = os.getenv("TEST_QDRANT_URL", "http://localhost:6334")
    os.environ["QDRANT_DEFAULT_COLLECTION_NAME"] = "test_default_ci_collection"

    yield  # This is where the testing happens

    # Teardown: Restore original environment variables
    # Qdrant cleanup is complex and best handled by ephemeral test environments.
    # Here, we just restore environment variables.
    if original_openai_key is None:
        del os.environ["OPENAI_API_KEY"]
    else:
        os.environ["OPENAI_API_KEY"] = original_openai_key

    if original_qdrant_url is None:
        if "QDRANT_URL" in os.environ:
            del os.environ["QDRANT_URL"]
    else:
        os.environ["QDRANT_URL"] = original_qdrant_url

    if original_qdrant_coll is None:
        if "QDRANT_DEFAULT_COLLECTION_NAME" in os.environ:
            del os.environ["QDRANT_DEFAULT_COLLECTION_NAME"]
    else:
        os.environ["QDRANT_DEFAULT_COLLECTION_NAME"] = original_qdrant_coll


def test_embed_documentation_success_custom_collection():
    """
    Test the /embed-documentation endpoint with a custom collection name.
    """
    if not os.getenv("QDRANT_URL"):  # Check if Qdrant is meant to be available
        pytest.skip("QDRANT_URL not set by environment or fixture, skipping Qdrant integration test.")

    response: Response = client.post(
        "/api/v1/embed-documentation", json={"query": TEST_QUERY, "collection_name": TEST_COLLECTION_NAME}
    )

    assert response.status_code == 200
    data = response.json()

    assert data["collection_name"] == TEST_COLLECTION_NAME
    # Success of this operation depends on external services (DuckDuckGo, website, OpenAI API)
    # The test verifies the API responds correctly based on the tool's outcome.
    assert "status" in data
    if data["status"] == "Success":
        assert data["processed_source_documents"] > 0
        assert data["embedded_nodes"] > 0
    else:
        print(f"Test for custom collection passed with tool status: {data['status']}")


def test_embed_documentation_success_default_collection():
    """
    Test the /embed-documentation endpoint using the default collection name.
    """
    if not os.getenv("QDRANT_URL"):
        pytest.skip("QDRANT_URL not set, skipping Qdrant integration test.")

    response: Response = client.post(
        "/api/v1/embed-documentation", json={"query": "What is Python global interpreter lock?"}
    )
    assert response.status_code == 200
    data = response.json()

    # The fixture sets QDRANT_DEFAULT_COLLECTION_NAME to "test_default_ci_collection"
    expected_default_collection = os.environ.get("QDRANT_DEFAULT_COLLECTION_NAME")
    assert data["collection_name"] == expected_default_collection
    assert "status" in data
    if data["status"] == "Success":
        assert data["processed_source_documents"] > 0
        assert data["embedded_nodes"] > 0
    else:
        print(f"Test for default collection passed with tool status: {data['status']}")


def test_embed_documentation_invalid_query_handling():
    """
    Test how the endpoint handles a query that likely won't return search results.
    """
    if not os.getenv("QDRANT_URL"):
        pytest.skip("QDRANT_URL not set, skipping Qdrant integration test.")

    response: Response = client.post(
        "/api/v1/embed-documentation", json={"query": "asdfjklasdfjklasdfjklasdfjklasdfjkl"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] != "Success"
    assert data["processed_source_documents"] == 0
    assert data["embedded_nodes"] == 0


def test_embed_documentation_missing_query_param():
    """
    Test request validation: query parameter is missing.
    """
    response: Response = client.post(
        "/api/v1/embed-documentation",
        json={"collection_name": TEST_COLLECTION_NAME},  # Missing 'query'
    )
    assert response.status_code == 422  # Unprocessable Entity for Pydantic validation error


# General notes for running these tests:
# - Qdrant should be running (the fixture directs tests to QDRANT_URL, default http://localhost:6334).
# - OPENAI_API_KEY should be set (fixture provides a dummy if not present, but real embeddings will fail).
# - Run with `pytest tests/integration/test_doc_embed.py` from the project root.
# - Full "Success" status in responses depends on live external services. Tests check structural correctness
#   and graceful error handling by the tool (e.g., "No search results").
# - Qdrant collection cleanup is not performed by this test script; manage test Qdrant instances externally.
