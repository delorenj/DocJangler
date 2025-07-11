from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional

from api.routes.playground import playground_router
from api.routes.health import health_check_router
from core.doc_embed_tool import DocEmbedTool
from core.settings import (
    qdrant_settings,
)  # To potentially access default collection name if needed, though tool handles it
import logging

logger = logging.getLogger(__name__)

v1_router = APIRouter(prefix="/v1")

# Include existing routers
v1_router.include_router(playground_router)
v1_router.include_router(health_check_router)


# --- New Route for Document Embedding ---
class DocEmbedRequest(BaseModel):
    query: str = Field(..., description="The search query for the documentation to embed.")
    collection_name: Optional[str] = Field(
        None, description="Optional Qdrant collection name. Uses default if not provided."
    )


class DocEmbedResponse(BaseModel):
    processed_source_documents: int
    embedded_nodes: int
    collection_name: str
    status: str
    message: Optional[str] = None


@v1_router.post("/embed-documentation", response_model=DocEmbedResponse, tags=["Documentation Embedding"])
async def embed_documentation_route(request: DocEmbedRequest = Body(...)):
    """
    Accepts a search query, finds relevant documentation online,
    embeds it, and stores it in a Qdrant vector collection.
    """
    try:
        # Initialize the tool. Qdrant URL and API key are taken from settings by default.
        # OPENAI_API_KEY needs to be in the environment for OpenAIEmbeddings.
        doc_embed_tool = DocEmbedTool()

        logger.info(
            f"Received request to embed documentation for query: '{request.query}' in collection: '{request.collection_name}'."
        )

        # DocEmbedTool.embed_from_query is synchronous. Calling it directly in an async def.
        # For production, consider using `run_in_threadpool`.
        result = doc_embed_tool.embed_from_query(
            query=request.query,
            collection_name=request.collection_name,  # Pass None if not provided, tool handles default
        )

        if result.get("status") == "Success":
            return DocEmbedResponse(**result)
        else:
            # status might contain error details from the tool
            logger.error(f"Embedding failed for query '{request.query}': {result.get('status')}")
            # Return a 200 with the error details from the tool, or choose to raise HTTPException
            return DocEmbedResponse(
                processed_source_documents=result.get("processed_source_documents", 0),
                embedded_nodes=result.get("embedded_nodes", 0),
                collection_name=result.get(
                    "collection_name", request.collection_name or qdrant_settings.default_collection_name
                ),
                status=result.get("status", "Unknown error"),
                message=f"Embedding process reported: {result.get('status')}",
            )

    except Exception as e:
        logger.exception(f"Unhandled error in /embed-documentation route for query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Note: The DocEmbedTool.embed_from_query is currently synchronous.
# For a production FastAPI app, CPU-bound or IO-bound tasks like web requests,
# and heavy computations (embeddings) should be run in a thread pool
# using `fastapi.concurrency.run_in_threadpool` or by making the underlying
# methods asynchronous if the libraries support it (LlamaIndex operations can sometimes be async).
# For this initial implementation, keeping it simple.
# Example modification for async execution if embed_from_query was naturally async:
# result = await doc_embed_tool.embed_from_query(...)
# If it's sync but we want to avoid blocking the main event loop:
# from fastapi.concurrency import run_in_threadpool
# result = await run_in_threadpool(doc_embed_tool.embed_from_query, query=request.query, collection_name=request.collection_name)

# For now, the `await` on `doc_embed_tool.embed_from_query` will cause a warning if it's not a true async def function.
# I will remove the `await` as `embed_from_query` is currently synchronous.
