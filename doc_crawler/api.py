# This file will contain the FastAPI application to expose crawling functionalities.
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
import asyncio
from typing import List, Optional

from .main import crawl_and_embed
from .utils import get_urls as util_get_urls
from .utils import find_docs as util_find_docs

# Default values from cli.py, consider a shared config module later
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
DEFAULT_QDRANT_COLLECTION = "ai_docs_api" # Slightly different default for API

app = FastAPI(
    title="Documentation Crawler API",
    description="API for crawling documentation, embedding content, and storing it in Qdrant.",
    version="0.1.0"
)

class CrawlRequest(BaseModel):
    url: HttpUrl
    qdrant_url: str = DEFAULT_QDRANT_URL
    qdrant_collection: str = DEFAULT_QDRANT_COLLECTION
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    max_pages_to_crawl: int = Field(default=1, ge=1, description="Maximum number of pages to crawl.")
    output_type: str = Field(default="markdown", pattern="^(markdown|text|html)$", description="Preferred output type from crawl4ai (markdown, text, html).")

    # New fields for chunking and filtering
    chunking_strategy: str = Field(default="paragraph", pattern="^(paragraph|sentence|fixed)$", description="Strategy for chunking content.")
    chunk_size: int = Field(default=512, ge=32, description="Target size for content chunks.") # Added ge=32 for sanity
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap size between chunks.")
    is_doc_filter_active: bool = Field(default=True, description="Enable filter to attempt to only process documentation-like pages.")

class CrawlResponse(BaseModel):
    message: str
    processed_url: HttpUrl
    qdrant_collection: str

class GetUrlsRequest(BaseModel):
    start_url: HttpUrl
    max_pages: int = Query(10, ge=1)

class GetUrlsResponse(BaseModel):
    start_url: HttpUrl
    found_urls: List[str]

class FindDocsRequest(BaseModel):
    project_identifier: str # Could be a URL or a string representing a local path concept

class FindDocsResponse(BaseModel):
    project_identifier: str
    potential_doc_urls: List[str]


@app.post("/crawl/", response_model=CrawlResponse)
async def crawl_endpoint(request: CrawlRequest):
    """
    Crawls a given URL, extracts content, chunks, embeds, and stores it.
    """
    try:
        await crawl_and_embed(
            initial_url=str(request.url), # Parameter name changed in crawl_and_embed
            qdrant_url=request.qdrant_url,
            qdrant_collection=request.qdrant_collection,
            embedding_model_name=request.embedding_model_name,
            ollama_base_url=request.ollama_base_url,
            max_pages_to_crawl=request.max_pages_to_crawl,
            output_type=request.output_type,
            chunking_strategy=request.chunking_strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            is_doc_filter_active=request.is_doc_filter_active
        )
        return CrawlResponse(
            message="Crawling and embedding process initiated successfully.",
            processed_url=request.url,
            qdrant_collection=request.qdrant_collection
        )
    except Exception as e:
        # Log the exception details for debugging
        print(f"Error during /crawl/ endpoint: {e}") # Basic logging
        # Consider logging traceback: import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-urls/", response_model=GetUrlsResponse)
async def get_urls_endpoint(request: GetUrlsRequest):
    """
    Fetches a list of relevant URLs starting from a given URL.
    (Currently uses placeholder implementation)
    """
    try:
        urls = await util_get_urls(start_url=str(request.start_url), max_pages=request.max_pages)
        return GetUrlsResponse(start_url=request.start_url, found_urls=urls)
    except Exception as e:
        print(f"Error during /get-urls/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find-docs/", response_model=FindDocsResponse)
async def find_docs_endpoint(request: FindDocsRequest):
    """
    Analyzes a project identifier (URL or path concept) to find documentation URLs.
    (Currently uses placeholder implementation)
    """
    try:
        doc_urls = await util_find_docs(project_path_or_prd_url=request.project_identifier)
        return FindDocsResponse(project_identifier=request.project_identifier, potential_doc_urls=doc_urls)
    except Exception as e:
        print(f"Error during /find-docs/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# To run this FastAPI app (save as api.py within doc_crawler directory):
# Ensure you are in the directory *above* doc_crawler
# Command: uvicorn doc_crawler.api:app --reload
# Then access docs at http://127.0.0.1:8000/docs

if __name__ == "__main__":
    # This part is for direct execution (e.g., python -m doc_crawler.api)
    # which is not how uvicorn typically runs it for development,
    # but can be useful for some deployment scenarios or simple tests.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
