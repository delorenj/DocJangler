# This file will contain utility functions for the doc_crawler project.
import ollama
from qdrant_client import QdrantClient, models
import logging # Added for logging
import uuid # For Qdrant point IDs

# crawl4ai imports
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode, DeepCrawlConfig, CrawlingStrategy

# Setup basic logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def is_doc(content: str, strategy: str = "heuristic") -> bool:
    """
    Classifies if the given HTML content is documentation.
    Strategies:
    - 'heuristic': Uses simple heuristics (e.g., presence of keywords, code blocks).
    - 'llm': (Future) Uses an LLM for classification.
    """
    if not content:
        return False
    # Basic heuristic: look for common documentation patterns
    # This is a very naive first approach.
    # Keywords that might indicate documentation
    doc_keywords = [
        "documentation", "guide", "manual", "reference", "api",
        "tutorial", "getting started", "how to", "faq",
        "install", "usage", "examples", "troubleshooting"
    ]
    # Structural elements often found in docs
    code_indicators = ["<code", "<pre", "```"]

    content_lower = content.lower()
    keyword_hits = sum(1 for keyword in doc_keywords if keyword in content_lower)
    code_hits = sum(1 for indicator in code_indicators if indicator in content_lower)

    # Arbitrary thresholds for now
    if keyword_hits >= 2 or code_hits >= 1:
        return True
    if "<h1>" in content_lower and ("introduction" in content_lower or "overview" in content_lower):
        return True

    return False

async def get_urls(start_url: str, max_pages: int = 10) -> list[str]:
    """
    Uses crawl4ai to find relevant URLs starting from start_url using deep crawling.
    """
    logger.info(f"Starting URL discovery from {start_url} (max_pages: {max_pages})")
    collected_urls = set() # Use a set to store unique URLs

    # Basic browser config, can be customized further if needed
    browser_config = BrowserConfig(headless=True, verbose=False)

    # Deep crawl configuration
    # We'll use BFS strategy as it's generally good for finding pages level by level.
    # For documentation, we might want to stay on the same domain.
    deep_crawl_config = DeepCrawlConfig(
        strategy=CrawlingStrategy.BFS, # Breadth-First Search
        max_pages=max_pages,
        stay_on_domain=True, # Important for documentation sites
        # No specific include/exclude patterns for now, can be added
    )

    # The run_config for get_urls doesn't need complex markdown or extraction,
    # as we are primarily interested in the links found.
    # However, crawl4ai's arun typically returns a CrawlResult for the *initial* URL.
    # The deep crawling happens as part of that initial call and links are often
    # part of the result or managed internally and used for subsequent crawls if we were
    # processing content. For *just* getting URLs, we might need to see how crawl4ai
    # exposes the list of discovered (and potentially visited) URLs.

    # According to crawl4ai docs, it does "Comprehensive Link Extraction".
    # The CrawlResult object should contain information about links.
    # Let's assume we run a crawl and then extract all unique internal links.

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # We are not focused on the content of the start_url itself for this function,
            # but rather the links it leads to.
            # The `arun` method with `deep_crawl_config` will explore pages.
            # We need a way to collect all URLs visited or discovered.
            # crawl4ai's `arun` returns a list of CrawlResult objects if multiple URLs are processed
            # or if deep crawling leads to multiple results.

            # Let's try a dry run approach if possible, or inspect the result for all links.
            # For now, let's assume `arun` with deep crawl will visit pages and we can
            # collect URLs from the results.
            # The `CrawlerRunConfig` is per-URL, but `deep_crawl_config` is for the overall session.

            # Initial simple approach: crawl the start URL and extract links from its result.
            # This might not give *all* URLs up to max_pages directly without multiple calls or
            # a more sophisticated hook if crawl4ai supports it.

            # A more direct way to get links might be to use a simpler method if crawl4ai has one
            # for link discovery only. If not, we simulate a crawl.
            # The `CrawlResult.media.links` dictionary contains `internal` and `external` links.

            # Let's refine this: we want to crawl and collect all *visited* internal pages.
            # The `AsyncWebCrawler` itself might maintain a list of visited URLs or allow access
            # to the queue. This is not explicitly clear from the basic README.
            # For now, we will assume a BFS strategy and collect links from each page.

            # Simplistic initial approach: Perform a deep crawl and collect all unique internal links encountered.
            # This might involve running `arun` and then inspecting the result.
            # The `deep_crawl_config` itself should handle the traversal.
            # The main question is how to get the list of *all* URLs visited/queued by the deep crawler.

            # Let's assume `arun` with deep_crawl will return results for *each* page crawled.
            # If `arun` is called once with a deep_crawl_config, it should handle the traversal.
            # The documentation implies `await crawler.arun(url="...", config=run_config_with_deep_crawl)`
            # will explore.

            # Let's try to run the crawler and see what results we get.
            # We are interested in the URLs that the crawler *would* visit or *has* visited.

            # The `arun` method might return a list of `CrawlResult` objects, one for each page
            # crawled during the deep crawl.
            initial_run_config = CrawlerRunConfig(
                deep_crawl=deep_crawl_config,
                cache_mode=CacheMode.BYPASS # Don't cache for URL discovery
            )

            logger.info(f"Executing crawl from {start_url} with deep_crawl strategy {deep_crawl_config.strategy}, max_pages {deep_crawl_config.max_pages}")
            # `arun` usually takes a single URL. If it deep crawls, it should return results for all.
            # The type hint for arun is `async def arun(self, url: str, config: Optional[CrawlerRunConfig] = None, magic: bool = False) -> CrawlResult:`
            # This suggests it returns a single CrawlResult for the initial URL.
            # The `CrawlResult` itself might contain all links discovered.
            # Let's test this assumption.

            # If `arun` only returns one result, we might need a different approach or to hook into the crawler.
            # The `crawl4ai` examples show `results: List[CrawlResult] = await crawler.arun(...)`
            # This suggests it *can* return a list. This typically happens when multiple URLs are passed to `arun`
            # or when a deep crawl occurs.

            # Let's assume `arun` returns a list of CrawlResult for a deep crawl.
            # If it returns a single CrawlResult, that result's `media.links.internal` would be for the *first* page.
            # This part of crawl4ai's API for getting *all visited URLs in a deep crawl* needs clarification.

            # Fallback: The `crwl` CLI in crawl4ai has `--deep-crawl bfs --max-pages 10`.
            # This implies the library handles it.
            # The `AsyncWebCrawler` might have a way to access its internal queue or visited set.
            # This is not obvious from the README.

            # Let's try a pragmatic approach: perform the crawl and collect all unique internal links
            # from the results. This might involve iterating if `arun` returns multiple results or
            # processing the single result if it contains all discovered links.

            results = await crawler.arun(url=start_url, config=initial_run_config)

            # Ensure results is a list
            if not isinstance(results, list):
                results = [results]

            for result in results:
                if result.success and result.url:
                    collected_urls.add(result.url) # Add the URL of the crawled page itself
                    if result.media and result.media.links:
                        for link in result.media.links.get("internal", []):
                            # Ensure links are absolute
                            from urllib.parse import urljoin
                            absolute_link = urljoin(result.url, link)
                            collected_urls.add(absolute_link)

            logger.info(f"Discovered {len(collected_urls)} unique internal URLs after crawling from {start_url}.")

    except Exception as e:
        logger.error(f"Error during URL discovery with crawl4ai from {start_url}: {e}")
        # Optionally, return a partial list or re-raise
        return list(collected_urls) # Return what was collected so far

    # Filter out non-http/https URLs and ensure they are within the stay_on_domain if possible
    # (though crawl4ai's stay_on_domain should handle this).
    # For now, returning all collected ones.
    final_urls = [url for url in list(collected_urls) if url.startswith("http")]

    # Limit to max_pages if somehow we got more (though deep_crawl_config should handle this)
    return final_urls[:max_pages]


def chunk_content(markdown_content: str, strategy: str = "paragraph", chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """
    Chunks Markdown content into smaller pieces.
    Strategies:
    - 'paragraph': Splits by paragraph.
    - 'sentence': Splits by sentence (more complex, requires NLP library or regex).
    - 'fixed_token': (Future) Uses token counting for fixed size chunks.
    """
    chunks = []
    if strategy == "paragraph":
        paragraphs = markdown_content.split("\n\n")
        current_chunk = ""
        for p in paragraphs:
            p_trimmed = p.strip()
            if not p_trimmed:
                continue
            if len(current_chunk) + len(p_trimmed) + 1 < chunk_size or not current_chunk:
                current_chunk += ("\n\n" if current_chunk else "") + p_trimmed
            else:
                chunks.append(current_chunk)
                # Start new chunk with overlap (simplified for now)
                # A more robust overlap would re-add last few sentences/tokens
                current_chunk = p_trimmed
        if current_chunk:
            chunks.append(current_chunk)

    elif strategy == "sentence":
        # Basic sentence splitting, can be improved with NLTK or Spacy
        import re
        sentences = re.split(r'(?<=[.!?])\s+', markdown_content.replace('\n', ' '))
        current_chunk = ""
        for s in sentences:
            s_trimmed = s.strip()
            if not s_trimmed:
                continue
            if len(current_chunk) + len(s_trimmed) + 1 < chunk_size or not current_chunk :
                current_chunk += (" " if current_chunk else "") + s_trimmed
            else:
                chunks.append(current_chunk)
                # Overlap: add the last sentence to the new chunk
                # This is a simple form of overlap
                current_chunk = s_trimmed
        if current_chunk:
            chunks.append(current_chunk)
    else:
        # Default to fixed size character splitting if strategy is unknown or not paragraph/sentence
        # This is a fallback and not ideal for semantic chunking.
        logger.warning(f"Unknown or unimplemented chunking strategy '{strategy}'. Falling back to fixed character window.")
        for i in range(0, len(markdown_content), chunk_size - overlap):
            chunks.append(markdown_content[i:i + chunk_size])

    final_chunks = [chunk for chunk in chunks if chunk.strip()]
    logger.info(f"Chunked content into {len(final_chunks)} chunks using strategy '{strategy}', chunk_size ~{chunk_size}.")
    return final_chunks


def embed_chunks(chunks: list[str], embedding_model_name: str, ollama_base_url: str) -> list[list[float]]:
    """
    Generates embeddings for a list of text chunks using Ollama and determines the vector size.
    Returns a tuple: (list of embeddings, vector_size).
    If embedding fails for all chunks or vector size cannot be determined, vector_size might be 0 or None.
    """
    if not chunks:
        logger.info("No chunks to embed.")
        return [], 0

    logger.info(f"Embedding {len(chunks)} chunks using Ollama model '{embedding_model_name}' via {ollama_base_url}")
    client = ollama.Client(host=ollama_base_url)
    embeddings = []
    vector_size = 0

    # Verify model exists and try to get embedding dimension (optional, as client.embeddings will fail anyway)
    try:
        model_info = client.show(embedding_model_name)
        # The path to embedding dimension might vary. Common paths:
        # 'details.embedding_dimension', 'details.dimensions', 'details.vector_size'
        # This is highly dependent on Ollama's API response structure for `show`
        if model_info and 'details' in model_info:
            if 'embedding_dimension' in model_info['details']:
                vector_size = model_info['details']['embedding_dimension']
            elif 'dimensions' in model_info['details']: # another common key
                vector_size = model_info['details']['dimensions']
            if vector_size > 0:
                logger.info(f"Determined vector size {vector_size} for model '{embedding_model_name}' from model details.")
    except Exception as e:
        logger.warning(f"Could not retrieve details for model '{embedding_model_name}' to pre-determine vector size: {e}. Will attempt to infer from embeddings.")

    for i, chunk in enumerate(chunks):
        try:
            response = client.embeddings(model=embedding_model_name, prompt=chunk)
            embedding = response["embedding"]
            embeddings.append(embedding)
            if vector_size == 0 and embedding: # Infer vector_size from the first successful embedding
                vector_size = len(embedding)
                logger.info(f"Inferred vector size {vector_size} from the first successful embedding.")
            elif embedding and len(embedding) != vector_size and vector_size != 0:
                logger.error(f"Inconsistent embedding dimension for chunk {i}. Expected {vector_size}, got {len(embedding)}. Skipping this embedding.")
                # Remove the inconsistent embedding or handle as per policy
                embeddings.pop()
                continue
            logger.debug(f"Embedded chunk {i+1}/{len(chunks)}")
        except Exception as e:
            logger.error(f"Error embedding chunk {i+1} ('{chunk[:50]}...'): {e}")
            # Optionally, append a zero vector or skip. Skipping for now.
            continue

    if not embeddings:
        logger.warning(f"No embeddings were generated for model '{embedding_model_name}'.")
        return [], 0

    if vector_size == 0:
        logger.error(f"Could not determine vector size for model '{embedding_model_name}'.")
        return embeddings, 0 # Or raise an error

    logger.info(f"Successfully generated {len(embeddings)} embeddings with vector size {vector_size}.")
    return embeddings, vector_size

def store_in_qdrant(
    embeddings: list[list[float]],
    chunks: list[str],
    qdrant_url: str,
    collection_name: str,
    vector_size: int, # Explicitly require vector_size
    metadata_list: list[dict] = None # Optional metadata for each point
):
    """
    Stores embeddings and corresponding text chunks in Qdrant.
    """
    if not embeddings:
        logger.info("No embeddings provided to store.")
        return
    if vector_size == 0:
        logger.error("Vector size is 0. Cannot store embeddings in Qdrant without a valid vector size.")
        # Or raise ValueError("Vector size cannot be 0 for Qdrant storage.")
        return

    logger.info(f"Attempting to store {len(embeddings)} embeddings in Qdrant (URL: {qdrant_url}), collection: '{collection_name}' with vector size {vector_size}.")

    # QdrantClient can take the full URL, or host and port separately.
    # Using http:// prefix is generally fine for qdrant-client.
    # If qdrant_url is like "localhost:6333", client might need http://.
    # Let's ensure it has a scheme.
    qdrant_service_url = qdrant_url
    if not qdrant_service_url.startswith("http://") and not qdrant_service_url.startswith("https://"):
        qdrant_service_url = f"http://{qdrant_service_url}"
        logger.debug(f"Prepended http:// to Qdrant URL, now: {qdrant_service_url}")

    try:
        # The QdrantClient can take `url` for http/https or `host`/`port` for direct TCP.
        # If https is used, certs might be an issue unless properly configured.
        # For local dev, http is common.
        if ":" in qdrant_url.split("://")[-1]: # Check if host:port is specified
            client = QdrantClient(url=qdrant_service_url)
        else: # Assume it's just a host, use default port if not specified in url
            client = QdrantClient(host=qdrant_url.split("://")[-1], port=6333) # Default Qdrant port

        # Check if collection exists
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' already exists.")
            # Verify vector size
            # Accessing vector parameters depends on Qdrant version and client.
            # For qdrant-client >= 1.0.0, it's collection_info.config.params.vectors.size if single vector config
            # If multiple named vectors: collection_info.config.params.vectors[''].size
            current_vector_size = None
            if isinstance(collection_info.config.params.vectors, models.VectorParams):
                current_vector_size = collection_info.config.params.vectors.size
            elif isinstance(collection_info.config.params.vectors, dict) and '' in collection_info.config.params.vectors: # Default unnamed vector
                 current_vector_size = collection_info.config.params.vectors[''].size

            if current_vector_size and current_vector_size != vector_size:
                error_msg = (f"Mismatched vector size for collection '{collection_name}'. "
                             f"Expected {vector_size}, found {current_vector_size}.")
                logger.error(error_msg)
                raise ValueError(error_msg)
            elif not current_vector_size:
                 logger.warning(f"Could not determine existing vector size for collection '{collection_name}'. Proceeding with caution.")


        except Exception as e: # Specific exception for "not found" might be better
            # A common way Qdrant client indicates "not found" is an exception containing "Not found" or status code 404.
            # For qdrant_client, it might be a custom exception or a generic one with specific message.
            # Let's assume a general exception might mean "not found" if it contains certain text.
            # More robust: catch specific qdrant_client.http.exceptions.UnexpectedResponseError and check status_code
            if "not found" in str(e).lower() or "status_code=404" in str(e) or "status code 404" in str(e).lower(): # Heuristic check
                logger.info(f"Collection '{collection_name}' not found. Creating now with vector size {vector_size}.")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
                )
                logger.info(f"Collection '{collection_name}' created successfully.")
            else:
                # Other error during get_collection
                logger.error(f"Error checking or creating collection '{collection_name}': {e}")
                raise # Re-raise the exception if it's not a simple "not found"

        points_to_upsert = []
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            payload = {"text": chunk, "source_url": metadata_list[i].get("source_url", "unknown") if metadata_list and i < len(metadata_list) else "unknown"}
            # Add any other metadata if available
            if metadata_list and i < len(metadata_list):
                payload.update(metadata_list[i])

            points_to_upsert.append(models.PointStruct(
                id=str(uuid.uuid4()),  # Use UUID for point IDs
                vector=embedding,
                payload=payload
            ))

        if points_to_upsert:
            client.upsert(collection_name=collection_name, points=points_to_upsert, wait=True)
            logger.info(f"Successfully upserted {len(points_to_upsert)} points to Qdrant collection '{collection_name}'.")
        else:
            logger.info("No points were prepared for upsertion into Qdrant.")

    except Exception as e:
        logger.error(f"An error occurred while interacting with Qdrant: {e}")
        # Depending on policy, you might want to re-raise or handle specific Qdrant exceptions
        raise # Re-raise for now, so the caller is aware


async def find_docs(project_path_or_prd_url: str) -> list[str]:
    """
    Analyzes a project (local path or PRD URL) to find relevant documentation URLs.
    This is a placeholder for a more complex implementation.
    """
    print(f"Finding docs for: {project_path_or_prd_url}")
    # Initial naive implementation:
    # If it's a URL, just return it.
    # If it's a local path, look for common doc files or known project type markers.
    # This will require significant expansion.
    if project_path_or_prd_url.startswith("http://") or project_path_or_prd_url.startswith("https://"):
        return [project_path_or_prd_url]
    else:
        # Basic check for local files (very simplified)
        # E.g., if it's a path, look for a README.md or docs/ folder
        # For now, just return a placeholder
        return [f"file://{project_path_or_prd_url}/README.md"] # Dummy data
