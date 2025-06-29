import asyncio
import logging
from typing import List

from .utils import (
    is_doc,
    get_urls as util_get_urls,
    chunk_content,
    embed_chunks,
    store_in_qdrant
)
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MarkdownGenerationStrategy

# Setup logger for this module
logger = logging.getLogger(__name__)
# Ensure basicConfig is called, preferably at application entry point or in utils if not already.
# For safety, can call it here too, but it's best practice to configure logging once.
if not logger.handlers: # Check if logger already has handlers
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


async def fetch_page_content(crawler: AsyncWebCrawler, target_url: str, output_type: str) -> str:
    """Helper to fetch content from a single page using crawl4ai."""
    logger.info(f"Fetching content from: {target_url} with output_type: {output_type}")
    try:
        # Configure for fetching content - primarily markdown
        # crawl4ai's DefaultMarkdownGenerator has `fit_markdown` which is good.
        # We need to ensure we get that.
        # The `output_type` parameter might map to specific strategies in crawl4ai.
        # For now, let's assume "markdown" means we want the best possible markdown.

        run_config = CrawlerRunConfig(
            # Use DefaultMarkdownGenerator which includes fit_markdown
            # To get fit_markdown, it's usually part of the result.markdown object.
            # result.markdown.fit_markdown
            # The `output_type` from user might need mapping if crawl4ai has specific enums.
            # For now, we assume it expects 'markdown', 'text', etc.
            # Let's rely on crawl4ai's default markdown generation which should be good.
            # If specific markdown strategy is needed:
            # markdown_generator=DefaultMarkdownGenerator(content_filter=PruningContentFilter(...))
            cache_mode=CacheMode.BYPASS # Or CacheMode.ENABLED for repeated runs during dev
        )

        result = await crawler.arun(url=target_url, config=run_config)

        if not result or not result.success:
            logger.error(f"Failed to crawl {target_url}. Result: {result}")
            return ""

        content_to_return = ""
        if output_type == "markdown":
            if result.markdown:
                # Prioritize fit_markdown if available and good, else raw_markdown
                if hasattr(result.markdown, 'fit_markdown') and result.markdown.fit_markdown and len(result.markdown.fit_markdown) > 10: # Heuristic for non-empty
                    content_to_return = result.markdown.fit_markdown
                    logger.info(f"Using fit_markdown for {target_url} (length: {len(content_to_return)})")
                elif hasattr(result.markdown, 'raw_markdown') and result.markdown.raw_markdown:
                    content_to_return = result.markdown.raw_markdown
                    logger.info(f"Using raw_markdown for {target_url} (length: {len(content_to_return)})")
                else: # Fallback if markdown object is structured differently or empty
                    content_to_return = str(result.markdown) # General fallback
                    logger.info(f"Using string representation of markdown object for {target_url} (length: {len(content_to_return)})")

            else:
                logger.warning(f"No markdown content found for {target_url}, though crawl was successful.")
        elif output_type == "text":
            content_to_return = result.text_content or ""
            logger.info(f"Using text_content for {target_url} (length: {len(content_to_return)})")
        elif output_type == "html":
            content_to_return = result.html_content or ""
            logger.info(f"Using html_content for {target_url} (length: {len(content_to_return)})")
        else:
            logger.warning(f"Unsupported output_type '{output_type}'. Defaulting to empty string.")

        return content_to_return

    except Exception as e:
        logger.error(f"Exception while fetching content from {target_url}: {e}", exc_info=True)
        return ""


async def crawl_and_embed(
    initial_url: str,
    qdrant_url: str,
    qdrant_collection: str,
    embedding_model_name: str,
    ollama_base_url: str,
    max_pages_to_crawl: int = 1,
    output_type: str = "markdown", # 'markdown', 'text', or 'html' (for is_doc)
    chunking_strategy: str = "paragraph",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    is_doc_filter_active: bool = True # New parameter to control is_doc usage
):
    """
    Orchestrates the crawling of URLs, processing content, embedding, and storing in Qdrant.
    """
    logger.info(f"Starting crawl and embed process for initial_url: {initial_url}")
    logger.info(f"Params: Qdrant='{qdrant_url}/{qdrant_collection}', Model='{embedding_model_name}', "
                f"Ollama='{ollama_base_url}', MaxPages='{max_pages_to_crawl}', Output='{output_type}', "
                f"Chunking='{chunking_strategy}/{chunk_size}/{chunk_overlap}', IsDocFilter='{is_doc_filter_active}'")

    urls_to_process: List[str] = []
    processed_urls_count = 0
    vector_size_overall = 0 # To store the determined vector size from the first successful embedding batch

    # Initialize crawl4ai browser
    browser_config = BrowserConfig(headless=True, verbose=False) # Keep verbose False for less noise

    async with AsyncWebCrawler(config=browser_config) as crawler:
        if max_pages_to_crawl > 1:
            logger.info(f"Deep crawling enabled. Discovering up to {max_pages_to_crawl} URLs from {initial_url}.")
            # util_get_urls uses its own crawler instance internally.
            # This is okay, but for larger scale, a shared crawler instance might be more efficient if possible.
            discovered_urls = await util_get_urls(start_url=initial_url, max_pages=max_pages_to_crawl)
            if discovered_urls:
                urls_to_process.extend(discovered_urls)
            if not urls_to_process: # If get_urls returns empty, fallback to initial_url
                 logger.warning(f"get_urls did not return any URLs from {initial_url}. Processing initial_url only.")
                 urls_to_process.append(initial_url)
            # Ensure initial_url is included if not already
            if initial_url not in urls_to_process:
                urls_to_process.insert(0, initial_url)
            urls_to_process = urls_to_process[:max_pages_to_crawl] # Respect max_pages
        else:
            urls_to_process.append(initial_url)

        logger.info(f"Total URLs to process: {len(urls_to_process)}. URLs: {urls_to_process}")

        for target_url in urls_to_process:
            if processed_urls_count >= max_pages_to_crawl:
                logger.info(f"Reached max_pages_to_crawl limit ({max_pages_to_crawl}). Stopping.")
                break

            logger.info(f"Processing URL ({processed_urls_count+1}/{len(urls_to_process)}): {target_url}")

            # Fetch content (HTML for is_doc, then preferred output_type for chunking)
            # If output_type is HTML, we can use it for both. Otherwise, fetch HTML first for is_doc.
            page_html_content = ""
            if is_doc_filter_active:
                page_html_content = await fetch_page_content(crawler, target_url, "html")
                if not page_html_content:
                    logger.warning(f"No HTML content fetched for {target_url} to check if it's a doc. Skipping is_doc check.")
                elif not is_doc(page_html_content):
                    logger.info(f"Page {target_url} classified as non-documentation by is_doc filter. Skipping.")
                    continue

            # Fetch the desired content type for processing (markdown, text)
            # If output_type was 'html' and used for is_doc, no need to re-fetch unless is_doc was skipped.
            content_for_chunking = ""
            if output_type == "html" and page_html_content: # Already fetched if is_doc was active
                content_for_chunking = page_html_content
            elif page_html_content and output_type != "html" and is_doc_filter_active: # HTML was fetched for is_doc, now get preferred type
                content_for_chunking = await fetch_page_content(crawler, target_url, output_type)
            else: # HTML not fetched or is_doc filter not active
                 content_for_chunking = await fetch_page_content(crawler, target_url, output_type)


            if not content_for_chunking or content_for_chunking.strip() == "":
                logger.warning(f"No content (type: {output_type}) to process for {target_url}. Skipping.")
                continue

            logger.info(f"Content fetched for {target_url} (length: {len(content_for_chunking)}). Type: {output_type}. Now chunking.")

            chunks = chunk_content(
                content_for_chunking,
                strategy=chunking_strategy,
                chunk_size=chunk_size,
                overlap=chunk_overlap
            )

            if not chunks:
                logger.warning(f"No chunks created for {target_url}. Skipping embedding and storage.")
                continue

            logger.info(f"Generated {len(chunks)} chunks for {target_url}. Now embedding.")

            embeddings, current_vector_size = await asyncio.to_thread(
                embed_chunks, chunks, embedding_model_name, ollama_base_url
            ) # embed_chunks is sync, run in thread

            if not embeddings or current_vector_size == 0:
                logger.error(f"Failed to generate embeddings or determine vector size for {target_url}. Skipping storage.")
                continue

            if vector_size_overall == 0:
                vector_size_overall = current_vector_size # Store the first valid vector size
            elif vector_size_overall != current_vector_size:
                logger.error(f"Inconsistent vector size detected for {target_url} ({current_vector_size}) "
                             f"compared to previous ({vector_size_overall}). This can corrupt Qdrant collection. Skipping storage for this URL.")
                continue # Critical error, skip storing these embeddings

            logger.info(f"Embeddings generated for {target_url} (count: {len(embeddings)}, vector_size: {vector_size_overall}). Now storing.")

            # Prepare metadata for Qdrant
            metadata_list = [{"source_url": target_url, "original_content_type": output_type} for _ in chunks]

            await asyncio.to_thread(
                store_in_qdrant,
                embeddings,
                chunks,
                qdrant_url,
                qdrant_collection,
                vector_size_overall, # Use the consistently determined vector size
                metadata_list
            ) # store_in_qdrant is sync

            logger.info(f"Successfully processed and stored content from {target_url}.")
            processed_urls_count += 1

    logger.info(f"Crawl and embed process finished. Processed {processed_urls_count} URLs.")


if __name__ == "__main__":
    # Example of how this might be called (for testing purposes)
    async def main_test():
        # Ensure Ollama and Qdrant are running for this test
        # Example: ollama serve
        # Example: docker run -p 6333:6333 qdrant/qdrant

        # Pull a model if you haven't: ollama pull nomic-embed-text
        test_embedding_model = "nomic-embed-text" # Ensure this model is available in your Ollama

        # A website that is likely to be simple and have some markdown-like content or just text
        # For a real doc site: "https://docs.python.org/3/tutorial/index.html" (might be heavy)
        # Using a simpler, more controllable target for a quick test:
        # test_url = "https://www.markdownguide.org/basic-syntax" # Good for markdown
        test_url = "https://www.iana.org/help/example-domains" # Simple text, few links

        logger.info("Starting main_test execution...")
        try:
            await crawl_and_embed(
                initial_url=test_url,
                qdrant_url="http://localhost:6333",
                qdrant_collection="test_doc_crawler_collection",
                embedding_model_name=test_embedding_model,
                ollama_base_url="http://localhost:11434", # Default Ollama URL
                max_pages_to_crawl=2, # Test with a small number of pages
                output_type="markdown", # Try to get markdown
                chunking_strategy="paragraph",
                chunk_size=256, # Smaller chunks for testing
                chunk_overlap=30,
                is_doc_filter_active=False # Disable for broad test URLs
            )
            logger.info("main_test execution completed.")
        except Exception as e:
            logger.error(f"Error during main_test: {e}", exc_info=True)

    # To run this test: python -m doc_crawler.main
    # asyncio.run(main_test())
    pass
