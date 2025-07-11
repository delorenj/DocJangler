import logging
from typing import List, Optional

from llama_index.core import Document, VectorStoreIndex, StorageContext

# SentenceSplitter was unused, removed.
from llama_index.readers.web import SimpleWebPageReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding  # Using OpenAI embeddings by default for now
from qdrant_client import QdrantClient, models
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

from core.settings import qdrant_settings

logger = logging.getLogger(__name__)


class DocEmbedTool:
    def __init__(
        self, qdrant_url: str = qdrant_settings.url, qdrant_api_key: Optional[str] = qdrant_settings.api_key
    ):
        """
        Initializes the DocEmbedTool.

        Args:
            qdrant_url: URL for the Qdrant service.
            qdrant_api_key: API key for the Qdrant service (optional).
        """
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embed_model = OpenAIEmbedding()  # We can make this configurable later
        logger.info(f"DocEmbedTool initialized with Qdrant URL: {qdrant_url}")

    def _ensure_collection_exists(self, collection_name: str):
        """
        Ensures that the specified Qdrant collection exists. If not, it creates it.
        """
        try:
            self.qdrant_client.get_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' already exists.")
        except Exception as e:
            # A more specific exception would be better if Qdrant client provides one for "not found"
            logger.info(f"Collection '{collection_name}' not found, creating it. Error: {e}")
            # Determine vector size from the embedding model
            # This is a common pattern, but specific model details might vary.
            # For OpenAI's text-embedding-ada-002, it's 1536.
            # It's better to get this dynamically if possible, or make it a known constant for the chosen model.
            try:
                dummy_emb = self.embed_model.get_text_embedding("test")
                vector_size = len(dummy_emb)
            except Exception as emb_err:
                logger.warning(
                    f"Could not dynamically determine vector size from embedding model: {emb_err}. Defaulting to 1536."
                )
                vector_size = 1536  # Default for text-embedding-ada-002

            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            logger.info(f"Collection '{collection_name}' created with vector size {vector_size}.")

    def embed_from_query(self, query: str, collection_name: Optional[str] = None) -> dict:
        """
        Performs a web search for the query, fetches documentation, embeds it, and stores it in Qdrant.

        Args:
            query: The search term for the documentation.
            collection_name: The Qdrant collection to use. Defaults to `qdrant_settings.default_collection_name`.

        Returns:
            A dictionary containing the number of documents embedded and the collection name.
        """
        if collection_name is None:
            collection_name = qdrant_settings.default_collection_name

        logger.info(
            f"Starting documentation embedding for query: '{query}' into collection: '{collection_name}'"
        )

        self._ensure_collection_exists(collection_name)

        # 1. Search for relevant URLs
        search_tool = DuckDuckGoSearchToolSpec()
        try:
            # According to LlamaIndex docs, results are a list of dicts.
            # Each dict should have 'href' for URL if using ddg_search or similar from duckduckgo_search directly.
            # DuckDuckGoSearchToolSpec().duckduckgo_full_search returns list of dicts with 'link' and 'snippet'.
            search_results_raw = search_tool.duckduckgo_full_search(
                query=query, max_results=1
            )  # Fetch top 1 result

            if not search_results_raw:
                logger.warning(f"No search results found for query: '{query}'")
                return {
                    "processed_source_documents": 0,
                    "embedded_nodes": 0,
                    "collection_name": collection_name,
                    "status": "No search results",
                }

            # Ensure results are in the expected format (list of dicts with 'link')
            # The tool spec returns dicts with 'title', 'link', 'snippet' for duckduckgo_full_search
            urls_to_process = [result["link"] for result in search_results_raw if "link" in result]

            if not urls_to_process:
                logger.warning(
                    f"No valid URLs with 'link' key found in search results for query: '{query}'. Results: {search_results_raw}"
                )
                return {
                    "processed_source_documents": 0,
                    "embedded_nodes": 0,
                    "collection_name": collection_name,
                    "status": "No URLs in search results",
                }

            logger.info(f"Found URLs for query '{query}': {urls_to_process}")

        except Exception as e:
            logger.error(f"Error during web search for query '{query}': {e}")
            return {
                "processed_source_documents": 0,
                "embedded_nodes": 0,
                "collection_name": collection_name,
                "status": f"Search error: {e}",
            }

        # At this point, urls_to_process is valid and contains at least one URL, or we would have returned.

        # 2. Load documents from URLs
        try:
            loader = SimpleWebPageReader(html_to_text=True)
            documents: List[Document] = loader.load_data(urls=urls_to_process)
            if not documents:
                logger.warning(f"Could not load any documents from URLs: {urls_to_process}")
                return {
                    "embedded_documents": 0,
                    "collection_name": collection_name,
                    "status": "Failed to load documents from URLs",
                }
        except Exception as e:
            logger.error(f"Error loading documents from URLs {urls_to_process}: {e}")
            return {
                "embedded_documents": 0,
                "collection_name": collection_name,
                "status": f"Document loading error: {e}",
            }

        logger.info(f"Successfully loaded {len(documents)} document(s) from URLs.")

        # 3. Initialize QdrantVectorStore and StorageContext
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embed_model=self.embed_model,  # Pass the embedding model
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 4. Create VectorStoreIndex (this will chunk, embed, and insert)
        # Using a default splitter. Can be configured.
        # The index creation itself handles embedding and storing if documents are passed.
        try:
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                # embed_model is configured in QdrantVectorStore, so not strictly needed here.
                # If provided, it should be consistent. Removing for clarity to rely on VectorStore's config.
                show_progress=True,  # Helpful for logging
            )
            # LlamaIndex typically adds nodes to the vector store during index construction.
            # The number of nodes might be more indicative than raw documents due to chunking.
            # We can try to get a count of nodes added if the API supports it easily,
            # otherwise, len(documents) is a reasonable proxy for "sources processed".
            num_embedded_nodes = len(index.index_struct.nodes)  # This might give a count of nodes
            logger.info(
                f"Successfully embedded {num_embedded_nodes} nodes from {len(documents)} source document(s) into collection '{collection_name}'."
            )
            return {
                "processed_source_documents": len(documents),
                "embedded_nodes": num_embedded_nodes,
                "collection_name": collection_name,
                "status": "Success",
            }
        except Exception as e:
            logger.error(f"Error creating/updating VectorStoreIndex for collection '{collection_name}': {e}")
            return {
                "processed_source_documents": len(documents),  # Documents were loaded
                "embedded_nodes": 0,
                "collection_name": collection_name,
                "status": f"Indexing error: {e}",
            }


if __name__ == "__main__":
    # Basic test and usage example
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Ensure QDRANT_URL and OPENAI_API_KEY are set as environment variables for this test
    # For example:
    # export QDRANT_URL="http://localhost:6333"
    # export OPENAI_API_KEY="your_openai_api_key"
    # export QDRANT_DEFAULT_COLLECTION_NAME="my_test_docs"
    # (or it will use the default from core.settings)

    if not qdrant_settings.api_key and not OpenAIEmbedding().api_key:
        logger.warning("OPENAI_API_KEY environment variable not set. Embedding will likely fail.")
        # Note: OpenAIEmbedding checks for OPENAI_API_KEY env var by default.

    tool = DocEmbedTool()

    # Test Case 1: Successful embedding
    test_query = "FastAPI query parameters"
    logger.info(f"--- Test Case 1: Query: '{test_query}' ---")
    result = tool.embed_from_query(query=test_query, collection_name="fastapi_docs_test")
    logger.info(f"Embedding result: {result}")

    # Test Case 2: Search query with no results (hypothetical)
    test_query_no_results = "asdfqwerzxcvasdfqwerzxcv"
    logger.info(f"--- Test Case 2: Query (no results): '{test_query_no_results}' ---")
    result_no_results = tool.embed_from_query(query=test_query_no_results, collection_name="no_results_test")
    logger.info(f"Embedding result (no results): {result_no_results}")

    # Test Case 3: Using default collection name from settings
    # Ensure QDRANT_DEFAULT_COLLECTION_NAME is set or use the hardcoded default in settings.py
    default_collection_query = "Python type hints"
    logger.info(f"--- Test Case 3: Query (default collection): '{default_collection_query}' ---")
    result_default_collection = tool.embed_from_query(query=default_collection_query)
    logger.info(f"Embedding result (default collection): {result_default_collection}")

    # To verify in Qdrant, you would typically:
    # 1. Go to your Qdrant dashboard (e.g., http://localhost:6333/dashboard)
    # 2. Check the collections (e.g., "fastapi_docs_test", "no_results_test", qdrant_settings.default_collection_name)
    # 3. See if points (vectors) have been added to the collections.
    logger.info("--- Testing Complete ---")
    logger.info(f"Default collection for testing was: {qdrant_settings.default_collection_name}")

    # Example: How to list collections to see if they were created
    try:
        collections = tool.qdrant_client.get_collections()
        logger.info(f"Current Qdrant collections: {collections}")
    except Exception as e:
        logger.error(f"Could not connect to Qdrant to list collections: {e}")
