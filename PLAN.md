# self-hosted pipeline to scrape a site, chunk content for RAG, and store it in qdrant

## The Stack

- Firecrawl as the self-hosted web crawler
- LlamaIndex as the self-hosted RAG orchestration layer
- Ollama with nomic-embed-large model for self-hosted embeddings
- Qdrant as the self-hosted vector database.
- FastMCP to serve communication tools to the LLM client (Claude Code, Gemini CLI, etc.)
- Gradio to serve a UI for dev and testing
- Docker Compose to deploy the stack


## The Implementation

> Test Target: https://gofastmcp.com/

### Phase 1: Scraping and ingestion pipeline

	1. Write a Python script that uses Firecrawl to scrape the target site.
	2. Using LlamaIndex, pass the Markdown content through a text splitter to create chunks. 
	3. Use Ollama's embedding model to convert text chunks into vectors. 
	4. Connect to your self-hosted Qdrant instance to store the text chunks and their associated vector embeddings. 

### Phase 2: Question-answering flow

	• The user submits a query through Gradio.
	• The query is passed to the RAG application backend, which uses Ollama to create an embedding of the query. 
	• Your backend queries the self-hosted Qdrant to find the most relevant document chunks based on vector similarity. 
	• The retrieved context and the user's original query are combined into a final prompt and sent back to Gradio to see.
	• The LLM generates an answer based on the provided context, which is returned to the user via the Open WebUI. [26, 27, 28, 29, 30]  

AI responses may include mistakes.

[1] https://ardor.cloud/blog/ai-powered-web-scraping-with-rag[2] https://jeroenjanssens.com/dsatcl/chapter-5-scrubbing-data[3] https://pub.towardsai.net/build-a-reliable-rag-agent-that-can-scrape-any-website-e366eb5bc197[4] https://medium.com/mitb-for-all/would-you-like-context-with-that-a-rag-as-a-service-drive-thru-2e5a2867102a[5] https://www.firecrawl.dev/blog/deepseek-rag-documentation-assistant[6] https://ardor.cloud/blog/ai-powered-web-scraping-with-rag[7] https://mehmetozkaya.medium.com/semantic-search-development-with-c-using-ollama-vectordb-orchestrate-in-net-aspire-d82eec73696a[8] https://www.reddit.com/r/LangChain/comments/1cf2dwc/what_web_scraper_for_web_search_agent/[9] https://github.com/unclecode/crawl4ai[10] https://docs.crawl4ai.com/[11] https://lightning.ai/lightning-ai/studios/self-hosted-rag-app-using-cohere-s-r[12] https://lightning.ai/lightning-ai/studios/self-hosted-rag-app-using-cohere-s-r[13] https://zilliz.com/learn/beginner-guide-to-website-chunking-and-embedding-for-your-genai-applications[14] https://medium.com/@bukowski.daniel/you-built-a-rag-proof-of-concept-now-what-e73799bbddcf[15] https://wandb.ai/byyoung3/rag-eval/reports/RAG-vs-prompt-stuffing-Do-we-still-need-vector-retrieval---VmlldzoxMzE5Mjk0NA[16] https://medium.com/@fadil.parves/qdrant-self-hosted-28a30106e9dd[17] https://www.reddit.com/r/Rag/comments/1jwmrz8/how_to_scrape_websites_into_an_inhouse_db_for_rag/[18] https://qdrant.tech/documentation/agentic-rag-langgraph/[19] https://lightning.ai/lightning-ai/studios/self-hosted-rag-app-using-cohere-s-r[20] https://railway.com/deploy/i1tz3T[21] https://www.reddit.com/r/LangChain/comments/1gb496k/best_tutorial_or_tech_stack_for_a_production_rag/[22] https://www.reddit.com/r/selfhosted/comments/1icu6jp/build_a_local_rag_using_deepseekr1_langchain_and/[23] https://www.pondhouse-data.com/blog/introduction-to-open-web-ui[24] https://medium.com/@hassan.tbt1989/build-a-rag-powered-llm-service-with-ollama-open-webui-a-step-by-step-guide-a688ec58ac97[25] https://sliplane.io/blog/5-awesome-open-webui-alternatives[26] https://www.youtube.com/watch?v=c5dw_jsGNBk[27] https://github.com/unclecode/crawl4ai/discussions/710[28] https://arxiv.org/html/2407.11987v1[29] https://docs.e2enetworks.com/docs/tir/FoundationStudio/GenAI_API/tutorials/rag_with_llm/[30] https://zeeshankhawar.medium.com/connecting-chatgpt-with-your-own-data-using-llama-index-and-langchain-74ba79fb7429

