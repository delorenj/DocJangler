## Agent App

This repo contains the code for running an agent-app and supports 2 environments:

1. **dev**: A development environment running locally on docker
2. **prd**: A production environment running on AWS ECS

## Setup Workspace

1. [Install uv](https://docs.astral.sh/uv/#getting-started): `curl -LsSf https://astral.sh/uv/install.sh | sh`

> from the `agent-app` dir:

2. Install workspace and activate the virtual env:

```sh
./scripts/install.sh
source .venv/bin/activate
```

3. Setup workspace:

```sh
phi ws setup
```

4. Copy `workspace/example_secrets` to `workspace/secrets`:

```sh
cp -r workspace/example_secrets workspace/secrets
```

5. Optional: Create `.env` file:

```sh
cp example.env .env
```

## Run Agent App locally

1. Install [docker desktop](https://www.docker.com/products/docker-desktop)

2. Set OpenAI Key

Set the `OPENAI_API_KEY` environment variable using

```sh
export OPENAI_API_KEY=sk-***
```

**OR** set in the `.env` file. Make sure to uncomment it.

### 2.1. Setup Qdrant (for Document Embedding Tool)

The document embedding tool requires a running Qdrant vector database instance.

**Run Qdrant using Docker:**

```sh
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```
This command mounts a local directory `./qdrant_storage` for persistence. Create this directory if it doesn't exist.
Qdrant will be accessible at `http://localhost:6333`.

**Configure Qdrant in `.env` file:**

Ensure the following variables are set in your `.env` file (copy from `example.env` if needed and uncomment/edit):

```env
# --- Qdrant Configuration (for DocEmbedTool) ---
QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY="" # Set if your Qdrant instance requires an API key
QDRANT_DEFAULT_COLLECTION_NAME="default_repo_docs"
```

If you are running the integration tests, they are configured to use `QDRANT_URL="http://localhost:6334"` by default to avoid conflicts with a development instance on port 6333. You can start a separate Qdrant instance for tests on port 6334 or override `TEST_QDRANT_URL` environment variable.

3. Start the workspace using:

```sh
phi ws up
```

- Open [localhost:8501](http://localhost:8501) to view the Streamlit App.
- Open [localhost:8000/docs](http://localhost:8000/docs) to view the FastAPI docs.

### Using the Document Embedding API

Once the application and Qdrant are running, you can use the new document embedding tool via its API endpoint.

**Endpoint:** `POST /api/v1/embed-documentation`

**Request Body (JSON):**

```json
{
  "query": "Your technical search query",
  "collection_name": "optional_custom_collection_name"
}
```

- `query` (string, required): The search term for the documentation you want to find and embed.
- `collection_name` (string, optional): The name of the Qdrant collection to store the embeddings in. If not provided, uses the default collection name configured in `QDRANT_DEFAULT_COLLECTION_NAME` (e.g., `default_repo_docs`).

**Example using `curl`:**

```sh
curl -X POST "http://localhost:8000/api/v1/embed-documentation" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "FastAPI background tasks",
           "collection_name": "fastapi_specific_docs"
         }'
```

**Successful Response (Example):**

```json
{
  "processed_source_documents": 1,
  "embedded_nodes": 15,
  "collection_name": "fastapi_specific_docs",
  "status": "Success",
  "message": null
}
```

If the tool encounters an issue (e.g., no search results found, unable to fetch webpage), the `status` field will indicate the problem, and `processed_source_documents` or `embedded_nodes` might be 0.

4. Stop the workspace using:

```sh
phi ws down
```

## Next Steps:

- [Run the Agent App on AWS](https://docs.phidata.com/templates/agent-app/run-aws)
- Read how to [manage the development application](https://docs.phidata.com/how-to/development-app)
- Read how to [manage the production application](https://docs.phidata.com/how-to/production-app)
- Read how to [add python libraries](https://docs.phidata.com/how-to/python-libraries)
- Read how to [format & validate your code](https://docs.phidata.com/how-to/format-and-validate)
- Read how to [manage secrets](https://docs.phidata.com/how-to/secrets)
- Add [CI/CD](https://docs.phidata.com/how-to/ci-cd)
- Add [database tables](https://docs.phidata.com/how-to/database-tables)
- Read the [Agent App guide](https://docs.phidata.com/templates/agent-app)
