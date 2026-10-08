# AI Engineering Bootcamp — RAG Shopping Assistant

A shopping assistant that answers questions using product descriptions from the Amazon Reviews 2023 Electronics dataset. The project includes a Streamlit chat UI, a FastAPI backend, a Qdrant vector database, preparation notebooks, and a LangSmith evaluation script.

## How it works

```text
Streamlit → POST /rag/ → local MiniLM query embedding → Qdrant top-5 products
                                                       ↓
                           Streamlit ← answer ← Gemini prompt with products
```

The retrieval pipeline embeds each question with `sentence-transformers/all-MiniLM-L6-v2`, retrieves up to five products from `Amazon-items-minilm-l6-v2`, and formats their IDs, descriptions, and ratings into a Gemini prompt. The prompt instructs Gemini to answer using only those products.

The UI keeps messages in the Streamlit session. Each API request sends only the latest question; previous conversation messages are not passed to Gemini. LangSmith tracing covers embedding, retrieval, prompt construction, and answer generation. API middleware attaches a unique request ID to the response body and `X-Request-ID` header.

## Requirements

- Python 3.14 or newer; `.python-version` selects 3.14.
- `uv` for dependency and workspace management.
- Docker with Compose for the application services.
- A Google API key with access to the configured Gemini models.
- A LangSmith API key for tracing and evaluation.
- Amazon Electronics data for populating the product collection.

The current application and evaluation script use Google for remote model calls. OpenAI and Groq dependencies remain in the workspace for other code and notebooks.

## Setup

Run commands from the repository root unless stated otherwise:

```bash
uv sync --all-packages --frozen
cp env.example .env
```

Fill in `GOOGLE_API_KEY` and `LANGSMITH_API_KEY` in `.env`. The example also contains OpenAI and Groq keys; those are not needed by the current RAG pipeline or evaluation script.

| Variable | Purpose | Default in code or example |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Gemini answers, evaluation scoring, and evaluation embeddings | Required for Google calls |
| `LANGSMITH_API_KEY` | LangSmith tracing and evaluation dataset access | Required for evaluation |
| `LANGSMITH_TRACING` | Enable LangSmith traces | `true` in `env.example` |
| `LANGSMITH_ENDPOINT` | LangSmith service endpoint | `https://api.smith.langchain.com` in `env.example` |
| `LANGSMITH_PROJECT` | Project for application traces | `rag-tracing` in `env.example` |
| `GEMINI_MODEL` | Answer generation model | `gemini-3.5-flash-lite` |
| `EVAL_GEMINI_MODEL` | Evaluation judge model | Uses `GEMINI_MODEL` |
| `EVAL_EMBEDDING_MODEL` | Embeddings for answer relevancy | `gemini-embedding-001` |
| `API_URL` | Streamlit backend URL | `http://api:8000` |

Choose model names available to your Google project. Keep `.env` out of version control.

## Prepare the product database

Starting Qdrant creates an empty database. Populate its collection before asking questions or running evaluations.

```bash
docker compose up -d qdrant
mkdir -p data
```

Download the Electronics metadata (`meta_Electronics.jsonl.gz`) and reviews (`Electronics.jsonl.gz`) from the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/main.html) into `data/`.

Open the notebooks with a Jupyter-capable editor and select the workspace's `.venv` kernel. Run the following notebooks in order. Notebooks 01 and 02 use `../../data/...` paths, so their kernel working directory should be `notebook/week-1/`.

| Notebook | Purpose |
| --- | --- |
| [01-explore-amazon-dataset.ipynb](notebook/week-1/01-explore-amazon-dataset.ipynb) | Filter Electronics metadata and write a 1,000-product sample and matching reviews. |
| [02-RAG-preprocessing-items.ipynb](notebook/week-1/02-RAG-preprocessing-items.ipynb) | Combine titles and features into descriptions, sample 50 products, embed them locally, and upsert them into Qdrant. |
| [03-RAG-pipeline.ipynb](notebook/week-1/03-RAG-pipeline.ipynb) | Try the MiniLM/Qdrant/Gemini pipeline directly. |
| [04-RAG-Eval-dataset.ipynb](notebook/week-1/04-RAG-Eval-dataset.ipynb) | Use Gemini to generate test questions and upload `rag-evaluation-dataset` to LangSmith. |

The collection uses 384-dimensional MiniLM vectors and cosine distance. Product payloads must contain `parent_asin`, `description`, and `average_rating`, which the API reads directly. Notebook 02 also stores image, price, and rating-count fields.

Use the same embedding model and normalization for ingestion and retrieval. The first MiniLM use downloads model weights; later calls use the local cache. The model truncates long inputs beyond 256 word pieces, and the current preparation notebook does not chunk descriptions.

Notebook 04 creates a new LangSmith dataset. If the name already exists, reuse the existing dataset or adjust the notebook before creating it again.

## Run the application

After preparing the collection:

```bash
docker compose up --build
```

| Service | Local address |
| --- | --- |
| Streamlit shopping assistant | `http://localhost:8501` |
| FastAPI documentation | `http://localhost:8000/docs` |
| Qdrant HTTP API | `http://localhost:6333` |
| Qdrant dashboard | `http://localhost:6333/dashboard` |

Qdrant persists its data in `qdrant_data/`. The API and UI source directories are mounted into their containers for development. The API starts with Uvicorn's reload mode enabled.

```bash
docker compose logs -f api streamlit-app
docker compose down
```

`make run-docker-compose` is also available; it runs `uv sync` and the legacy `docker-compose up --build` command.

### API example

```bash
curl -X POST http://localhost:8000/rag/ \
  -H 'Content-Type: application/json' \
  -d '{"query":"Which available headphones have good ratings?"}'
```

The response has this shape:

```json
{
  "request_id": "generated-uuid",
  "message": "Answer generated from the retrieved products."
}
```

The pipeline internally returns retrieved IDs, descriptions, and similarity scores as well, but the HTTP response exposes only the request ID and answer.

### Service addresses outside Docker

The API currently connects to Qdrant at `http://qdrant:6333`, a Compose service address. The notebooks and evaluation script use `http://localhost:6333` and run on the host.

To run the API directly on the host, first change its Qdrant address in [endpoints.py](apps/api/src/api/api/endpoints.py) to `http://localhost:6333`, then run:

```bash
uv run --package api --env-file .env uvicorn api.app:app --reload --port 8000
```

The Streamlit backend address is configured separately through `API_URL`. Docker Compose is the documented path for running both application services together.

## Evaluate retrieval and answers

Evaluation requires a populated local Qdrant collection, a Google API key, a LangSmith API key, and the `rag-evaluation-dataset` dataset created in notebook 04.

```bash
uv run --package api --env-file .env python apps/api/eval/eval_retriever.py
```

Each dataset example must include:

```json
{
  "inputs": {"question": "Which products match my needs?"},
  "outputs": {"reference_context_ids": ["product-parent-asin"]}
}
```

Reference IDs are product `parent_asin` values, rather than Qdrant point IDs. The script calls the RAG pipeline for each question and records four metrics:

| Metric | Calculation |
| --- | --- |
| Context precision | Matching unique product IDs divided by unique retrieved IDs. |
| Context recall | Matching unique product IDs divided by unique expected IDs. |
| Faithfulness | Ragas uses Gemini to judge whether the answer is supported by retrieved descriptions. |
| Answer relevancy | Ragas uses Gemini and Google semantic-similarity embeddings to score how well the answer addresses the question. |

Precision and recall return `0.0` when their denominator is empty. These two metrics use direct ID comparisons to support the installed Ragas 0.3.1 version.

The script prints a LangSmith experiment URL under the `retriever-...` prefix. It currently evaluates two examples concurrently (`max_concurrency=2`). Multiple judge and embedding requests per example, plus retries, can make evaluation take minutes. Adjust concurrency in the script to fit your Google project quota.

The runtime serializes local MiniLM loading and inference to avoid concurrent initialization. Answer generation retries transient failures up to six total attempts, with backoff delays capped at 30 seconds. Evaluation uses async Ragas scoring and prevents its import-time `nest_asyncio` patch for Python 3.14 compatibility.

[05-RAG-Evals.ipynb](notebook/week-1/05-RAG-Evals.ipynb) retains an older OpenAI-based pipeline and the `Amazon-items-collection-01` collection. Use the Python evaluation script for the current Google/MiniLM setup.

## Troubleshooting

- **Missing collection or empty results:** run notebook 02 against the same Qdrant instance used by the application. Starting the containers does not ingest products.
- **Gemini `503 UNAVAILABLE`:** the service is overloaded. Retries may recover; if they are exhausted, rerun later or select another model available to your project.
- **Evaluation errors:** check the evaluator trace in LangSmith. An experiment URL or progress counter does not establish that all metrics succeeded.
- **Slow first request:** MiniLM downloads and loads its weights on first use. Container model caches live under `/tmp/huggingface` and are not mounted as persistent volumes.
- **HF-token, AFC, or ignored-temperature warnings:** these messages do not by themselves mean an evaluation failed. An HF token is optional; the configured Gemini model may ignore Ragas's temperature setting.
- **UI timeout:** the Streamlit HTTP request has a 120-second timeout. Model latency and retry delays can exceed it.

## Repository layout

```text
apps/
  api/
    eval/eval_retriever.py       # LangSmith + Ragas evaluation
    src/api/
      app.py                    # FastAPI application
      api/                      # Routes, request/response models, middleware
      agents/retrieval_generation.py
      core/config.py            # Additional provider settings
  chatbot_ui/
    src/chatbot_ui/
      app.py                    # Streamlit chat UI
      core/config.py            # API_URL setting
notebook/
  prerequisites/                # Introductory LLM API notebook
  week-1/                       # Data preparation, RAG, and evaluation notebooks
src/prerequisites/              # Root workspace package
docker-compose.yml              # UI, API, and Qdrant services
env.example                     # Environment template
pyproject.toml                  # uv workspace and root dependencies
uv.lock                         # Locked dependencies
```
