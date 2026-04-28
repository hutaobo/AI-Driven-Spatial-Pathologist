# PDC Local Pathology AI Service

This document adds a parallel local deployment path for `pathology_review_backend = "pathology_ai_api"` while preserving the existing `openai` workflow option.

## What stays the same

- `spatho` workflow JSON does not gain new public fields.
- `pathology_review_backend = "openai"` remains valid.
- `pathology_review_backend = "pathology_ai_api"` still points to an HTTP service at `pathology_ai_api_base_url`.

## New local stack

The new PDC-oriented stack consists of:

- `pathology-ai`: a lightweight HTTP orchestration layer in this repo
- `vllm`: an OpenAI-compatible local LLM endpoint
- `embedder`: a Hugging Face TEI deployment for `BAAI/bge-m3`
- `reranker`: a Hugging Face TEI deployment for `BAAI/bge-reranker-v2-m3`
- `qdrant`: local vector storage for chunk retrieval

The default `.env` values are:

- `LLM_BASE_URL=http://vllm:8000/v1`
- `LLM_MODEL=openai/gpt-oss-120b`
- `EMBED_MODEL=BAAI/bge-m3`
- `RERANK_MODEL=BAAI/bge-reranker-v2-m3`
- `VECTOR_DB=qdrant`
- `DEFAULT_TOP_K=6`
- `STRICT_JSON=true`

## Endpoints

The service intentionally keeps the contract simple:

- `GET /health`
- `POST /documents/upsert`
- `POST /review`
- `POST /reviews/structure`
- `POST /reviews/case`

Compatibility aliases are also available under `/v1/...`.

### `POST /documents/upsert`

Single-document form:

```json
{
  "document_id": "who-lung-2021",
  "title": "WHO Thoracic Tumours",
  "text": "Long reference text...",
  "source": "who",
  "metadata": {
    "edition": "2021"
  }
}
```

Batch form:

```json
{
  "documents": [
    {
      "document_id": "who-lung-2021",
      "title": "WHO Thoracic Tumours",
      "text": "Long reference text..."
    }
  ]
}
```

### `POST /reviews/structure`

```json
{
  "question": "What pathology interpretation best matches this structure?",
  "document_ids": ["who-lung-2021"],
  "answer_language": "en",
  "top_k": 6,
  "entity_name": "Tumor-rich structure 4",
  "evidence": {
    "markers": ["EPCAM", "KRT19", "MUC1"],
    "notes": "Polygon-linked H&E region shows gland-forming epithelium."
  }
}
```

### `POST /reviews/case`

The request body is the same shape as `structure`, but the question and evidence represent whole-case context.

## Running locally

From the repo root:

```bash
cp deploy/pathology_ai/pathology-ai.env.example deploy/pathology_ai/pathology-ai.env
docker compose -f deploy/pathology_ai/docker-compose.pdc.yml up --build
```

The `pathology-ai` service will be available at `http://localhost:8000`.

## Swapping the local LLM later

If you want to keep the same architecture but stop using `gpt-oss`, only change the `vllm` command and `LLM_MODEL`.

Example alternative:

- `Qwen/Qwen3.6-35B-A3B-FP8`

The `pathology-ai` interface and `spatho` workflow contract stay unchanged.
