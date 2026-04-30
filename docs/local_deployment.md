# Local Deployment

This guide describes how to deploy `spatho` locally for development, single-case analysis, and self-hosted pathology review. For the Dardel/PDC Slurm and Apptainer deployment path, see the [PDC local pathology AI guide](PDC_LOCAL_PATHOLOGY_AI.md).

## What Runs Locally

`spatho` has two layers:

- the `spatho` Python package and CLI, which prepares and runs workflow JSON files
- an optional local `pathology-ai` HTTP service, which can replace the paid OpenAI API for pathology review

The package can run with any of these review paths:

- `openai`: calls the OpenAI API through `OPENAI_API_KEY`
- `pathology_ai_api`: calls a local HTTP service at `pathology_ai_api_base_url`
- heuristic-only mode: disables AI review calls for environment checks and deterministic fallback runs

## Install the Python Package

For normal local use, install the published package:

```bash
python -m pip install -U spatho
```

For development from a source checkout:

```bash
git clone https://github.com/hutaobo/AI-Driven-Spatial-Pathologist.git
cd AI-Driven-Spatial-Pathologist
python -m pip install -U pip
python -m pip install -e .[dev]
```

If you are actively developing against a local `histoseg` checkout, install it before installing `spatho`:

```bash
python -m pip install -e ../HistoSeg
python -m pip install -e .[dev]
```

Confirm the CLI is available:

```bash
spatho list-organ-packs
spatho doctor
```

## Create a Workflow

Generate a starter workflow JSON for a Xenium case:

```bash
spatho init-workflow \
  --organ breast \
  --case-name breast_case_01 \
  --dataset-root /path/to/Xenium_outs \
  --base-pipeline-config /path/to/project/configs/breast_case_01.json \
  --output /path/to/workflows/breast_case_01.json
```

Check the environment and workflow before running:

```bash
spatho doctor --config /path/to/workflows/breast_case_01.json
```

Run the workflow:

```bash
spatho run --config /path/to/workflows/breast_case_01.json
```

## Use the OpenAI Backend

Use the OpenAI backend when you want the fastest setup with managed models:

```bash
export OPENAI_API_KEY=sk-...
```

The workflow JSON should contain:

```json
{
  "pathology_review_backend": "openai"
}
```

Then run:

```bash
spatho run --config /path/to/workflow.json
```

## Run Without AI Calls

Use heuristic-only mode for local smoke tests, deterministic fallback checks, or environments where no model service is available:

```bash
spatho run --config /path/to/workflow.json --heuristic-only
```

This path disables OpenAI calls and does not require the local `pathology-ai` stack.

## Use the Local `pathology-ai` Backend

Use the local backend when pathology review should call a self-hosted service instead of the paid OpenAI API:

```json
{
  "pathology_review_backend": "pathology_ai_api",
  "pathology_ai_api_base_url": "http://localhost:8000",
  "cluster_annotation_backend": "pathology_ai_api",
  "cluster_annotation_llm_base_url": "http://localhost:8000"
}
```

`cluster_annotation_backend="pathology_ai_api"` is optional. When enabled, Spatial Pathologist first builds the marker-based heuristic cluster labels, then asks the local LLM to review every cluster and writes a conservative consensus `cluster_celltype_annotation.csv`. OpenAI-backed annotation remains available through the existing OpenAI path.

The local stack consists of:

- `pathology-ai`: HTTP orchestration service from this repository
- `vllm`: OpenAI-compatible local LLM endpoint
- `embedder`: embedding service for `BAAI/bge-m3`
- `reranker`: reranking service for `BAAI/bge-reranker-v2-m3`
- `qdrant`: local vector database

Default service settings are stored in `deploy/pathology_ai/pathology-ai.env.example`.

## Start the Local Docker Compose Stack

Use Docker Compose on machines that support GPU containers:

```bash
cp deploy/pathology_ai/pathology-ai.env.example deploy/pathology_ai/pathology-ai.env
docker compose -f deploy/pathology_ai/docker-compose.pdc.yml up --build
```

The orchestration service listens at:

```text
http://localhost:8000
```

Verify readiness:

```bash
curl http://localhost:8000/health
```

A ready service returns `ready: true` and marks all configured components as healthy:

```json
{
  "service": "pathology-ai",
  "ready": true
}
```

After the health check passes, run `spatho` with a workflow that uses `pathology_ai_api`.

## Load Reference Documents

The local service can store reference text in Qdrant for retrieval during structure-level and case-level reviews:

```bash
curl -X POST http://localhost:8000/documents/upsert \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "who-lung-2021",
    "title": "WHO Thoracic Tumours",
    "text": "Long reference text...",
    "source": "who"
  }'
```

The service also accepts batch upserts with a top-level `documents` array.

## Troubleshooting

- If `spatho doctor` cannot import `histoseg`, install `histoseg` from PyPI or install a local checkout with `python -m pip install -e ../HistoSeg`.
- If `OPENAI_API_KEY` is missing and the workflow uses `openai`, either set the key, switch to `pathology_ai_api`, or run with `--heuristic-only`.
- If `curl /health` returns `ready: false`, inspect the component errors and the matching container logs.
- If Docker Compose is unavailable on PDC login nodes, use the Slurm/Apptainer path in the [PDC local pathology AI guide](PDC_LOCAL_PATHOLOGY_AI.md).
- If model downloads fail with authorization errors, set `HF_TOKEN` in your shell before starting the local model stack.
