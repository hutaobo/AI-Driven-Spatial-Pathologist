# PDC Local Pathology AI Service

This document adds a parallel local deployment path for
`pathology_review_backend = "pathology_ai_api"` while preserving the existing
`openai` workflow option.

## What stays the same

- `pathology_review_backend = "openai"` remains valid and unchanged.
- `pathology_review_backend = "pathology_ai_api"` still points to an HTTP
  service at `pathology_ai_api_base_url`.
- The public `spatho` workflow JSON schema does not need PDC-specific fields.

## New local stack

The PDC-oriented stack consists of:

- `pathology-ai`: the lightweight HTTP orchestration layer in this repo
- `vllm`: an OpenAI-compatible local LLM endpoint
- `embedder`: a TEI-compatible Python service for `BAAI/bge-m3`
- `reranker`: a TEI-compatible Python service for `BAAI/bge-reranker-v2-m3`
- `qdrant`: local vector storage for chunk retrieval

Default values:

- `LLM_MODEL=openai/gpt-oss-120b`
- `EMBED_MODEL=BAAI/bge-m3`
- `RERANK_MODEL=BAAI/bge-reranker-v2-m3`
- `VECTOR_DB=qdrant`
- `DEFAULT_TOP_K=6`
- `STRICT_JSON=true`

## PDC Slurm/Apptainer deployment

Use this path on Dardel GPU nodes. PDC login nodes do not provide Docker
Compose, and the Hugging Face TEI `cpu-1.9` image is amd64-only. The PDC path
therefore uses Slurm plus Apptainer sandboxes and replaces TEI with small Python
HTTP services that expose the same `/embed`, `/rerank`, and `/health` endpoints
used by `pathology-ai`.

Defaults:

- Current Dardel Slurm account: `naiss2026-4-680-gpu`
- Current Dardel Slurm partition: `gpu`
- Runtime root: `/cfs/klemming/projects/supr/naiss2023-23-563/pathology-ai`
- vLLM GPUs: `CUDA_VISIBLE_DEVICES=0,1`
- embedder GPU: `CUDA_VISIBLE_DEVICES=2`
- reranker GPU: `CUDA_VISIBLE_DEVICES=3`

The prepare script auto-detects the runtime image family:

- `x86_64` Dardel `gpu` nodes: ROCm, `vllm/vllm-openai-rocm:latest`, Apptainer `--rocm`
- `aarch64` GraceHopper nodes: CUDA, `vllm/vllm-openai:latest`, Apptainer `--nv`

Prepare the environment file from the repo root:

```bash
cp deploy/pathology_ai/pathology-ai.gpugh.env.example deploy/pathology_ai/pathology-ai.gpugh.env
```

If a Hugging Face token is needed for model downloads, add it outside git, for
example in your shell before submitting:

```bash
export HF_TOKEN=...
```

Build the Apptainer sandboxes. On current Dardel `gpu`, run this on the normal
login node so it builds x86_64 ROCm sandboxes:

```bash
ssh dardel.pdc.kth.se
cd /cfs/klemming/home/h/hutaobo/AI-Driven-Spatial-Pathologist
bash deploy/pathology_ai/pdc_prepare_gh200.sh
```

If a `gpugh` partition is available and you want the GH200/CUDA path instead,
run the same command from `ssh logingh`.

The prepare script creates:

```text
/cfs/klemming/projects/supr/naiss2023-23-563/pathology-ai/images/vllm-openai-rocm-latest
/cfs/klemming/projects/supr/naiss2023-23-563/pathology-ai/images/qdrant-latest
/cfs/klemming/projects/supr/naiss2023-23-563/pathology-ai/runtime.env
```

Submit the service job:

```bash
sbatch deploy/pathology_ai/pathology-ai.gpugh.sbatch
```

Check the allocated node and logs:

```bash
squeue -u "$USER" -n pathology-ai
tail -f /cfs/klemming/projects/supr/naiss2023-23-563/pathology-ai/logs/<job-id>/pathology-ai.log
```

Verify health from PDC:

```bash
curl http://<allocated-node>:8000/health
```

Successful readiness means the response has:

```json
{
  "service": "pathology-ai",
  "ready": true
}
```

and all four components under `components` have `"ok": true`.

## Docker Compose deployment

Use this path only on machines that support Docker Compose and GPU containers.
It is kept for non-PDC local hosts and does not replace the PDC GH200 path.

From the repo root:

```bash
cp deploy/pathology_ai/pathology-ai.env.example deploy/pathology_ai/pathology-ai.env
docker compose -f deploy/pathology_ai/docker-compose.pdc.yml up --build
```

The `pathology-ai` service will be available at:

```text
http://localhost:8000
```

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

The request body is the same shape as `structure`, but the question and
evidence represent whole-case context.

## Troubleshooting

- If `docker` or `docker compose` is missing on PDC, use the GH200 Slurm path.
- If `sbatch --test-only` fails with an invalid partition, inspect `sinfo -s`
  and override with `sbatch -A <account> -p <partition> ...`.
- If `curl /health` returns `ready=false`, inspect the component errors and the
  matching log file in `$PDC_PATHOLOGY_AI_ROOT/logs/<job-id>/`.
- If model downloads fail with an authorization error, set `HF_TOKEN` before
  running the prepare or Slurm job.
- If the runtime storage fills up, set `PDC_PATHOLOGY_AI_ROOT` to another
  project path before running both prepare and `sbatch`.

## Swapping the local LLM later

If you want to keep the same architecture but stop using `gpt-oss`, change
`LLM_MODEL` and the vLLM model argument in the environment file or Slurm job.
The `pathology-ai` interface and `spatho` workflow contract stay unchanged.
