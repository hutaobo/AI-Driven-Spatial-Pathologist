# SPatho

[![PyPI version](https://img.shields.io/pypi/v/spatho.svg)](https://pypi.org/project/spatho/)
[![Python versions](https://img.shields.io/pypi/pyversions/spatho.svg)](https://pypi.org/project/spatho/)
[![PyPI downloads](https://img.shields.io/pypi/dm/spatho.svg)](https://pypi.org/project/spatho/)
[![License](https://img.shields.io/pypi/l/spatho.svg)](LICENSE)
[![Python Package](https://github.com/hutaobo/AI-Driven-Spatial-Pathologist/actions/workflows/python-package.yml/badge.svg)](https://github.com/hutaobo/AI-Driven-Spatial-Pathologist/actions/workflows/python-package.yml)
[![Publish to PyPI](https://github.com/hutaobo/AI-Driven-Spatial-Pathologist/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/hutaobo/AI-Driven-Spatial-Pathologist/actions/workflows/publish-pypi.yml)
[![Docker image](https://github.com/hutaobo/AI-Driven-Spatial-Pathologist/actions/workflows/docker-image-ghcr.yml/badge.svg)](https://github.com/hutaobo/AI-Driven-Spatial-Pathologist/actions/workflows/docker-image-ghcr.yml)
[![GitHub release](https://img.shields.io/github/v/release/hutaobo/AI-Driven-Spatial-Pathologist?include_prereleases)](https://github.com/hutaobo/AI-Driven-Spatial-Pathologist/releases)
[![Last commit](https://img.shields.io/github/last-commit/hutaobo/AI-Driven-Spatial-Pathologist.svg)](https://github.com/hutaobo/AI-Driven-Spatial-Pathologist/commits/main)
[![Issues](https://img.shields.io/github/issues/hutaobo/AI-Driven-Spatial-Pathologist.svg)](https://github.com/hutaobo/AI-Driven-Spatial-Pathologist/issues)

`spatho` is a Python package and CLI for AI-driven spatial pathology workflows around Xenium-scale spatial transcriptomics. It wraps the lower-level `histoseg` engine with workflow configuration, organ packs, artifact manifests, H&E overlays, structure review, and report generation.

> Legacy standalone surface: the canonical product-layer implementation is being integrated into
> [`ASTRO`](https://github.com/hutaobo/ASTRO) under
> `app/src/xenium_ai_discovery/pathology_app/`.
> This repository remains the compatibility, packaging, and deployment-oriented shell for `spatho`.

## Two Parallel AI Backends

`spatho` now supports two parallel AI review paths. The paid OpenAI API route remains available and unchanged, while the local `pathology-ai` route adds a self-hosted option for private or cluster deployments.

| Backend | How it runs | Best for | Key settings |
| --- | --- | --- | --- |
| `openai` | Calls the paid OpenAI API with your `OPENAI_API_KEY`. | Fast setup, managed models, lightweight local machine. | `pathology_review_backend="openai"` |
| `pathology_ai_api` | Calls a local HTTP service backed by vLLM, embeddings, reranking, and Qdrant. | PDC/HPC, private data, cost control, local model operations. | `pathology_review_backend="pathology_ai_api"` and `pathology_ai_api_base_url` |

The two paths are intentionally independent: enabling the local service does not remove or disable the OpenAI backend.

## Quick Start

### Install from PyPI

```bash
python -m pip install -U spatho
```

For local development:

```bash
git clone https://github.com/hutaobo/AI-Driven-Spatial-Pathologist.git
cd AI-Driven-Spatial-Pathologist
python -m pip install -U pip
python -m pip install -e .[dev]
```

If you are actively developing against a local `histoseg` checkout, install that editable copy first:

```bash
python -m pip install -e ../HistoSeg
```

### Path A: paid OpenAI API

Use this path when you want the simplest managed-model setup.

```bash
export OPENAI_API_KEY=sk-...
spatho init-workflow \
  --organ breast \
  --case-name breast_case_01 \
  --dataset-root /path/to/Xenium_outs \
  --base-pipeline-config /path/to/project/configs/breast_case_01.json \
  --output /path/to/workflows/breast_case_01_openai.json

spatho run --config /path/to/workflows/breast_case_01_openai.json
```

The generated workflow can keep:

```json
{
  "pathology_review_backend": "openai"
}
```

Disable OpenAI and force heuristic mode when needed:

```bash
spatho run --config /path/to/workflow.json --heuristic-only
```

### Path B: local pathology-ai service

Use this path when you want pathology review to call a self-hosted service instead of the paid OpenAI API.

```json
{
  "pathology_review_backend": "pathology_ai_api",
  "pathology_ai_api_base_url": "http://localhost:8000"
}
```

For PDC/Dardel deployment, see [docs/PDC_LOCAL_PATHOLOGY_AI.md](docs/PDC_LOCAL_PATHOLOGY_AI.md). The local stack is:

- `pathology-ai`: lightweight HTTP orchestration from this repo
- `vllm`: OpenAI-compatible local LLM endpoint
- `embedder`: TEI-compatible Python embedding service for `BAAI/bge-m3`
- `reranker`: TEI-compatible Python reranking service for `BAAI/bge-reranker-v2-m3`
- `qdrant`: local vector storage

Default local model configuration:

```text
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_MODEL=openai/gpt-oss-120b
EMBED_MODEL=BAAI/bge-m3
RERANK_MODEL=BAAI/bge-reranker-v2-m3
VECTOR_DB=qdrant
DEFAULT_TOP_K=6
STRICT_JSON=true
```

## Common CLI Tasks

Check an environment and workflow config:

```bash
spatho doctor --config /path/to/workflow.json
```

List built-in organ packs:

```bash
spatho list-organ-packs
```

Export the workflow JSON schema:

```bash
spatho config-schema --output /path/to/workflow.schema.json
```

Build or refresh an artifact manifest:

```bash
spatho build-manifest --config /path/to/workflow.json
```

Write Xenium RNA+protein + H&E alignment fixtures:

```bash
spatho write-xenium-alignment-fixtures \
  --output-dir /path/to/output/pipeline/validation \
  --segmentation-source ranger_protein_assisted
```

This writes a Xenium RNA+protein alignment note, a fixture manifest, and transform cases covering identity, `um -> pixel`, translation, axis order, and composed polygon export.

## Python Usage

```python
from spatho import run_workflow

result = run_workflow("/path/to/workflows/breast_case_01_openai.json")
print(result["pathology_report_html"])
```

Generate a starter config from Python:

```python
from spatho import init_workflow

result = init_workflow(
    "/path/to/workflows/breast_case_01_openai.json",
    organ="breast",
    case_name="breast_case_01",
    dataset_root="/path/to/Xenium_outs",
    base_pipeline_config="/path/to/project/configs/breast_case_01.json",
)
print(result["workflow_config"])
```

## What a Workflow Produces

A typical full run produces:

- cluster evidence bundles
- OpenAI, local pathology-ai, or heuristic cluster annotations
- dendrogram-guided structure assignments
- clustermap and H&E overlay artifacts
- structure-level pathology reviews
- case-level HTML report
- machine-readable artifact manifest

## Organ Packs

`spatho` ships with built-in organ packs that define the annotation taxonomy, default study context, workflow parameter defaults, and expected artifact contract.

Built-in packs:

- `lung`
- `breast`

These packs live in [src/spatho/organ_packs](src/spatho/organ_packs).

## Config Contract

Workflow JSON files are backed by a formal schema exported from the package. For Xenium RNA+protein workflows, the config template records:

- `dataset_modality = xenium_rna_protein`
- `canonical_space = physical_um`
- `export_space = xenium_explorer_pixel`
- `xenium_pixel_size_um`
- `segmentation_source`

See [docs/XENIUM_RNA_PROTEIN_ALIGNMENT.md](docs/XENIUM_RNA_PROTEIN_ALIGNMENT.md) for the rationale and polygon-level analysis model.

## Repository Layout

- `src/spatho`: public-facing Python package and CLI
- `src/pathology_ai_service`: local pathology AI HTTP service
- `deploy/pathology_ai`: Docker Compose and PDC Slurm/Apptainer deployment assets
- `docs/PDC_LOCAL_PATHOLOGY_AI.md`: local/PDC pathology-ai deployment guide
- `docs/PYPI_RELEASE.md`: PyPI publishing checklist
- `examples/workflows`: public-safe starter workflow templates
- `main.py`: older Gradio/Serve deployment surface kept for compatibility

## Relationship to HistoSeg and ASTRO

Current implementation model:

- `histoseg` executes the geometry, segmentation, and workflow internals
- `spatho` wraps and presents the workflow as a product-facing package

Target implementation model:

- `histoseg` remains the geometry/segmentation engine
- `spatho` owns workflow UX, organ packs, public docs, reports, and deployment surfaces
- the canonical integrated product implementation continues to move into [ASTRO](https://github.com/hutaobo/ASTRO)

## Publishing

This repo includes a PyPI publishing workflow based on GitHub Actions Trusted Publishing. See [docs/PYPI_RELEASE.md](docs/PYPI_RELEASE.md) for setup and release steps.

## License

This project is intended for noncommercial research use unless separately licensed. Before public release or commercial use, review the license text and commercial boundary together with the underlying `histoseg` dependency.
