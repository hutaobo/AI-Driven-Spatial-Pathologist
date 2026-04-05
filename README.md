# SPatho

`spatho` is the public-facing product layer for an AI-driven spatial pathologist workflow built around Xenium-scale spatial transcriptomics.

It is designed to sit above the lower-level `histoseg` engine and expose a cleaner public experience:

- OpenAI-driven cluster annotation
- dendrogram-guided structure discovery
- H&E overlay generation
- structure-level pathology review
- HTML reporting for human-in-the-loop interpretation

This repo is where the public product experience should live.  
The underlying geometry and segmentation engine still comes from `histoseg`.

## Current Status

This is the first public product-layer scaffold.

Today it provides:

- a package name: `spatho`
- a user-facing CLI
- a starter-workflow generator
- built-in `lung` and `breast` organ packs
- a formal workflow config schema
- automatic artifact manifest generation
- a wrapper API that runs the existing `histoseg` full-auto workflow
- a roadmap for gradually migrating product logic out of `histoseg`

For the detailed developer handoff and architecture snapshot, see [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md).

## Why a Separate Repo?

`histoseg` started as a segmentation and contour-generation toolkit.

The full pathology workflow now has a different audience and a different contract:

- users care about complete case analysis, not just contour extraction
- users want reports, review priorities, and organ-specific workflows
- public documentation should focus on pathology workflows, not internal engine history

This repo creates that separation.

## Quick Start

### Install for local development

```bash
git clone https://github.com/hutaobo/AI-Driven-Spatial-Pathologist.git
cd AI-Driven-Spatial-Pathologist
pip install -U pip
pip install -e .
```

If you are actively developing against a local `histoseg` checkout, also install that editable copy first:

```bash
pip install -e ../HistoSeg
```

### Check your environment

```bash
spatho doctor --config /path/to/workflow.json
```

### List built-in organ packs

```bash
spatho list-organ-packs
```

### Generate a starter workflow

```bash
spatho init-workflow \
  --organ breast \
  --case-name breast_case_01 \
  --dataset-root /path/to/Xenium_outs \
  --base-pipeline-config /path/to/project/configs/breast_case_01.json \
  --output /path/to/workflows/breast_case_01_full_auto_openai.json
```

### Run a full workflow

```bash
spatho run --config /path/to/workflow.json
```

Disable OpenAI and force heuristic mode:

```bash
spatho run --config /path/to/workflow.json --heuristic-only
```

### Export the workflow JSON schema

```bash
spatho config-schema --output /path/to/workflow.schema.json
```

### Build or refresh an artifact manifest

```bash
spatho build-manifest --config /path/to/workflow.json
```

## Python Usage

```python
from spatho import run_workflow

result = run_workflow(r"D:\GitHub\HistoSeg\workflows\breast_s1_top_graphclust_full_auto_openai.json")
print(result["pathology_report_html"])
```

You can also generate the starter config from Python:

```python
from spatho import init_workflow

result = init_workflow(
    r"D:\GitHub\HistoSeg\workflows\breast_case_01_full_auto_openai.json",
    organ="breast",
    case_name="breast_case_01",
    dataset_root=r"Y:\long\10X_datasets\Xenium\Xenium_Breast_Cancer\Human_Breast_Biomarkers_S1_Top_outs",
    base_pipeline_config=r"D:\GitHub\sfplot\segmentation_methods\projects\breast_s1_top_graphclust\configs\breast_s1_top_graphclust.json",
)
print(result["workflow_config"])
```

## What a Workflow Produces

A typical full run produces:

- cluster evidence bundles
- OpenAI or heuristic cluster annotations
- structure assignments
- clustermap and overlay artifacts
- structure-level pathology reviews
- case-level HTML report
- a machine-readable artifact manifest

## Organ Packs

`spatho` now ships with built-in organ packs that define:

- the annotation taxonomy
- default study context
- workflow parameter defaults
- the expected artifact contract for completed runs

The first built-in packs are:

- `lung`
- `breast`

These packs live in [src/spatho/organ_packs](src/spatho/organ_packs).

## Config Contract

Workflow JSON files are now backed by a formal schema exported from the package.
This is the first step toward stable public contracts and backward-compatible workflow upgrades.

## Repository Layout

- `src/spatho`  
  Public-facing Python package and CLI

- `src/spatho/organ_packs`  
  Built-in public organ packs

- `docs/SPATHO_ROADMAP.md`  
  Productization and migration plan

- `docs/COMMERCIALIZATION_PLAN.md`  
  Academic/community vs commercial edition strategy

- `docs/PYPI_RELEASE.md`  
  Official PyPI publishing checklist for this package

- `examples/workflows`  
  Public-safe starter workflow templates for `lung` and `breast`

- `main.py`  
  Existing Gradio/Serve deployment surface kept for backward compatibility

## Relationship to HistoSeg

Current implementation model:

- `histoseg` executes the workflow
- `spatho` wraps and presents it as a product

Target implementation model:

- `histoseg` becomes the geometry/segmentation engine
- `spatho` owns workflow UX, organ packs, public docs, reports, and deployment surfaces

## Public Release Plan

The next milestones are:

1. expand organ packs beyond `lung` and `breast`
2. add richer tests and CI for CLI + workflow smoke checks
3. stabilize config schema and artifact manifest versions
4. migrate public-safe workflow logic from `histoseg` into `spatho`
5. add example reports and example datasets

See [docs/SPATHO_ROADMAP.md](docs/SPATHO_ROADMAP.md) and [docs/COMMERCIALIZATION_PLAN.md](docs/COMMERCIALIZATION_PLAN.md).

## Publishing

This repo now includes a PyPI publishing workflow based on GitHub Actions Trusted Publishing.
See [docs/PYPI_RELEASE.md](docs/PYPI_RELEASE.md) for the exact setup and release steps.

## Existing Serve App

This repo also contains an older Gradio deployment layer in [main.py](main.py).

That app should now be treated as one deployment surface, not the core product definition.
The package and CLI in `src/spatho` are the preferred direction for public-tool development.

## License

This project is intended for noncommercial research use unless separately licensed.
Before public release, the license text and commercial boundary should be reviewed together with the underlying `histoseg` dependency.
