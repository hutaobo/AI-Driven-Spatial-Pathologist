# AI-Driven Spatial Pathologist

::::{div} spatho-hero
AI-driven spatial pathology workflows for Xenium-scale spatial transcriptomics.
::::

AI-Driven Spatial Pathologist is a workflow layer for building, running, and reviewing pathology-aware Xenium analyses. The public Python package and CLI remain named `spatho`.

The documentation focuses on practical deployment paths first, from local workstations to PDC/HPC environments with a self-hosted pathology AI backend.

::::{grid} 1 1 2 3
:gutter: 2
:class-container: spatho-card-grid

:::{grid-item-card} Tutorial
:link: ATERA_WTA_BREAST_PDC_TUTORIAL
:link-type: doc

Run the ATERA WTA breast PDC walkthrough and inspect the generated workflow artifacts.
:::

:::{grid-item-card} Local Deployment
:link: local_deployment
:link-type: doc

Install and run `spatho` on a local workstation for development or lightweight analysis.
:::

:::{grid-item-card} PDC/HPC AI Backend
:link: PDC_LOCAL_PATHOLOGY_AI
:link-type: doc

Deploy the local pathology AI service with vLLM, embeddings, reranking, and Qdrant.
:::

:::{grid-item-card} Development Guide
:link: DEVELOPMENT_GUIDE
:link-type: doc

Understand the repository layout, development workflow, and compatibility layer.
:::

:::{grid-item-card} Release & Roadmap
:link: SPATHO_ROADMAP
:link-type: doc

Review the roadmap and package release notes for the `spatho` distribution.
:::

:::{grid-item-card} Project Notes
:link: XENIUM_RNA_PROTEIN_ALIGNMENT
:link-type: doc

Read design notes for Xenium RNA/protein alignment, commercialization, and packaging.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2
:caption: Tutorials

ATERA_WTA_BREAST_PDC_TUTORIAL
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Deployment

local_deployment
PDC_LOCAL_PATHOLOGY_AI
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Project Notes

DEVELOPMENT_GUIDE
XENIUM_RNA_PROTEIN_ALIGNMENT
PYPI_RELEASE
SPATHO_ROADMAP
COMMERCIALIZATION_PLAN
```
