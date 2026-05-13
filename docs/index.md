# Agentic Spatial Pathologist

::::{div} spatho-hero
Agentic workflows for spatial pathology.
::::

Agentic Spatial Pathologist is an evidence-generating workbench for building, running, and reviewing pathology-aware Xenium analyses. The public Python package and CLI remain named `spatho`.

The platform narrative is: stGPT learns reusable contour/region morpho-molecular representations; spatho plans, validates, and turns them into auditable spatial pathology evidence.

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

:::{grid-item-card} stGPT Upgrade
:link: STGPT_UPGRADE_PLAN
:link-type: doc

Plan the optional stGPT foundation-model runtime, region-first artifact contract, and agentic evidence loop.
:::

:::{grid-item-card} Foundation Evidence
:link: FOUNDATION_EVIDENCE_LAYER
:link-type: doc

Use stGPT region artifacts, scGPT-like RNA mapping, pathway activity, PLIP morphology, and lightweight niche fusion as auditable evidence.
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
STGPT_UPGRADE_PLAN
FOUNDATION_EVIDENCE_LAYER
XENIUM_RNA_PROTEIN_ALIGNMENT
PYPI_RELEASE
SPATHO_ROADMAP
COMMERCIALIZATION_PLAN
```
