# Atera WTA Breast Cancer on PDC

This tutorial records a reproducible PDC run of AI-Driven Spatial Pathologist on the 10x Xenium Atera WTA Preview FFPE Breast Cancer dataset, using the local `pathology-ai` backend and pyXenium core workflows.

## Dataset

- PDC dataset: `/cfs/klemming/projects/supr/naiss2025-22-606/data/WTA_Preview_FFPE_Breast_Cancer_outs`
- Local Windows mirror: `Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs`
- PDC output root: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429`

The dataset copy contains `cell_feature_matrix.h5`, cell and nucleus boundary parquet files, `cells.parquet`, `experiment.xenium`, `metrics_summary.csv`, `WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv`, and a registered H&E image pyramid under `spatialdata.zarr/images/he`.

## Service Check

The workflow uses the PDC local `pathology-ai` API as the review backend:

```bash
curl http://nid002802:8000/health
```

This run used Slurm job `20140027` on node `nid002802`; the captured health payload was:

```{literalinclude} _static/tutorials/atera_wta_breast_pdc/pathology_ai_health.json
:language: json
```

## Submit the PDC Job

From the PDC login node:

```bash
cd /cfs/klemming/home/h/hutaobo/AI-Driven-Spatial-Pathologist
git fetch origin
git checkout main
git pull --ff-only origin main

sbatch \
  --export=ALL,PATHOLOGY_AI_BASE_URL=http://nid002802:8000 \
  deploy/pathology_ai/atera_wta_breast_pdc.sbatch
```

The Slurm wrapper creates a venv under the output root, installs `spatho`, `pyXenium`, `histoseg`, and scientific dependencies, clones `sfplot` for `tissue_structure_pipeline`, then runs:

```bash
python scripts/pdc_atera_breast_workflow.py \
  --dataset-root /cfs/klemming/projects/supr/naiss2025-22-606/data/WTA_Preview_FFPE_Breast_Cancer_outs \
  --run-root /cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429 \
  --sfplot-root /cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/deps/sfplot \
  --pathology-ai-base-url http://nid002802:8000
```

## Generated Inputs

The source dataset already has graph-cluster assignments, but it does not include the standard 10x differential-expression and UMAP projection CSVs expected by the full-auto `spatho` workflow. The PDC driver generates:

- `inputs/analysis/clustering/gene_expression_graphclust/clusters.csv`
- `inputs/analysis/diffexp/gene_expression_graphclust/differential_expression.csv`
- `inputs/analysis/umap/gene_expression_2_components/projection.csv`
- `inputs/he/atera_wta_breast_pdc_20260429_registered_he_level6.tif`
- `inputs/he/atera_wta_breast_pdc_20260429_he_alignment_level6.csv`

The differential-expression table is a cluster-pseudobulk log2 fold-change approximation from `cell_feature_matrix.h5`; the projection is derived from cell centroids. The H&E tutorial asset is extracted from `spatialdata.zarr/images/he` level 6 and uses the stored affine transform from image pixel coordinates to Xenium pixel coordinates.

```{literalinclude} _static/tutorials/atera_wta_breast_pdc/generated_inputs_metadata.json
:language: json
```

## AI-Driven Spatial Pathologist Run

The workflow config uses the local backend and keeps OpenAI disabled for this run:

```json
{
  "annotation_taxonomy": "breast",
  "pathology_review_backend": "pathology_ai_api",
  "pathology_ai_api_base_url": "http://nid002802:8000",
  "openai_enabled": false
}
```

The readiness check is:

```bash
spatho doctor --config workflows/atera_wta_breast_pdc_20260429_pathology_ai.json
```

The captured doctor output and workflow summary are included here:

```{literalinclude} _static/tutorials/atera_wta_breast_pdc/spatho_doctor.json
:language: json
```

```{literalinclude} _static/tutorials/atera_wta_breast_pdc/spatho/workflow_summary.json
:language: json
```

Selected overlays:

![H&E structure overlay](_static/tutorials/atera_wta_breast_pdc/spatho/he_structure_isoline_overlay.png)

![Spatial structure overlay](_static/tutorials/atera_wta_breast_pdc/spatho/spatial_structure_isoline_overlay.png)

## pyXenium Core Results

The same job runs the pyXenium Atera WTA breast LR/pathway topology smoke workflow and a mechanostress snapshot. Full GMI controls are intentionally skipped for this tutorial pass.

Topology report:

```{literalinclude} _static/tutorials/atera_wta_breast_pdc/pyxenium_topology/report.md
:language: markdown
```

Mechanostress report:

```{literalinclude} _static/tutorials/atera_wta_breast_pdc/pyxenium_mechanostress/report.md
:language: markdown
```

Machine-readable summaries:

```{literalinclude} _static/tutorials/atera_wta_breast_pdc/pyxenium_summary.json
:language: json
```

## Artifact Locations

Lightweight tutorial assets are committed under:

`docs/_static/tutorials/atera_wta_breast_pdc/`

Large outputs remain on PDC under:

`/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429`

```{literalinclude} _static/tutorials/atera_wta_breast_pdc/artifact_index.json
:language: json
```
