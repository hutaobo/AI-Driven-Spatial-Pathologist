# Foundation Evidence Layer

Agentic Spatial Pathologist now includes an optional foundation evidence layer. It adds auditable stGPT/scGPT/SpatialFusion-inspired evidence to the existing workflow without replacing the OpenAI backend, the local `pathology-ai` backend, heuristic annotation, or the PLIP H&E contour workflow.

The v1 design is intentionally conservative: it consumes frozen or precomputed features, summarizes them at cluster and structure level, and lets the local or OpenAI reviewer adjudicate conflicts. It does not train a new SpatialFusion-style joint embedding model.

The guiding product sentence is:

> stGPT learns reusable contour/region morpho-molecular representations; spatho plans, validates, and turns them into auditable spatial pathology evidence.

## What It Borrows

From scGPT-style workflows:

- reference-mapping evidence from pretrained RNA representations;
- cluster- and structure-level label distributions;
- confidence and unknown/ambiguous rates that can temper marker-only annotation.

From SpatialFusion-like multimodal thinking:

- explicit RNA, H&E, pathway, and spatial evidence channels;
- structure/niche summaries rather than isolated cluster labels;
- cross-modal concordance, conflict, and complementarity statements.

From pathology foundation-model practice:

- H&E contour evidence is summarized as feature signals, not used as a silent replacement for molecular labels;
- visual uncertainty, artifact, tumor, inflammation, and stroma signals remain visible in the report.

## Workflow Fields

All new fields default to disabled:

```json
{
  "rna_foundation_enabled": true,
  "rna_foundation_backend": "precomputed_scgpt",
  "rna_foundation_cell_mapping_path": "/path/to/scgpt_cell_mapping.csv",
  "rna_foundation_cluster_summary_path": null,
  "pathway_activity_enabled": true,
  "pathway_activity_csv": null,
  "niche_fusion_enabled": true,
  "niche_fusion_backend": "lightweight"
}
```

The stGPT workbench fields also default to disabled:

```json
{
  "stgpt_enabled": true,
  "stgpt_backend": "precomputed_artifacts",
  "stgpt_artifact_dir": "/path/to/stgpt/spatho_export",
  "stgpt_min_cell_coverage": 0.95,
  "stgpt_require_qc_pass": true
}
```

`stgpt_backend="precomputed_artifacts"` consumes exported files without importing `stgpt`. `stgpt_backend="local_stgpt"` can call a local stGPT runtime when `stgpt_model_path` and `stgpt_config_path` are configured.

`rna_foundation_backend="precomputed_scgpt"` means the workflow reads a prepared cell-reference mapping or cluster summary. The package does not require scGPT, CUDA, Scanpy, or SpatialFusion for normal installation.

If `pathway_activity_csv` is not provided, the workflow computes a lightweight pathway score from the configured differential-expression CSV using built-in breast-relevant gene sets.

## Outputs

The layer writes a `foundation/` directory under the workflow output root:

- `rna_foundation_cluster_summary.csv/json`
- `rna_foundation_structure_summary.csv/json`
- `pathway_activity_cluster_summary.csv/json`
- `pathway_activity_structure_summary.csv/json`
- `he_morphology_feature_summary.csv/json`
- `niche_fusion_summary.csv/json`
- `foundation_evidence_metadata.json`
- `stgpt_evidence_summary.csv/json` when stGPT evidence is enabled

These files are also included in `artifact_manifest.json` when the layer is enabled.

## Evidence Bundle Contract

The long-term fusion target is an evidence graph built from small, typed evidence bundles rather than a report assembled directly from raw CSV rows. Each stGPT, RNA foundation, pathway, H&E, or niche-fusion statement should be representable as:

```json
{
  "evidence_id": "stgpt.structure.3",
  "unit": "structure",
  "unit_id": "3",
  "source": "stgpt",
  "evidence_type": "morpho_molecular_embedding",
  "measured": false,
  "model_derived": true,
  "qc_status": "warning",
  "summary": "Region embedding is available but image coverage is below the configured threshold.",
  "supporting_artifacts": [
    "region_embeddings.parquet",
    "region_qc_report.json"
  ]
}
```

Required fields are `evidence_id`, `unit`, `unit_id`, `source`, `evidence_type`, `measured`, `model_derived`, `qc_status`, `summary`, and `supporting_artifacts`. Embeddings, imputation, reconstruction, retrieval, or model scores must set `model_derived=true`; directly measured Xenium expression and H&E metadata may set `measured=true` only when the value is not inferred.

The preferred stGPT artifact set is region-first:

- `region_embeddings.parquet`
- `region_cell_membership.parquet`
- `region_molecular_summary.parquet`
- `region_image_manifest.json`
- `region_qc_report.json`
- `evidence_manifest.json`

Compatibility files such as `cell_embeddings.parquet`, `structure_embedding_summary.csv`, and `qc_report.json` should remain readable by `spatho`.

## Report Semantics

The HTML report gains a `Foundation Evidence` section. Each structure-level review can carry:

- marker heuristic evidence;
- scGPT-like RNA reference evidence;
- pathway activity evidence;
- PLIP H&E morphology evidence;
- lightweight niche-fusion consistency notes;
- stGPT morpho-molecular embedding summaries;
- final LLM adjudication.

Guardrails are explicit: missing stGPT artifacts make `spatho doctor` not ready; fatal stGPT QC blocks a run when `stgpt_require_qc_pass=true`; warning-only QC is shown as cautionary evidence; imputed or reconstructed signals must be labeled as model-derived, not measured expression. Fatal QC blocks biological claims; warning-only QC enters cautionary report language; imputation and reconstruction are never reported as measured expression.

The review text is expected to distinguish agreement from complementarity. For example, a tumor RNA reference plus tumor-like H&E morphology is concordant; a tumor RNA reference plus macrophage-rich H&E evidence may be complementary inflammation; high artifact signal is treated as a quality caveat.

## Limitations

This v1 layer is not a replacement for trained multimodal representation learning. It standardizes the evidence interface and reporting surface first. Later versions can plug in true SpatialFusion-style joint embeddings, UNI/CONCH image embeddings, or scGPT-spatial zero-shot embeddings behind the same structure-level evidence files.

For the current Atera tutorial, the generated `precomputed_scgpt` smoke mapping validates the interface only. It is not a real stGPT or scGPT-spatial result and should not be interpreted biologically. The next real integration step is to produce stGPT exports with `export_spatho_artifacts`, then let `spatho` consume those artifacts through `stgpt_backend="precomputed_artifacts"` or `stgpt_backend="local_stgpt"`.
