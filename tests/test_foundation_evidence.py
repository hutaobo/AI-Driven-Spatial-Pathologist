from __future__ import annotations

import json
from pathlib import Path

from spatho.foundation import apply_foundation_evidence
from spatho.manifest import build_artifact_manifest
from spatho.schema import WorkflowConfig, validate_workflow_config


def _touch_json(path: Path, payload: object | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({} if payload is None else payload), encoding="utf-8")


def _touch_text(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_foundation_schema_defaults_are_backward_compatible(tmp_path) -> None:
    base_config = tmp_path / "base.json"
    _touch_json(base_config, {"dataset_root": str(tmp_path)})
    workflow = tmp_path / "workflow.json"
    _touch_json(
        workflow,
        {
            "case_name": "breast_case",
            "study_context": "Breast context",
            "base_pipeline_config": str(base_config),
            "output_root": str(tmp_path / "out"),
            "annotation_taxonomy": "breast",
            "openai_enabled": False,
        },
    )

    cfg = validate_workflow_config(workflow)

    assert cfg.rna_foundation_enabled is False
    assert cfg.rna_foundation_backend == "precomputed_scgpt"
    assert cfg.pathway_activity_enabled is False
    assert cfg.niche_fusion_enabled is False
    assert cfg.niche_fusion_backend == "lightweight"


def test_apply_foundation_evidence_writes_auditable_outputs(tmp_path) -> None:
    output_root = tmp_path / "out"
    pathology_dir = output_root / "pathology_review"
    annotation_dir = output_root / "annotation"
    he_dir = output_root / "he_foundation"
    base_config = tmp_path / "base.json"
    runtime_config = output_root / "runtime_base_pipeline_config.json"
    _touch_json(base_config, {"dataset_root": str(tmp_path)})
    _touch_json(runtime_config, {"dataset_root": str(tmp_path)})

    required_annotation = {
        "cluster_evidence_json": annotation_dir / "cluster_evidence.json",
        "cluster_annotations_json": annotation_dir / "cluster_annotations.json",
        "cluster_annotations_csv": annotation_dir / "cluster_annotations.csv",
        "compatibility_csv": annotation_dir / "cluster_celltype_annotation.csv",
        "case_review_json": annotation_dir / "case_review.json",
        "report_html": annotation_dir / "index.html",
    }
    for key, path in required_annotation.items():
        if key.endswith("_csv"):
            _touch_text(path, "cluster_id,label\n0,tumor\n")
        elif key.endswith("_html"):
            _touch_text(path, "<html></html>")
        else:
            _touch_json(path, [])

    structure_reviews = [
        {
            "structure_id": 1,
            "title": "Tumor-rich structure",
            "assigned_label": "Tumor-rich structure",
            "top_clusters": [{"cluster_id": 0, "cell_count": 12}],
            "key_evidence": ["Original marker evidence."],
            "recommended_checks": [],
        }
    ]
    _touch_json(pathology_dir / "structure_reviews.json", structure_reviews)
    _touch_json(pathology_dir / "cluster_reviews.json", [])
    _touch_json(pathology_dir / "case_summary.json", {"key_findings": []})
    _touch_text(pathology_dir / "index.html", "<html><body><main><h1>Report</h1></main></body></html>")

    cell_mapping = tmp_path / "scgpt_mapping.csv"
    _touch_text(
        cell_mapping,
        "cell_id,cluster_id,predicted_label,confidence\n"
        "c1,0,tumor epithelial cell,0.91\n"
        "c2,0,tumor epithelial cell,0.87\n",
    )
    diffexp = tmp_path / "diffexp.csv"
    _touch_text(
        diffexp,
        "Feature Name,Cluster 0 Log2 fold change,Cluster 0 Adjusted p value\n"
        "EPCAM,1.4,0.001\n"
        "KRT8,1.1,0.001\n"
        "MKI67,0.6,0.001\n",
    )
    he_classification = he_dir / "he_contour_classification.csv"
    _touch_text(
        he_classification,
        "contour_id,structure_id,top_label,top_score,top_classes_json\n"
        "a,1,Invasive tumor epithelium,0.82,\"[{\"\"label\"\":\"\"Invasive tumor epithelium\"\",\"\"score\"\":0.82}]\"\n",
    )

    workflow_summary = output_root / "workflow_summary.json"
    _touch_json(
        workflow_summary,
        {
            "output_root": str(output_root),
            "annotation_taxonomy": "breast",
            "runtime_base_pipeline_config": str(runtime_config),
            "annotation_outputs": {key: str(path) for key, path in required_annotation.items()},
            "pathology_outputs": {
                "output_dir": str(pathology_dir),
                "report_html": str(pathology_dir / "index.html"),
                "cluster_reviews_json": str(pathology_dir / "cluster_reviews.json"),
                "structure_reviews_json": str(pathology_dir / "structure_reviews.json"),
                "case_summary_json": str(pathology_dir / "case_summary.json"),
            },
            "he_foundation_outputs": {
                "classification_csv": str(he_classification),
            },
        },
    )
    cfg = WorkflowConfig(
        case_name="breast_case",
        study_context="Breast context",
        base_pipeline_config=base_config,
        output_root=output_root,
        annotation_taxonomy="breast",
        openai_enabled=False,
        rna_foundation_enabled=True,
        rna_foundation_cell_mapping_path=cell_mapping,
        pathway_activity_enabled=True,
        differential_expression_csv=diffexp,
        niche_fusion_enabled=True,
    )

    result = apply_foundation_evidence(cfg, {"workflow_summary_json": str(workflow_summary)})

    foundation_dir = Path(result["foundation_dir"])
    assert (foundation_dir / "rna_foundation_cluster_summary.csv").exists()
    assert (foundation_dir / "pathway_activity_structure_summary.csv").exists()
    assert (foundation_dir / "he_morphology_feature_summary.csv").exists()
    assert (foundation_dir / "niche_fusion_summary.csv").exists()

    updated_summary = json.loads(workflow_summary.read_text(encoding="utf-8"))
    assert "foundation_outputs" in updated_summary
    updated_reviews = json.loads((pathology_dir / "structure_reviews.json").read_text(encoding="utf-8"))
    assert updated_reviews[0]["foundation_evidence"]["rna_foundation"]["top_reference_label"] == "tumor epithelial cell"
    assert "Foundation Evidence" in (pathology_dir / "index.html").read_text(encoding="utf-8")

    manifest = build_artifact_manifest(workflow_config=cfg, workflow_summary_path=workflow_summary)
    artifact_ids = {artifact["id"] for artifact in manifest["artifacts"]}
    assert "foundation.niche_fusion_summary_csv" in artifact_ids
