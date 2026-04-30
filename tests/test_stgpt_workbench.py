from __future__ import annotations

import json
from pathlib import Path

from spatho.api import workflow_doctor_report
from spatho.manifest import build_artifact_manifest
from spatho.reports import build_evidence_report_section
from spatho.schema import WorkflowConfig, validate_workflow_config
from spatho.stgpt import apply_stgpt_evidence, inspect_stgpt_evidence
from spatho.workbench import run_evidence_workbench


def _touch_json(path: Path, payload: object | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({} if payload is None else payload), encoding="utf-8")


def _touch_text(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _workflow_config(tmp_path: Path, **extra: object) -> Path:
    base_config = tmp_path / "base.json"
    _touch_json(base_config, {"dataset_root": str(tmp_path)})
    workflow = tmp_path / "workflow.json"
    payload = {
        "case_name": "stgpt_case",
        "study_context": "Synthetic stGPT evidence case",
        "base_pipeline_config": str(base_config),
        "output_root": str(tmp_path / "out"),
        "annotation_taxonomy": "breast",
        "openai_enabled": False,
        **extra,
    }
    _touch_json(workflow, payload)
    return workflow


def _required_workflow_summary(tmp_path: Path) -> tuple[WorkflowConfig, Path]:
    output_root = tmp_path / "out"
    annotation_dir = output_root / "annotation"
    pathology_dir = output_root / "pathology_review"
    runtime_config = output_root / "runtime_base_pipeline_config.json"
    _touch_json(runtime_config, {"dataset_root": str(tmp_path)})
    annotation_outputs = {
        "cluster_evidence_json": annotation_dir / "cluster_evidence.json",
        "cluster_annotations_json": annotation_dir / "cluster_annotations.json",
        "cluster_annotations_csv": annotation_dir / "cluster_annotations.csv",
        "compatibility_csv": annotation_dir / "cluster_celltype_annotation.csv",
        "case_review_json": annotation_dir / "case_review.json",
        "report_html": annotation_dir / "index.html",
    }
    for path in annotation_outputs.values():
        if path.suffix == ".csv":
            _touch_text(path, "cluster_id,label\n0,tumor\n")
        elif path.suffix == ".html":
            _touch_text(path, "<html><body><main><h1>Annotation</h1></main></body></html>")
        else:
            _touch_json(path, [])
    pathology_outputs = {
        "report_html": pathology_dir / "index.html",
        "cluster_reviews_json": pathology_dir / "cluster_reviews.json",
        "structure_reviews_json": pathology_dir / "structure_reviews.json",
        "case_summary_json": pathology_dir / "case_summary.json",
    }
    _touch_text(pathology_outputs["report_html"], "<html><body><main><h1>Review</h1></main></body></html>")
    _touch_json(pathology_outputs["cluster_reviews_json"], [])
    _touch_json(pathology_outputs["structure_reviews_json"], [])
    _touch_json(pathology_outputs["case_summary_json"], {"key_findings": []})

    artifact_dir = tmp_path / "stgpt_artifacts"
    _touch_text(artifact_dir / "cell_embeddings.parquet", "placeholder parquet bytes")
    _touch_text(artifact_dir / "structure_embedding_summary.csv", "structure_label,n_cells,emb_0\n1,5,0.1\n")
    _touch_json(artifact_dir / "qc_report.json", {"status": "pass", "warnings": ["registration warning"], "image_coverage": 0.90})

    workflow = _workflow_config(
        tmp_path,
        stgpt_enabled=True,
        stgpt_artifact_dir=str(artifact_dir),
        stgpt_min_cell_coverage=0.95,
    )
    cfg = validate_workflow_config(workflow)
    summary_path = output_root / "workflow_summary.json"
    _touch_json(
        summary_path,
        {
            "output_root": str(output_root),
            "annotation_taxonomy": "breast",
            "runtime_base_pipeline_config": str(runtime_config),
            "annotation_outputs": {key: str(path) for key, path in annotation_outputs.items()},
            "pathology_outputs": {key: str(path) for key, path in pathology_outputs.items()},
        },
    )
    return cfg, summary_path


def test_stgpt_schema_defaults_and_path_resolution(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "relative_artifacts"
    workflow = _workflow_config(tmp_path, stgpt_enabled=True, stgpt_artifact_dir="relative_artifacts")
    cfg = validate_workflow_config(workflow)
    assert cfg.stgpt_enabled is True
    assert cfg.stgpt_backend == "precomputed_artifacts"
    assert cfg.stgpt_artifact_dir == artifact_dir.resolve()
    assert cfg.stgpt_require_qc_pass is True


def test_doctor_reports_missing_stgpt_artifacts(tmp_path: Path) -> None:
    workflow = _workflow_config(tmp_path, stgpt_enabled=True, stgpt_artifact_dir=str(tmp_path / "missing"))
    report = workflow_doctor_report(workflow)
    assert report["ready_to_run"] is False
    assert any("stGPT evidence is enabled" in issue for issue in report["issues"])


def test_doctor_allows_warning_only_stgpt_qc(tmp_path: Path) -> None:
    cfg, _ = _required_workflow_summary(tmp_path)
    report = workflow_doctor_report(tmp_path / "workflow.json")
    assert report["ready_to_run"] is True
    assert report["stgpt_evidence"]["warnings"]
    assert cfg.stgpt_enabled is True


def test_apply_stgpt_evidence_writes_auditable_outputs(tmp_path: Path) -> None:
    cfg, summary_path = _required_workflow_summary(tmp_path)
    result = apply_stgpt_evidence(cfg, {"workflow_summary_json": str(summary_path)})
    assert Path(result["stgpt_evidence_summary_csv"]).exists()
    updated_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    outputs = updated_summary["foundation_outputs"]
    assert "stgpt_cell_embeddings_parquet" in outputs
    assert "stgpt_evidence_summary_csv" in outputs
    report_html = (cfg.output_root / "pathology_review" / "index.html").read_text(encoding="utf-8")
    assert "stGPT Evidence" in report_html
    assert "Cautionary evidence" in report_html
    assert "model-derived" in report_html
    manifest = build_artifact_manifest(workflow_config=cfg, workflow_summary_path=summary_path)
    artifact_ids = {artifact["id"] for artifact in manifest["artifacts"]}
    assert "foundation.stgpt_evidence_summary_csv" in artifact_ids


def test_stgpt_fatal_qc_blocks_when_required(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "stgpt_artifacts"
    _touch_text(artifact_dir / "cell_embeddings.parquet", "placeholder")
    _touch_text(artifact_dir / "structure_embedding_summary.csv", "structure_label,n_cells\n1,2\n")
    _touch_json(artifact_dir / "qc_report.json", {"status": "fail", "fatal_errors": ["bad registration"]})
    workflow = _workflow_config(tmp_path, stgpt_enabled=True, stgpt_artifact_dir=str(artifact_dir))
    cfg = validate_workflow_config(workflow)
    inspection = inspect_stgpt_evidence(cfg)
    assert inspection["ready"] is False
    assert any("fatal" in error.lower() for error in inspection["errors"])


def test_public_workbench_and_report_imports() -> None:
    assert run_evidence_workbench is not None
    assert "stGPT Evidence" in build_evidence_report_section(summary_rows=[])
