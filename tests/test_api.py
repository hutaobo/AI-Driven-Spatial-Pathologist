from __future__ import annotations

import json
from pathlib import Path

from spatho.api import build_manifest, init_workflow, list_available_organ_packs, workflow_doctor_report, write_schema
from spatho.manifest import build_artifact_manifest
from spatho.schema import validate_workflow_config


def test_workflow_doctor_report_flags_missing_inputs(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_path = tmp_path / "workflow.json"
    payload = {
        "case_name": "demo_case",
        "study_context": "Demo context",
        "base_pipeline_config": str(tmp_path / "missing_base.json"),
        "output_root": str(tmp_path / "outputs"),
        "annotation_taxonomy": "lung",
        "differential_expression_csv": str(tmp_path / "missing_diffexp.csv"),
        "projection_csv": str(tmp_path / "missing_projection.csv"),
        "openai_enabled": True,
        "openai_api_key_env": "OPENAI_API_KEY",
        "openai_model": "gpt-5.4",
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    report = workflow_doctor_report(config_path)

    assert report["config_exists"] is True
    assert report["base_pipeline_config_exists"] is False
    assert report["differential_expression_csv_exists"] is False
    assert report["projection_csv_exists"] is False
    assert report["openai_api_key_present"] is False
    assert report["ready_to_run"] is False
    assert len(report["issues"]) >= 3
    assert report["schema_valid"] is True


def test_workflow_doctor_report_handles_schema_errors(tmp_path) -> None:
    config_path = tmp_path / "workflow.json"
    payload = {
        "case_name": "demo_case",
        "study_context": "Demo context",
        "base_pipeline_config": str(tmp_path / "base.json"),
        "output_root": str(tmp_path / "outputs"),
        "annotation_taxonomy": "pancreas"
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    report = workflow_doctor_report(config_path)

    assert report["config_exists"] is True
    assert report["schema_valid"] is False
    assert report["ready_to_run"] is False
    assert any("validation failed" in issue.lower() for issue in report["issues"])


def test_init_workflow_writes_expected_template(tmp_path) -> None:
    dataset_root = tmp_path / "outs"
    base_cfg = tmp_path / "project" / "configs" / "case.json"
    base_cfg.parent.mkdir(parents=True)
    base_cfg.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "workflow.json"

    result = init_workflow(
        output_path,
        organ="breast",
        case_name="breast_demo",
        dataset_root=dataset_root,
        base_pipeline_config=base_cfg,
    )

    assert result["workflow_config"] == str(output_path.resolve())
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["annotation_taxonomy"] == "breast"
    assert payload["case_name"] == "breast_demo"
    assert payload["openai_model"] == "gpt-5.4"
    assert payload["differential_expression_csv"].endswith("analysis\\diffexp\\gene_expression_graphclust\\differential_expression.csv")


def test_list_available_organ_packs_exposes_lung_and_breast() -> None:
    packs = list_available_organ_packs()
    pack_ids = {pack["id"] for pack in packs}
    assert {"lung", "breast"} <= pack_ids


def test_write_schema_exports_json_schema(tmp_path) -> None:
    output_path = tmp_path / "workflow.schema.json"
    result = write_schema(output_path)

    assert result["workflow_schema"] == str(output_path.resolve())
    schema = json.loads(output_path.read_text(encoding="utf-8"))
    assert schema["type"] == "object"
    assert "annotation_taxonomy" in schema["properties"]
    assert "lung" in schema["properties"]["annotation_taxonomy"]["enum"]
    assert "breast" in schema["properties"]["annotation_taxonomy"]["enum"]


def test_build_artifact_manifest_tracks_required_outputs(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    annotation_dir = output_root / "annotation"
    pathology_dir = output_root / "pathology_review"
    pipeline_structure_dir = output_root / "pipeline" / "structure_assignment"
    pipeline_validation_dir = output_root / "pipeline" / "validation"
    runtime_dir = output_root / "runtime_configs"
    for directory in [annotation_dir, pathology_dir, pipeline_structure_dir, pipeline_validation_dir, runtime_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    files_to_create = [
        annotation_dir / "cluster_evidence.json",
        annotation_dir / "cluster_annotations_openai.json",
        annotation_dir / "cluster_annotations_openai.csv",
        annotation_dir / "cluster_celltype_annotation.csv",
        annotation_dir / "annotation_case_review.json",
        annotation_dir / "annotation_review.html",
        pathology_dir / "index.html",
        pathology_dir / "cluster_reviews.json",
        pathology_dir / "structure_reviews.json",
        pathology_dir / "case_summary.json",
        pipeline_structure_dir / "structure_clustermap.pdf",
        pipeline_structure_dir / "cluster_structure_lookup.csv",
        pipeline_structure_dir / "run_summary.json",
        pipeline_structure_dir / "structure_assignments.csv",
        pipeline_validation_dir / "xenium_explorer_annotations_summary.csv",
        runtime_dir / "generated_runtime_config.json",
    ]
    for file_path in files_to_create:
        file_path.write_text("demo", encoding="utf-8")

    diffexp = tmp_path / "differential_expression.csv"
    projection = tmp_path / "projection.csv"
    base_config = tmp_path / "base_pipeline.json"
    diffexp.write_text("demo", encoding="utf-8")
    projection.write_text("demo", encoding="utf-8")
    base_config.write_text("{}", encoding="utf-8")

    workflow_config_path = tmp_path / "workflow.json"
    workflow_payload = {
        "case_name": "breast_case",
        "study_context": "Breast context",
        "base_pipeline_config": str(base_config),
        "output_root": str(output_root),
        "annotation_taxonomy": "breast",
        "differential_expression_csv": str(diffexp),
        "projection_csv": str(projection),
        "openai_enabled": True,
        "openai_api_key_env": "OPENAI_API_KEY",
        "openai_model": "gpt-5.4"
    }
    workflow_config_path.write_text(json.dumps(workflow_payload), encoding="utf-8")

    workflow_summary_path = output_root / "workflow_summary.json"
    workflow_summary = {
        "case_name": "breast_case",
        "output_root": str(output_root),
        "annotation_taxonomy": "breast",
        "runtime_base_pipeline_config": str(runtime_dir / "generated_runtime_config.json"),
        "annotation_outputs": {
            "cluster_evidence_json": str(annotation_dir / "cluster_evidence.json"),
            "cluster_annotations_json": str(annotation_dir / "cluster_annotations_openai.json"),
            "cluster_annotations_csv": str(annotation_dir / "cluster_annotations_openai.csv"),
            "compatibility_csv": str(annotation_dir / "cluster_celltype_annotation.csv"),
            "case_review_json": str(annotation_dir / "annotation_case_review.json"),
            "report_html": str(annotation_dir / "annotation_review.html")
        },
        "pathology_outputs": {
            "report_html": str(pathology_dir / "index.html"),
            "cluster_reviews_json": str(pathology_dir / "cluster_reviews.json"),
            "structure_reviews_json": str(pathology_dir / "structure_reviews.json"),
            "case_summary_json": str(pathology_dir / "case_summary.json")
        }
    }
    workflow_summary_path.write_text(json.dumps(workflow_summary), encoding="utf-8")

    manifest = build_artifact_manifest(
        workflow_config=validate_workflow_config(workflow_config_path),
        workflow_summary_path=workflow_summary_path,
    )

    assert manifest["organ_pack"]["id"] == "breast"
    assert manifest["artifact_counts"]["missing_required"] == 0
    assert manifest["artifact_counts"]["existing"] >= 10
    artifact_ids = {artifact["id"] for artifact in manifest["artifacts"]}
    assert "pipeline.structure_clustermap_pdf" in artifact_ids
    assert "workflow.workflow_summary_json" in artifact_ids


def test_build_manifest_writes_manifest_file(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    annotation_dir = output_root / "annotation"
    pathology_dir = output_root / "pathology_review"
    pipeline_structure_dir = output_root / "pipeline" / "structure_assignment"
    pipeline_validation_dir = output_root / "pipeline" / "validation"
    runtime_dir = output_root / "runtime_configs"
    for directory in [annotation_dir, pathology_dir, pipeline_structure_dir, pipeline_validation_dir, runtime_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    for file_path in [
        annotation_dir / "cluster_evidence.json",
        annotation_dir / "cluster_annotations_openai.json",
        annotation_dir / "cluster_annotations_openai.csv",
        annotation_dir / "cluster_celltype_annotation.csv",
        annotation_dir / "annotation_case_review.json",
        annotation_dir / "annotation_review.html",
        pathology_dir / "index.html",
        pathology_dir / "cluster_reviews.json",
        pathology_dir / "structure_reviews.json",
        pathology_dir / "case_summary.json",
        pipeline_structure_dir / "structure_clustermap.pdf",
        pipeline_structure_dir / "cluster_structure_lookup.csv",
        pipeline_structure_dir / "run_summary.json",
        pipeline_structure_dir / "structure_assignments.csv",
        pipeline_validation_dir / "xenium_explorer_annotations_summary.csv",
        runtime_dir / "generated_runtime_config.json",
    ]:
        file_path.write_text("demo", encoding="utf-8")

    diffexp = tmp_path / "differential_expression.csv"
    projection = tmp_path / "projection.csv"
    base_config = tmp_path / "base_pipeline.json"
    diffexp.write_text("demo", encoding="utf-8")
    projection.write_text("demo", encoding="utf-8")
    base_config.write_text("{}", encoding="utf-8")

    workflow_config_path = tmp_path / "workflow.json"
    workflow_payload = {
        "case_name": "lung_case",
        "study_context": "Lung context",
        "base_pipeline_config": str(base_config),
        "output_root": str(output_root),
        "annotation_taxonomy": "lung",
        "differential_expression_csv": str(diffexp),
        "projection_csv": str(projection),
        "openai_enabled": True,
        "openai_api_key_env": "OPENAI_API_KEY",
        "openai_model": "gpt-5.4"
    }
    workflow_config_path.write_text(json.dumps(workflow_payload), encoding="utf-8")

    workflow_summary_path = output_root / "workflow_summary.json"
    workflow_summary_path.write_text(
        json.dumps(
            {
                "case_name": "lung_case",
                "output_root": str(output_root),
                "annotation_taxonomy": "lung",
                "runtime_base_pipeline_config": str(runtime_dir / "generated_runtime_config.json"),
                "annotation_outputs": {
                    "cluster_evidence_json": str(annotation_dir / "cluster_evidence.json"),
                    "cluster_annotations_json": str(annotation_dir / "cluster_annotations_openai.json"),
                    "cluster_annotations_csv": str(annotation_dir / "cluster_annotations_openai.csv"),
                    "compatibility_csv": str(annotation_dir / "cluster_celltype_annotation.csv"),
                    "case_review_json": str(annotation_dir / "annotation_case_review.json"),
                    "report_html": str(annotation_dir / "annotation_review.html")
                },
                "pathology_outputs": {
                    "report_html": str(pathology_dir / "index.html"),
                    "cluster_reviews_json": str(pathology_dir / "cluster_reviews.json"),
                    "structure_reviews_json": str(pathology_dir / "structure_reviews.json"),
                    "case_summary_json": str(pathology_dir / "case_summary.json")
                }
            }
        ),
        encoding="utf-8",
    )

    result = build_manifest(workflow_config_path)

    manifest_path = Path(result["artifact_manifest_json"])
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["organ_pack"]["id"] == "lung"
