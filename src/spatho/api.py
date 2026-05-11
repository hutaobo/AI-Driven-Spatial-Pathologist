from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import json
from urllib import error, request

from pydantic import ValidationError

from histoseg.spatial_pathologist.full_auto import (
    load_full_auto_spatial_pathologist_config,
    run_full_auto_spatial_pathologist,
)

from .foundation import apply_foundation_evidence, foundation_evidence_requested
from .he_foundation import apply_he_contour_foundation, resolve_contour_geojson
from .local_annotation import run_workflow_with_local_cluster_annotation
from .manifest import write_artifact_manifest
from .organ_packs import get_organ_pack, list_organ_packs
from .schema import export_workflow_schema, validate_workflow_config
from .stgpt import apply_stgpt_evidence, inspect_stgpt_evidence, prepare_stgpt_evidence, stgpt_evidence_requested
from .templates import write_workflow_template
from .xenium import DEFAULT_XENIUM_PIXEL_SIZE_UM, write_xenium_rna_protein_alignment_bundle
from .evidence import EVIDENCE_SCHEMA_VERSION, export_evidence_schema


def run_workflow(config_path: str | Path, *, heuristic_only: bool = False) -> dict[str, str]:
    config_model = validate_workflow_config(config_path)
    if stgpt_evidence_requested(config_model):
        prepare_stgpt_evidence(config_model)
    if heuristic_only:
        config_model = config_model.model_copy(
            update={
                "openai_enabled": False,
                "cluster_annotation_backend": "heuristic",
                "pathology_review_backend": "heuristic",
            }
        )
    if config_model.cluster_annotation_backend == "pathology_ai_api":
        result = run_workflow_with_local_cluster_annotation(config_model)
    else:
        cfg = load_full_auto_spatial_pathologist_config(Path(config_path).resolve())
        if heuristic_only or config_model.cluster_annotation_backend == "heuristic":
            cfg = type(cfg)(
                **{
                    **cfg.__dict__,
                    "openai_enabled": False,
                }
            )
        result = run_full_auto_spatial_pathologist(cfg)
    if foundation_evidence_requested(config_model):
        result = apply_foundation_evidence(config_model, result)
    if config_model.he_contour_foundation_enabled:
        result = apply_he_contour_foundation(config_model, result)
    if foundation_evidence_requested(config_model):
        result = apply_foundation_evidence(config_model, result)
    if stgpt_evidence_requested(config_model):
        result = apply_stgpt_evidence(config_model, result)
    manifest_path = write_artifact_manifest(
        workflow_config=config_model,
        workflow_summary_path=result["workflow_summary_json"],
    )
    return {
        **result,
        "artifact_manifest_json": str(manifest_path),
    }


def _pathology_ai_health_report(base_url: str) -> dict[str, Any]:
    req = request.Request(
        url=base_url.rstrip("/") + "/health",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with request.urlopen(req, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("Expected JSON object", doc=str(payload), pos=0)
    return payload


def _health_ready(payload: dict[str, Any]) -> bool:
    if "ready" in payload:
        return payload.get("ready") is True
    return bool(payload.get("ok"))


def _check_evidence_schema_compat(workbench_dir: Path) -> list[str]:
    """Check that existing workbench artifact schema versions are compatible.

    Returns a list of human-readable incompatibility messages (empty when all
    is well).  This is surfaced by ``spatho doctor`` so operators learn about
    format drift before re-running a workflow.
    """
    issues: list[str] = []
    if not workbench_dir.exists():
        return issues

    # Check execution plan
    plan_path = workbench_dir / "execution_plan.json"
    if plan_path.exists():
        try:
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            file_ver = payload.get("schema_version", "unknown")
            if file_ver != EVIDENCE_SCHEMA_VERSION:
                issues.append(
                    f"execution_plan.json schema_version '{file_ver}' does not match "
                    f"current version '{EVIDENCE_SCHEMA_VERSION}'. Re-run the planner."
                )
        except json.JSONDecodeError:
            issues.append("execution_plan.json is not valid JSON.")

    # Check critic report
    critic_path = workbench_dir / "critic_report.json"
    if critic_path.exists():
        try:
            payload = json.loads(critic_path.read_text(encoding="utf-8"))
            file_ver = payload.get("schema_version", "unknown")
            if file_ver != EVIDENCE_SCHEMA_VERSION:
                issues.append(
                    f"critic_report.json schema_version '{file_ver}' does not match "
                    f"current version '{EVIDENCE_SCHEMA_VERSION}'. Re-run the critic."
                )
        except json.JSONDecodeError:
            issues.append("critic_report.json is not valid JSON.")

    # Check individual bundle files
    bundles_dir = workbench_dir / "bundles"
    if bundles_dir.exists():
        for bundle_path in sorted(bundles_dir.glob("*.json")):
            if bundle_path.name.endswith(".meta.json"):
                continue
            try:
                payload = json.loads(bundle_path.read_text(encoding="utf-8"))
                file_ver = payload.get("schema_version", "unknown")
                if file_ver != EVIDENCE_SCHEMA_VERSION:
                    issues.append(
                        f"Bundle '{bundle_path.name}' schema_version '{file_ver}' does not match "
                        f"current version '{EVIDENCE_SCHEMA_VERSION}'."
                    )
            except json.JSONDecodeError:
                issues.append(f"Bundle '{bundle_path.name}' is not valid JSON.")

    return issues


def workflow_doctor_report(config_path: str | Path | None = None) -> dict[str, Any]:
    config_resolved = Path(config_path).resolve() if config_path is not None else None
    api_key = os.environ.get("OPENAI_API_KEY")
    issues: list[str] = []
    report: dict[str, Any] = {
        "openai_api_key_present": bool(api_key),
        "openai_api_key_length": len(api_key) if api_key else 0,
        "config_path": str(config_resolved) if config_resolved is not None else None,
        "config_exists": bool(config_resolved and config_resolved.exists()),
    }
    if config_resolved is not None and not config_resolved.exists():
        issues.append("Workflow config file does not exist.")
    if config_resolved is not None and config_resolved.exists():
        try:
            cfg = validate_workflow_config(config_resolved)
        except (ValidationError, ValueError) as exc:
            issues.append(f"Workflow config validation failed: {exc}")
            report["schema_valid"] = False
            report["issues"] = issues
            report["ready_to_run"] = False
            return report

        pack = get_organ_pack(cfg.annotation_taxonomy)
        differential_expression_csv = cfg.differential_expression_csv
        projection_csv = cfg.projection_csv
        if not cfg.base_pipeline_config.exists():
            issues.append("Base pipeline config does not exist.")
        if differential_expression_csv is not None and not differential_expression_csv.exists():
            issues.append("Differential expression CSV does not exist.")
        if projection_csv is not None and not projection_csv.exists():
            issues.append("Projection CSV does not exist.")
        if cfg.openai_enabled and not api_key:
            issues.append("OpenAI is enabled but OPENAI_API_KEY is not available.")
        report.update(
            {
                "case_name": cfg.case_name,
                "annotation_taxonomy": cfg.annotation_taxonomy,
                "pathology_review_backend": cfg.pathology_review_backend,
                "pathology_ai_api_base_url": cfg.pathology_ai_api_base_url,
                "pathology_ai_top_k": cfg.pathology_ai_top_k,
                "pathology_ai_answer_language": cfg.pathology_ai_answer_language,
                "pathology_ai_document_ids": cfg.pathology_ai_document_ids,
                "cluster_annotation_backend": cfg.cluster_annotation_backend,
                "cluster_annotation_llm_base_url": cfg.cluster_annotation_llm_base_url,
                "cluster_annotation_min_llm_confidence": cfg.cluster_annotation_min_llm_confidence,
                "cluster_annotation_override_margin": cfg.cluster_annotation_override_margin,
                "cluster_annotation_require_marker_overlap": cfg.cluster_annotation_require_marker_overlap,
                "he_contour_foundation_enabled": cfg.he_contour_foundation_enabled,
                "he_contour_geojson": str(cfg.he_contour_geojson) if cfg.he_contour_geojson else None,
                "he_contour_key": cfg.he_contour_key,
                "he_foundation_model_id": cfg.he_foundation_model_id,
                "he_foundation_prompt_set": cfg.he_foundation_prompt_set,
                "he_foundation_top_k": cfg.he_foundation_top_k,
                "he_foundation_max_patch_side_px": cfg.he_foundation_max_patch_side_px,
                "he_visual_override_enabled": cfg.he_visual_override_enabled,
                "he_visual_override_min_llm_confidence": cfg.he_visual_override_min_llm_confidence,
                "he_visual_override_min_foundation_score": cfg.he_visual_override_min_foundation_score,
                "rna_foundation_enabled": cfg.rna_foundation_enabled,
                "rna_foundation_backend": cfg.rna_foundation_backend,
                "rna_foundation_cell_mapping_path": str(cfg.rna_foundation_cell_mapping_path) if cfg.rna_foundation_cell_mapping_path else None,
                "rna_foundation_cell_mapping_path_exists": cfg.rna_foundation_cell_mapping_path.exists() if cfg.rna_foundation_cell_mapping_path else None,
                "rna_foundation_cluster_summary_path": str(cfg.rna_foundation_cluster_summary_path) if cfg.rna_foundation_cluster_summary_path else None,
                "rna_foundation_cluster_summary_path_exists": cfg.rna_foundation_cluster_summary_path.exists() if cfg.rna_foundation_cluster_summary_path else None,
                "pathway_activity_enabled": cfg.pathway_activity_enabled,
                "pathway_activity_csv": str(cfg.pathway_activity_csv) if cfg.pathway_activity_csv else None,
                "pathway_activity_csv_exists": cfg.pathway_activity_csv.exists() if cfg.pathway_activity_csv else None,
                "niche_fusion_enabled": cfg.niche_fusion_enabled,
                "niche_fusion_backend": cfg.niche_fusion_backend,
                "stgpt_enabled": cfg.stgpt_enabled,
                "stgpt_backend": cfg.stgpt_backend,
                "stgpt_artifact_dir": str(cfg.stgpt_artifact_dir) if cfg.stgpt_artifact_dir else None,
                "stgpt_cell_embeddings_path": str(cfg.stgpt_cell_embeddings_path) if cfg.stgpt_cell_embeddings_path else None,
                "stgpt_structure_summary_path": str(cfg.stgpt_structure_summary_path) if cfg.stgpt_structure_summary_path else None,
                "stgpt_qc_report_path": str(cfg.stgpt_qc_report_path) if cfg.stgpt_qc_report_path else None,
                "stgpt_model_path": str(cfg.stgpt_model_path) if cfg.stgpt_model_path else None,
                "stgpt_config_path": str(cfg.stgpt_config_path) if cfg.stgpt_config_path else None,
                "stgpt_min_cell_coverage": cfg.stgpt_min_cell_coverage,
                "stgpt_require_qc_pass": cfg.stgpt_require_qc_pass,
                "pyxenium_mtm_enabled": cfg.pyxenium_mtm_enabled,
                "pyxenium_mtm_artifact_dir": (
                    str(cfg.pyxenium_mtm_artifact_dir) if cfg.pyxenium_mtm_artifact_dir else None
                ),
                "pyxenium_mtm_summary_path": (
                    str(cfg.pyxenium_mtm_summary_path) if cfg.pyxenium_mtm_summary_path else None
                ),
                "pyxenium_mtm_qc_report_path": (
                    str(cfg.pyxenium_mtm_qc_report_path) if cfg.pyxenium_mtm_qc_report_path else None
                ),
                "schema_valid": True,
                "organ_pack": pack.to_dict(),
                "base_pipeline_config": str(cfg.base_pipeline_config),
                "base_pipeline_config_exists": cfg.base_pipeline_config.exists(),
                "differential_expression_csv": str(differential_expression_csv) if differential_expression_csv else None,
                "differential_expression_csv_exists": differential_expression_csv.exists() if differential_expression_csv else None,
                "projection_csv": str(projection_csv) if projection_csv else None,
                "projection_csv_exists": projection_csv.exists() if projection_csv else None,
                "output_root": str(cfg.output_root),
                "dataset_modality": cfg.dataset_modality,
                "canonical_space": cfg.canonical_space,
                "export_space": cfg.export_space,
                "xenium_pixel_size_um": cfg.xenium_pixel_size_um,
                "segmentation_source": cfg.segmentation_source,
                "openai_enabled": bool(cfg.openai_enabled),
                "openai_model": cfg.openai_model,
            }
        )
        if cfg.rna_foundation_enabled:
            has_cell_mapping = bool(cfg.rna_foundation_cell_mapping_path and cfg.rna_foundation_cell_mapping_path.exists())
            has_cluster_summary = bool(
                cfg.rna_foundation_cluster_summary_path and cfg.rna_foundation_cluster_summary_path.exists()
            )
            if not has_cell_mapping and not has_cluster_summary:
                issues.append(
                    "RNA foundation evidence is enabled but neither rna_foundation_cell_mapping_path "
                    "nor rna_foundation_cluster_summary_path exists."
                )
        if cfg.pathway_activity_enabled:
            has_pathway_csv = bool(cfg.pathway_activity_csv and cfg.pathway_activity_csv.exists())
            has_diffexp = bool(differential_expression_csv and differential_expression_csv.exists())
            if not has_pathway_csv and not has_diffexp:
                issues.append(
                    "Pathway activity is enabled but neither pathway_activity_csv nor differential_expression_csv exists."
                )
        if cfg.stgpt_enabled:
            stgpt_report = inspect_stgpt_evidence(cfg, allow_local_pending=True)
            report["stgpt_evidence"] = {
                **stgpt_report,
                "paths": {key: str(value) for key, value in stgpt_report["paths"].items()},
            }
            issues.extend(stgpt_report["errors"])
        if cfg.pyxenium_mtm_enabled:
            has_summary = bool(cfg.pyxenium_mtm_summary_path and cfg.pyxenium_mtm_summary_path.exists())
            has_artifact_dir = bool(cfg.pyxenium_mtm_artifact_dir and cfg.pyxenium_mtm_artifact_dir.exists())
            report["pyxenium_mtm_evidence"] = {
                "enabled": True,
                "artifact_dir_exists": has_artifact_dir,
                "summary_path_exists": has_summary,
            }
            if not has_summary and not has_artifact_dir:
                issues.append(
                    "pyXenium mTM evidence is enabled but neither pyxenium_mtm_summary_path "
                    "nor pyxenium_mtm_artifact_dir exists."
                )

        # Schema version compatibility check for existing workbench artifacts
        report["evidence_schema_version"] = EVIDENCE_SCHEMA_VERSION
        report["human_review_policy"] = cfg.human_review_policy.model_dump()
        workbench_dir = cfg.output_root / "workbench"
        schema_compat_issues = _check_evidence_schema_compat(workbench_dir)
        if schema_compat_issues:
            report["evidence_schema_compat_issues"] = schema_compat_issues
            issues.extend(schema_compat_issues)
        if cfg.pathology_review_backend == "pathology_ai_api":
            try:
                health = _pathology_ai_health_report(str(cfg.pathology_ai_api_base_url))
                report["pathology_ai_api_health"] = health
                if not _health_ready(health):
                    issues.append("pathology-ai API health check did not report ready=true.")
            except (error.URLError, json.JSONDecodeError, TimeoutError) as exc:
                issues.append(f"pathology-ai API health check failed: {exc}")
        if cfg.cluster_annotation_backend == "pathology_ai_api":
            annotation_base_url = cfg.cluster_annotation_llm_base_url or cfg.pathology_ai_api_base_url
            try:
                health = _pathology_ai_health_report(str(annotation_base_url))
                report["cluster_annotation_api_health"] = health
                if not _health_ready(health):
                    issues.append("Local cluster annotation API health check did not report ready=true.")
            except (error.URLError, json.JSONDecodeError, TimeoutError) as exc:
                issues.append(f"Local cluster annotation API health check failed: {exc}")
        if cfg.he_contour_foundation_enabled:
            try:
                base_payload = json.loads(cfg.base_pipeline_config.read_text(encoding="utf-8"))
                base_dir = cfg.base_pipeline_config.parent
                base_cfg = {
                    key: (base_dir / value).resolve() if key.endswith("_dir") or key.endswith("_root") or key.endswith("_csv") or key.endswith("_tif") or key.endswith("_json") else value
                    for key, value in base_payload.items()
                }
                contour_geojson = resolve_contour_geojson(cfg, base_cfg)
                report["he_contour_geojson_resolved"] = str(contour_geojson)
            except Exception as exc:
                issues.append(f"H&E contour GeoJSON check failed: {exc}")
            try:
                health = _pathology_ai_health_report(str(cfg.pathology_ai_api_base_url))
                report["he_foundation_api_health"] = health
                if not _health_ready(health):
                    issues.append("pathology-ai API health check for H&E foundation mode did not report ready=true.")
            except (error.URLError, json.JSONDecodeError, TimeoutError) as exc:
                issues.append(f"pathology-ai API health check for H&E foundation mode failed: {exc}")
    report["issues"] = issues
    report["ready_to_run"] = not issues
    return report


def list_available_organ_packs() -> list[dict[str, Any]]:
    return [pack.to_dict() for pack in list_organ_packs()]


def init_workflow(
    output_path: str | Path,
    *,
    organ: str,
    case_name: str,
    dataset_root: str | Path,
    base_pipeline_config: str | Path,
    output_root: str | Path | None = None,
    study_context: str | None = None,
    openai_model: str = "gpt-5.4",
) -> dict[str, str]:
    path = write_workflow_template(
        output_path,
        organ=organ,
        case_name=case_name,
        dataset_root=dataset_root,
        base_pipeline_config=base_pipeline_config,
        output_root=output_root,
        study_context=study_context,
        openai_model=openai_model,
    )
    return {"workflow_config": str(path)}


def write_schema(output_path: str | Path) -> dict[str, str]:
    path = export_workflow_schema(output_path)
    evidence_schema_path = Path(output_path).with_suffix("").parent / "evidence_bundle.schema.json"
    export_evidence_schema(evidence_schema_path)
    return {"workflow_schema": str(path), "evidence_schema": str(evidence_schema_path)}


def build_manifest(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, str]:
    config = validate_workflow_config(config_path)
    summary_path = config.output_root / "workflow_summary.json"
    manifest_path = write_artifact_manifest(
        workflow_config=config,
        workflow_summary_path=summary_path,
        output_path=output_path,
    )
    return {
        "workflow_summary_json": str(summary_path),
        "artifact_manifest_json": str(manifest_path),
    }


def write_xenium_alignment_fixtures(
    output_dir: str | Path,
    *,
    metadata_pixel_size_um: float | int | str | None = None,
    fallback_pixel_size_um: float | int | str | None = None,
    segmentation_source: str = "ranger_default",
) -> dict[str, str]:
    bundle = write_xenium_rna_protein_alignment_bundle(
        output_dir,
        metadata_pixel_size_um=metadata_pixel_size_um,
        fallback_pixel_size_um=fallback_pixel_size_um if fallback_pixel_size_um is not None else DEFAULT_XENIUM_PIXEL_SIZE_UM,
        segmentation_source=segmentation_source,
    )
    return bundle
