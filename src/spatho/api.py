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

from .manifest import write_artifact_manifest
from .organ_packs import get_organ_pack, list_organ_packs
from .schema import export_workflow_schema, validate_workflow_config
from .templates import write_workflow_template
from .xenium import DEFAULT_XENIUM_PIXEL_SIZE_UM, write_xenium_rna_protein_alignment_bundle


def run_workflow(config_path: str | Path, *, heuristic_only: bool = False) -> dict[str, str]:
    config_model = validate_workflow_config(config_path)
    if heuristic_only:
        config_model = config_model.model_copy(update={"openai_enabled": False})
    cfg = load_full_auto_spatial_pathologist_config(Path(config_path).resolve())
    if heuristic_only:
        cfg = type(cfg)(
            **{
                **cfg.__dict__,
                "openai_enabled": False,
            }
        )
    result = run_full_auto_spatial_pathologist(cfg)
    manifest_path = write_artifact_manifest(
        workflow_config=config_model,
        workflow_summary_path=result["workflow_summary_json"],
    )
    return {
        **result,
        "artifact_manifest_json": str(manifest_path),
    }


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
        if cfg.pathology_review_backend == "pathology_ai_api":
            try:
                req = request.Request(
                    url=str(cfg.pathology_ai_api_base_url).rstrip("/") + "/health",
                    headers={"Accept": "application/json"},
                    method="GET",
                )
                with request.urlopen(req, timeout=5) as response:
                    report["pathology_ai_api_health"] = json.loads(response.read().decode("utf-8"))
            except (error.URLError, json.JSONDecodeError, TimeoutError) as exc:
                issues.append(f"pathology-ai API health check failed: {exc}")
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
    return {"workflow_schema": str(path)}


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
