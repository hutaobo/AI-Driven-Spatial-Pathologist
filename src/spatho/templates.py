from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .organ_packs import get_organ_pack
from .xenium import (
    CANONICAL_SPACE_PHYSICAL_UM,
    DATASET_MODALITY_XENIUM_RNA_PROTEIN,
    DEFAULT_XENIUM_PIXEL_SIZE_UM,
    EXPORT_SPACE_XENIUM_EXPLORER_PIXEL,
)


def build_workflow_template(
    *,
    organ: str,
    case_name: str,
    dataset_root: str | Path,
    base_pipeline_config: str | Path,
    output_root: str | Path | None = None,
    study_context: str | None = None,
    openai_model: str = "gpt-5.4",
) -> dict[str, Any]:
    pack = get_organ_pack(organ)
    dataset_path = Path(dataset_root).resolve()
    base_cfg_path = Path(base_pipeline_config).resolve()
    if output_root is None:
        output_path = base_cfg_path.parent.parent / "outputs" / "full_auto_openai"
    else:
        output_path = Path(output_root).resolve()

    payload = {
        "case_name": case_name,
        "study_context": study_context or pack.default_study_context,
        "base_pipeline_config": str(base_cfg_path),
        "output_root": str(output_path),
        "annotation_taxonomy": pack.annotation_taxonomy,
        "pathology_review_backend": "openai",
        "pathology_ai_api_base_url": "http://127.0.0.1:8000",
        "pathology_ai_top_k": 6,
        "pathology_ai_answer_language": "en",
        "pathology_ai_document_ids": [],
        "cluster_annotation_backend": "auto",
        "cluster_annotation_llm_base_url": None,
        "cluster_annotation_min_llm_confidence": 0.60,
        "cluster_annotation_override_margin": 0.15,
        "cluster_annotation_require_marker_overlap": True,
        "he_contour_foundation_enabled": False,
        "he_contour_geojson": None,
        "he_contour_key": "spatho_he_contours",
        "he_foundation_model_id": "vinid/plip",
        "he_foundation_prompt_set": "breast_contour_v1",
        "he_foundation_top_k": 5,
        "he_foundation_max_patch_side_px": 1024,
        "he_visual_override_enabled": True,
        "he_visual_override_min_llm_confidence": 0.70,
        "he_visual_override_min_foundation_score": 0.35,
        "differential_expression_csv": str(
            dataset_path / "analysis" / "diffexp" / "gene_expression_graphclust" / "differential_expression.csv"
        ),
        "projection_csv": str(
            dataset_path / "analysis" / "umap" / "gene_expression_2_components" / "projection.csv"
        ),
        "dataset_modality": DATASET_MODALITY_XENIUM_RNA_PROTEIN,
        "canonical_space": CANONICAL_SPACE_PHYSICAL_UM,
        "export_space": EXPORT_SPACE_XENIUM_EXPLORER_PIXEL,
        "xenium_pixel_size_um": DEFAULT_XENIUM_PIXEL_SIZE_UM,
        "segmentation_source": "ranger_default",
        "openai_enabled": True,
        "openai_api_key_env": "OPENAI_API_KEY",
        "openai_model": openai_model,
        "openai_reasoning_effort": pack.workflow_defaults["openai_reasoning_effort"],
        "openai_store": pack.workflow_defaults["openai_store"],
        "force_recompute_annotation": True,
        "force_recompute_pipeline": True,
        "top_positive_markers": pack.workflow_defaults["top_positive_markers"],
        "top_negative_markers": pack.workflow_defaults["top_negative_markers"],
        "min_log2fc": pack.workflow_defaults["min_log2fc"],
        "max_adjusted_p_value": pack.workflow_defaults["max_adjusted_p_value"],
        "top_neighbors": pack.workflow_defaults["top_neighbors"],
        "low_confidence_threshold": pack.workflow_defaults["low_confidence_threshold"],
        "ambiguity_margin_threshold": pack.workflow_defaults["ambiguity_margin_threshold"],
        "top_clusters_per_structure": pack.workflow_defaults["top_clusters_per_structure"],
    }
    return payload


def write_workflow_template(
    output_path: str | Path,
    *,
    organ: str,
    case_name: str,
    dataset_root: str | Path,
    base_pipeline_config: str | Path,
    output_root: str | Path | None = None,
    study_context: str | None = None,
    openai_model: str = "gpt-5.4",
) -> Path:
    path = Path(output_path).resolve()
    payload = build_workflow_template(
        organ=organ,
        case_name=case_name,
        dataset_root=dataset_root,
        base_pipeline_config=base_pipeline_config,
        output_root=output_root,
        study_context=study_context,
        openai_model=openai_model,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
