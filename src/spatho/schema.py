from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .organ_packs import get_organ_pack, list_organ_packs
from .xenium import (
    CANONICAL_SPACE_PHYSICAL_UM,
    DATASET_MODALITY_XENIUM_RNA_PROTEIN,
    DEFAULT_XENIUM_PIXEL_SIZE_UM,
    EXPORT_SPACE_XENIUM_EXPLORER_PIXEL,
    VALID_SEGMENTATION_SOURCES,
    validate_segmentation_source,
)


def _resolve_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


class WorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_name: str = Field(min_length=1)
    study_context: str = Field(min_length=1)
    base_pipeline_config: Path
    output_root: Path
    annotation_taxonomy: str = Field(default="lung")

    pathology_review_backend: str = Field(default="openai")
    pathology_ai_api_base_url: str = "http://127.0.0.1:8000"
    pathology_ai_top_k: int = Field(default=6, ge=1, le=12)
    pathology_ai_answer_language: str = Field(default="en", min_length=2, max_length=32)
    pathology_ai_document_ids: list[str] = Field(default_factory=list, max_length=64)

    cluster_annotation_backend: str = Field(default="auto")
    cluster_annotation_llm_base_url: str | None = None
    cluster_annotation_min_llm_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    cluster_annotation_override_margin: float = Field(default=0.15, ge=0.0, le=1.0)
    cluster_annotation_require_marker_overlap: bool = True

    he_contour_foundation_enabled: bool = False
    he_contour_geojson: Path | None = None
    he_contour_key: str = Field(default="spatho_he_contours", min_length=1)
    he_foundation_model_id: str = Field(default="vinid/plip", min_length=1)
    he_foundation_prompt_set: str = Field(default="breast_contour_v1", min_length=1)
    he_foundation_top_k: int = Field(default=5, ge=1, le=10)
    he_foundation_max_patch_side_px: int = Field(default=1024, ge=128, le=4096)
    he_visual_override_enabled: bool = True
    he_visual_override_min_llm_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    he_visual_override_min_foundation_score: float = Field(default=0.35, ge=0.0, le=1.0)

    differential_expression_csv: Path | None = None
    projection_csv: Path | None = None

    dataset_modality: str = Field(default=DATASET_MODALITY_XENIUM_RNA_PROTEIN)
    canonical_space: str = Field(default=CANONICAL_SPACE_PHYSICAL_UM)
    export_space: str = Field(default=EXPORT_SPACE_XENIUM_EXPLORER_PIXEL)
    xenium_pixel_size_um: float = Field(default=DEFAULT_XENIUM_PIXEL_SIZE_UM, gt=0.0)
    segmentation_source: str = Field(default="ranger_default")

    openai_enabled: bool = True
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_model: str = "gpt-5.4"
    openai_reasoning_effort: str = "medium"
    openai_store: bool = False

    force_recompute_annotation: bool = False
    force_recompute_pipeline: bool = False

    top_positive_markers: int = Field(default=15, ge=1)
    top_negative_markers: int = Field(default=6, ge=0)
    min_log2fc: float = 0.5
    max_adjusted_p_value: float = 0.05
    top_neighbors: int = Field(default=5, ge=1)

    low_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    ambiguity_margin_threshold: float = Field(default=0.08, ge=0.0, le=1.0)
    top_clusters_per_structure: int = Field(default=8, ge=1)

    @field_validator("annotation_taxonomy")
    @classmethod
    def _validate_annotation_taxonomy(cls, value: str) -> str:
        pack = get_organ_pack(value)
        return pack.annotation_taxonomy

    @field_validator("pathology_review_backend")
    @classmethod
    def _validate_pathology_review_backend(cls, value: str) -> str:
        supported = {"heuristic", "openai", "pathology_ai_api"}
        normalized = str(value).strip()
        if normalized not in supported:
            raise ValueError(f"pathology_review_backend must be one of: {', '.join(sorted(supported))}")
        return normalized

    @field_validator("cluster_annotation_backend")
    @classmethod
    def _validate_cluster_annotation_backend(cls, value: str) -> str:
        supported = {"auto", "heuristic", "openai", "pathology_ai_api"}
        normalized = str(value).strip()
        if normalized not in supported:
            raise ValueError(f"cluster_annotation_backend must be one of: {', '.join(sorted(supported))}")
        return normalized

    @field_validator("dataset_modality")
    @classmethod
    def _validate_dataset_modality(cls, value: str) -> str:
        normalized = str(value).strip()
        if normalized != DATASET_MODALITY_XENIUM_RNA_PROTEIN:
            raise ValueError(f"dataset_modality must be '{DATASET_MODALITY_XENIUM_RNA_PROTEIN}'")
        return normalized

    @field_validator("canonical_space")
    @classmethod
    def _validate_canonical_space(cls, value: str) -> str:
        normalized = str(value).strip()
        if normalized != CANONICAL_SPACE_PHYSICAL_UM:
            raise ValueError(f"canonical_space must be '{CANONICAL_SPACE_PHYSICAL_UM}'")
        return normalized

    @field_validator("export_space")
    @classmethod
    def _validate_export_space(cls, value: str) -> str:
        normalized = str(value).strip()
        if normalized != EXPORT_SPACE_XENIUM_EXPLORER_PIXEL:
            raise ValueError(f"export_space must be '{EXPORT_SPACE_XENIUM_EXPLORER_PIXEL}'")
        return normalized

    @field_validator("segmentation_source")
    @classmethod
    def _validate_segmentation_source(cls, value: str) -> str:
        return validate_segmentation_source(value)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "WorkflowConfig":
        config_path = Path(path).resolve()
        payload: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        path_fields = {
            "base_pipeline_config",
            "output_root",
            "differential_expression_csv",
            "projection_csv",
            "he_contour_geojson",
        }
        for field_name in path_fields:
            if field_name in payload:
                payload[field_name] = _resolve_path(payload[field_name], base_dir=config_path.parent)
        return cls.model_validate(payload)

    @classmethod
    def json_schema_document(cls) -> dict[str, Any]:
        schema = cls.model_json_schema()
        supported = [pack.id for pack in list_organ_packs()]
        properties = schema.setdefault("properties", {})
        if "annotation_taxonomy" in properties:
            properties["annotation_taxonomy"]["enum"] = supported
        if "pathology_review_backend" in properties:
            properties["pathology_review_backend"]["enum"] = ["heuristic", "openai", "pathology_ai_api"]
        if "cluster_annotation_backend" in properties:
            properties["cluster_annotation_backend"]["enum"] = ["auto", "heuristic", "openai", "pathology_ai_api"]
        if "he_foundation_model_id" in properties:
            properties["he_foundation_model_id"]["enum"] = ["vinid/plip"]
        if "he_foundation_prompt_set" in properties:
            properties["he_foundation_prompt_set"]["enum"] = ["breast_contour_v1"]
        if "dataset_modality" in properties:
            properties["dataset_modality"]["enum"] = [DATASET_MODALITY_XENIUM_RNA_PROTEIN]
        if "canonical_space" in properties:
            properties["canonical_space"]["enum"] = [CANONICAL_SPACE_PHYSICAL_UM]
        if "export_space" in properties:
            properties["export_space"]["enum"] = [EXPORT_SPACE_XENIUM_EXPLORER_PIXEL]
        if "segmentation_source" in properties:
            properties["segmentation_source"]["enum"] = list(VALID_SEGMENTATION_SOURCES)
        return schema

    def to_payload(self) -> dict[str, Any]:
        payload = self.model_dump()
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


def export_workflow_schema(output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(WorkflowConfig.json_schema_document(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def validate_workflow_config(config_path: str | Path) -> WorkflowConfig:
    return WorkflowConfig.from_json_file(config_path)
