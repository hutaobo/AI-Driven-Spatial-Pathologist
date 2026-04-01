from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .organ_packs import get_organ_pack, list_organ_packs


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

    differential_expression_csv: Path | None = None
    projection_csv: Path | None = None

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

    @classmethod
    def from_json_file(cls, path: str | Path) -> "WorkflowConfig":
        config_path = Path(path).resolve()
        payload: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        path_fields = {
            "base_pipeline_config",
            "output_root",
            "differential_expression_csv",
            "projection_csv",
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
