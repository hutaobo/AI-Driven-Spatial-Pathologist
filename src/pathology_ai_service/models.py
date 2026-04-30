from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_id: str = Field(min_length=1)
    title: str | None = None
    text: str = Field(min_length=1)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentInput] = Field(min_length=1)
    chunk_size_chars: int | None = Field(default=None, ge=200, le=4000)
    chunk_overlap_chars: int | None = Field(default=None, ge=0, le=1000)


class DocumentUpsertResult(BaseModel):
    document_id: str
    chunk_count: int


class DocumentUpsertResponse(BaseModel):
    collection: str
    documents: list[DocumentUpsertResult]
    chunk_count: int


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
    document_ids: list[str] = Field(default_factory=list, max_length=64)
    top_k: int | None = Field(default=None, ge=1, le=20)
    answer_language: str = Field(default="en", min_length=2, max_length=32)
    entity_id: str | None = None
    entity_name: str | None = None
    review_id: str | None = None


class AnnotationLabelSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    broad_family: str | None = None
    malignancy_state: str | None = None
    description: str | None = None
    marker_genes: list[str] = Field(default_factory=list)
    negative_markers: list[str] = Field(default_factory=list)


class ClusterAnnotationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_name: str = Field(min_length=1)
    study_context: str = Field(min_length=1)
    annotation_taxonomy: str = Field(min_length=1)
    controlled_vocabulary: list[AnnotationLabelSpec] = Field(min_length=1)
    cluster_evidence: dict[str, Any] = Field(default_factory=dict)
    heuristic_annotation: dict[str, Any] = Field(default_factory=dict)


class ClusterAnnotationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_id: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    review_priority: Literal["low", "medium", "high"]
    supporting_markers: list[str] = Field(default_factory=list, max_length=8)
    conflicting_markers: list[str] = Field(default_factory=list, max_length=8)
    alternative_label_ids: list[str] = Field(default_factory=list, max_length=4)
    reasoning_summary: str = ""
    tumor_evidence: list[str] = Field(default_factory=list, max_length=8)
    recommended_follow_up: list[str] = Field(default_factory=list, max_length=6)


class HEContourInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    contour_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    structure_id: int | None = None
    structure_name: str | None = None


class HEContourClass(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)


class HEContourClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contour_id: str
    image_path: str
    structure_id: int | None = None
    structure_name: str | None = None
    top_classes: list[HEContourClass] = Field(default_factory=list)
    patch_quality: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class HEContourClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_name: str = Field(min_length=1)
    contours: list[HEContourInput] = Field(min_length=1)
    model_id: str = "vinid/plip"
    prompt_set: str = "breast_contour_v1"
    top_k: int = Field(default=5, ge=1, le=10)


class HEContourClassifyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_name: str
    model_id: str
    prompt_set: str
    classifications: list[HEContourClassification]
    warnings: list[str] = Field(default_factory=list)


class StructureMultimodalNamingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_name: str = Field(min_length=1)
    study_context: str = Field(min_length=1)
    annotation_taxonomy: str = Field(min_length=1)
    structure: dict[str, Any] = Field(default_factory=dict)
    current_review: dict[str, Any] = Field(default_factory=dict)
    he_visual_summary: dict[str, Any] = Field(default_factory=dict)
    multimodal_evidence: dict[str, Any] = Field(default_factory=dict)
    override_policy: dict[str, Any] = Field(default_factory=dict)


class StructureMultimodalNamingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure_id: int | None = None
    pre_visual_name: str = ""
    final_name: str = Field(min_length=1)
    visual_override: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    review_priority: Literal["low", "medium", "high"] = "medium"
    reasoning_summary: str = ""
    visual_evidence: list[str] = Field(default_factory=list, max_length=10)
    molecular_evidence: list[str] = Field(default_factory=list, max_length=10)
    conflicts: list[str] = Field(default_factory=list, max_length=10)
    recommended_checks: list[str] = Field(default_factory=list, max_length=8)


class Citation(BaseModel):
    citation_id: str
    document_id: str
    chunk_id: str
    title: str | None = None
    score: float
    text_excerpt: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalStats(BaseModel):
    candidate_count: int
    returned_count: int
    document_filter_applied: bool


class ReviewModelResponse(BaseModel):
    summary: str
    interpretation: str
    confidence: float = Field(ge=0.0, le=1.0)
    key_evidence: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    recommended_follow_up: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)


class ReviewResponse(BaseModel):
    review_type: Literal["structure", "case"]
    question: str
    answer_language: str
    summary: str
    interpretation: str
    confidence: float
    key_evidence: list[str]
    caveats: list[str]
    recommended_follow_up: list[str]
    citations: list[Citation]
    retrieval: RetrievalStats
