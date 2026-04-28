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
