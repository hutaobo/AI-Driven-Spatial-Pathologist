from __future__ import annotations

"""Typed evidence schema for the spatho agentic evidence toolchain.

All evidence flowing through the Planner / Executor / Critic / Reporter
pipeline is represented as :class:`EvidenceBundle` objects so that every
source (stGPT, pyXenium, PLIP, pathway) produces the same schema and can
be validated, cached, and replayed deterministically.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field


EVIDENCE_SCHEMA_VERSION = "0.2.0"

QcStatus = Literal["ok", "warning", "fail", "unknown"]
EvidenceUnit = Literal["cell", "region", "structure", "case"]
ConflictResolution = Literal["auto_skip", "block", "flag_only"]


# ---------------------------------------------------------------------------
# EvidenceBundle – the canonical evidence atom
# ---------------------------------------------------------------------------


class EvidenceBundle(BaseModel):
    """A single piece of structured evidence produced by one tool call.

    Design goals
    ------------
    * All evidence channels (stGPT, PLIP, pathway, pyXenium …) emit the
      same schema so the Critic can compare them without source-specific
      parsing.
    * ``schema_version`` is written at creation time so ``spatho doctor``
      can detect format drift without executing the full workflow.
    * ``input_hash`` enables the Executor to skip redundant computation
      and replay prior results deterministically.
    * ``model_derived=True`` marks outputs that must never be presented as
      measured expression in a clinical report.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=EVIDENCE_SCHEMA_VERSION)
    evidence_id: str = Field(min_length=1, description="Globally unique ID, e.g. 'stgpt.structure.3'.")
    unit: EvidenceUnit = Field(description="Granularity at which this evidence was computed.")
    unit_id: str = Field(description="ID of the cell / region / structure / case this evidence describes.")
    source: str = Field(min_length=1, description="Tool that produced this evidence, e.g. 'stgpt', 'plip'.")
    evidence_type: str = Field(min_length=1, description="Sub-type tag, e.g. 'morpho_molecular_embedding'.")
    measured: bool = Field(default=False, description="True only for raw measured data, never for model outputs.")
    model_derived: bool = Field(default=True, description="True for any output that involves a learned model.")
    qc_status: QcStatus = Field(default="unknown")
    summary: str = Field(default="", description="Human-readable summary of the evidence.")
    supporting_artifacts: list[str] = Field(
        default_factory=list,
        description="File paths or artifact IDs that back this evidence.",
    )
    # Execution provenance
    tool_name: str = Field(default="", description="Specific tool function that produced this bundle.")
    input_hash: str = Field(default="", description="SHA-256 of all inputs (config + data) for cache lookup.")
    model_version: str = Field(default="", description="Checkpoint / model version string.")
    elapsed_seconds: float | None = Field(default=None, description="Wall-clock time of the tool call.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp of bundle creation.",
    )
    # Optional conflict / adjudication fields written by the Critic
    requires_human_review: bool = Field(default=False)
    conflict_note: str = Field(default="", description="Description of cross-modal conflict, if any.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        return self.model_json(indent=2, ensure_ascii=False)

    def save(self, path: Path) -> Path:
        """Persist this bundle to *path* and return the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "EvidenceBundle":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @classmethod
    def json_schema_document(cls) -> dict[str, Any]:
        return cls.model_json_schema()


# ---------------------------------------------------------------------------
# ToolCallMeta – deterministic cache record written alongside every artifact
# ---------------------------------------------------------------------------


class ToolCallMeta(BaseModel):
    """Written as ``<artifact_stem>.meta.json`` beside every tool output.

    The Executor reads this file before re-running a tool.  When the
    ``input_hash`` matches, the cached artifact is reused directly.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=EVIDENCE_SCHEMA_VERSION)
    tool_name: str
    input_hash: str = Field(description="SHA-256 of all inputs that determine tool output.")
    model_version: str = Field(default="")
    spatho_version: str = Field(default="")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    elapsed_seconds: float | None = None
    output_artifact: str = Field(description="Path of the primary output artifact this meta covers.")

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "ToolCallMeta":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @classmethod
    def meta_path_for(cls, artifact_path: Path) -> Path:
        """Return the companion ``.meta.json`` path for *artifact_path*."""
        return artifact_path.with_suffix("").with_suffix(artifact_path.suffix + ".meta.json")


# ---------------------------------------------------------------------------
# CorrectionBundle – human-review / fine-tuning feedback record
# ---------------------------------------------------------------------------


CorrectionType = Literal["label_override", "reject", "flag_for_finetuning"]


class CorrectionBundle(BaseModel):
    """Records a human reviewer's correction to a specific :class:`EvidenceBundle`.

    These records are aggregated in ``evidence_manifest.json`` and drive
    the closed fine-tuning loop.  When enough ``flag_for_finetuning``
    records accumulate (see :func:`should_trigger_finetuning`), a training
    split is generated automatically.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=EVIDENCE_SCHEMA_VERSION)
    correction_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1, description="ID of the EvidenceBundle being corrected.")
    unit_id: str = Field(description="Structure / region / cell ID affected.")
    correction_type: CorrectionType
    original_label: str = Field(default="")
    corrected_label: str = Field(default="")
    reviewer_id: str = Field(default="anonymous")
    note: str = Field(default="")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "CorrectionBundle":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# ExecutionPlan – serialisable DAG produced by the Planner
# ---------------------------------------------------------------------------


class ToolNode(BaseModel):
    """One node in the execution DAG."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list, description="node_ids that must complete first.")
    params: dict[str, Any] = Field(default_factory=dict)
    skip_if_cached: bool = Field(default=True)


class ExecutionPlan(BaseModel):
    """Serialisable tool DAG output by the Planner step."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=EVIDENCE_SCHEMA_VERSION)
    plan_id: str = Field(min_length=1)
    case_name: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    nodes: list[ToolNode] = Field(default_factory=list)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "ExecutionPlan":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def topological_order(self) -> list[ToolNode]:
        """Return nodes in a valid execution order respecting ``depends_on``."""
        index = {node.node_id: node for node in self.nodes}
        visited: set[str] = set()
        order: list[ToolNode] = []

        def _visit(node_id: str) -> None:
            if node_id in visited:
                return
            node = index[node_id]
            for dep in node.depends_on:
                _visit(dep)
            visited.add(node_id)
            order.append(node)

        for node in self.nodes:
            _visit(node.node_id)
        return order


# ---------------------------------------------------------------------------
# CriticReport – output of the Critic step
# ---------------------------------------------------------------------------


class CriticReport(BaseModel):
    """Summary produced after the Critic inspects all EvidenceBundles."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=EVIDENCE_SCHEMA_VERSION)
    case_name: str
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_bundles: int = 0
    passed: int = 0
    warned: int = 0
    failed: int = 0
    flagged_for_human_review: list[str] = Field(default_factory=list, description="evidence_ids.")
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list, description="Missing expected evidence channels.")
    overall_status: QcStatus = "unknown"

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "CriticReport":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Fine-tuning trigger helper
# ---------------------------------------------------------------------------


def should_trigger_finetuning(
    corrections: list[CorrectionBundle],
    *,
    threshold: int = 50,
) -> bool:
    """Return True when enough ``flag_for_finetuning`` corrections have accumulated."""
    count = sum(1 for c in corrections if c.correction_type == "flag_for_finetuning")
    return count >= threshold


def compute_input_hash(inputs: dict[str, Any]) -> str:
    """Stable SHA-256 of a JSON-serialisable *inputs* dict."""
    serialised = json.dumps(inputs, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def export_evidence_schema(output_path: str | Path) -> Path:
    """Write the JSON Schema for :class:`EvidenceBundle` to *output_path*."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(EvidenceBundle.json_schema_document(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "QcStatus",
    "EvidenceUnit",
    "ConflictResolution",
    "CorrectionType",
    "EvidenceBundle",
    "ToolCallMeta",
    "CorrectionBundle",
    "ToolNode",
    "ExecutionPlan",
    "CriticReport",
    "should_trigger_finetuning",
    "compute_input_hash",
    "export_evidence_schema",
]
