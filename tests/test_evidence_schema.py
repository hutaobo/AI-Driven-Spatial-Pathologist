"""Tests for the EvidenceBundle typed schema and related data models."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatho.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    CorrectionBundle,
    CriticReport,
    EvidenceBundle,
    ExecutionPlan,
    ToolCallMeta,
    ToolNode,
    compute_input_hash,
    export_evidence_schema,
    should_trigger_finetuning,
)
from spatho.schema import HumanReviewPolicy, WorkflowConfig


# ---------------------------------------------------------------------------
# EvidenceBundle
# ---------------------------------------------------------------------------


def test_evidence_bundle_defaults() -> None:
    bundle = EvidenceBundle(
        evidence_id="stgpt.structure.3",
        unit="structure",
        unit_id="3",
        source="stgpt",
        evidence_type="morpho_molecular_embedding",
    )
    assert bundle.schema_version == EVIDENCE_SCHEMA_VERSION
    assert bundle.model_derived is True
    assert bundle.measured is False
    assert bundle.qc_status == "unknown"
    assert bundle.requires_human_review is False


def test_evidence_bundle_round_trip(tmp_path: Path) -> None:
    bundle = EvidenceBundle(
        evidence_id="plip.region.7",
        unit="region",
        unit_id="7",
        source="plip",
        evidence_type="he_contour",
        qc_status="warning",
        summary="Possible artifact region.",
        supporting_artifacts=["he/patch_manifest.json"],
    )
    path = tmp_path / "bundle.json"
    bundle.save(path)
    loaded = EvidenceBundle.load(path)
    assert loaded.evidence_id == bundle.evidence_id
    assert loaded.qc_status == "warning"
    assert loaded.supporting_artifacts == ["he/patch_manifest.json"]


def test_evidence_bundle_json_schema_has_version_field() -> None:
    schema = EvidenceBundle.json_schema_document()
    assert "schema_version" in schema["properties"]


def test_evidence_bundle_rejects_extra_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EvidenceBundle(
            evidence_id="x",
            unit="case",
            unit_id="c",
            source="s",
            evidence_type="t",
            nonexistent_field="oops",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# ToolCallMeta
# ---------------------------------------------------------------------------


def test_tool_call_meta_round_trip(tmp_path: Path) -> None:
    artifact = tmp_path / "embeddings.parquet"
    artifact.write_bytes(b"placeholder")
    meta = ToolCallMeta(
        tool_name="stgpt.runtime.embed_regions",
        input_hash="abc123",
        model_version="v1.0",
        output_artifact=str(artifact),
        elapsed_seconds=3.14,
    )
    meta_path = ToolCallMeta.meta_path_for(artifact)
    meta.save(meta_path)
    loaded = ToolCallMeta.load(meta_path)
    assert loaded.input_hash == "abc123"
    assert loaded.elapsed_seconds == pytest.approx(3.14)


def test_tool_call_meta_path_convention(tmp_path: Path) -> None:
    artifact = tmp_path / "stgpt" / "cell_embeddings.parquet"
    meta_path = ToolCallMeta.meta_path_for(artifact)
    assert meta_path.name == "cell_embeddings.parquet.meta.json"
    assert meta_path.parent == artifact.parent


# ---------------------------------------------------------------------------
# CorrectionBundle
# ---------------------------------------------------------------------------


def test_correction_bundle_round_trip(tmp_path: Path) -> None:
    corr = CorrectionBundle(
        correction_id="corr-001",
        evidence_id="stgpt.structure.3",
        unit_id="3",
        correction_type="label_override",
        original_label="tumor",
        corrected_label="stroma",
        reviewer_id="dr_smith",
        note="H&E clearly shows stroma.",
    )
    path = tmp_path / "correction.json"
    corr.save(path)
    loaded = CorrectionBundle.load(path)
    assert loaded.correction_type == "label_override"
    assert loaded.corrected_label == "stroma"


# ---------------------------------------------------------------------------
# should_trigger_finetuning
# ---------------------------------------------------------------------------


def _make_corrections(n_flag: int, n_other: int) -> list[CorrectionBundle]:
    corrections: list[CorrectionBundle] = []
    for i in range(n_flag):
        corrections.append(
            CorrectionBundle(
                correction_id=f"flag-{i}",
                evidence_id=f"stgpt.structure.{i}",
                unit_id=str(i),
                correction_type="flag_for_finetuning",
            )
        )
    for i in range(n_other):
        corrections.append(
            CorrectionBundle(
                correction_id=f"reject-{i}",
                evidence_id=f"stgpt.structure.{n_flag + i}",
                unit_id=str(n_flag + i),
                correction_type="reject",
            )
        )
    return corrections


def test_should_trigger_finetuning_below_threshold() -> None:
    assert should_trigger_finetuning(_make_corrections(49, 10), threshold=50) is False


def test_should_trigger_finetuning_at_threshold() -> None:
    assert should_trigger_finetuning(_make_corrections(50, 5), threshold=50) is True


def test_should_trigger_finetuning_above_threshold() -> None:
    assert should_trigger_finetuning(_make_corrections(100, 0), threshold=50) is True


# ---------------------------------------------------------------------------
# compute_input_hash
# ---------------------------------------------------------------------------


def test_compute_input_hash_deterministic() -> None:
    inputs = {"tool": "embed_regions", "backend": "precomputed", "case": "breast_001"}
    h1 = compute_input_hash(inputs)
    h2 = compute_input_hash(inputs)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest


def test_compute_input_hash_sensitive_to_values() -> None:
    h1 = compute_input_hash({"tool": "a"})
    h2 = compute_input_hash({"tool": "b"})
    assert h1 != h2


# ---------------------------------------------------------------------------
# ExecutionPlan topological ordering
# ---------------------------------------------------------------------------


def test_execution_plan_topological_order() -> None:
    plan = ExecutionPlan(
        plan_id="test-plan",
        case_name="breast_001",
        nodes=[
            ToolNode(node_id="compare", tool_name="stgpt.runtime.compare_regions", depends_on=["embed"]),
            ToolNode(node_id="embed", tool_name="stgpt.runtime.embed_regions", depends_on=["workflow"]),
            ToolNode(node_id="workflow", tool_name="spatho.api.run_workflow", depends_on=[]),
        ],
    )
    order = plan.topological_order()
    ids = [n.node_id for n in order]
    assert ids.index("workflow") < ids.index("embed")
    assert ids.index("embed") < ids.index("compare")


def test_execution_plan_round_trip(tmp_path: Path) -> None:
    plan = ExecutionPlan(
        plan_id="plan-1",
        case_name="test_case",
        nodes=[ToolNode(node_id="workflow", tool_name="spatho.api.run_workflow")],
    )
    path = tmp_path / "plan.json"
    plan.save(path)
    loaded = ExecutionPlan.load(path)
    assert loaded.plan_id == "plan-1"
    assert loaded.schema_version == EVIDENCE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# CriticReport
# ---------------------------------------------------------------------------


def test_critic_report_round_trip(tmp_path: Path) -> None:
    report = CriticReport(
        case_name="breast_001",
        total_bundles=5,
        passed=3,
        warned=1,
        failed=1,
        flagged_for_human_review=["stgpt.case.breast_001"],
        overall_status="fail",
    )
    path = tmp_path / "critic.json"
    report.save(path)
    loaded = CriticReport.load(path)
    assert loaded.overall_status == "fail"
    assert loaded.flagged_for_human_review == ["stgpt.case.breast_001"]


# ---------------------------------------------------------------------------
# export_evidence_schema
# ---------------------------------------------------------------------------


def test_export_evidence_schema_writes_valid_json(tmp_path: Path) -> None:
    out = tmp_path / "evidence_bundle.schema.json"
    export_evidence_schema(out)
    assert out.exists()
    schema = json.loads(out.read_text(encoding="utf-8"))
    assert schema["title"] == "EvidenceBundle"
    assert "properties" in schema


# ---------------------------------------------------------------------------
# HumanReviewPolicy (schema.py)
# ---------------------------------------------------------------------------


def test_human_review_policy_defaults() -> None:
    policy = HumanReviewPolicy()
    assert policy.min_qc_status == "warning"
    assert policy.conflict_resolution == "flag_only"
    assert policy.finetuning_threshold == 50
    assert policy.required_evidence_channels == []


def test_workflow_config_embeds_human_review_policy(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    base.write_text('{"dataset_root": "/tmp"}', encoding="utf-8")
    cfg_path = tmp_path / "workflow.json"
    cfg_path.write_text(
        json.dumps(
            {
                "case_name": "policy_test",
                "study_context": "Test",
                "base_pipeline_config": str(base),
                "output_root": str(tmp_path / "out"),
                "annotation_taxonomy": "breast",
                "openai_enabled": False,
                "human_review_policy": {
                    "min_qc_status": "ok",
                    "conflict_resolution": "block",
                    "required_evidence_channels": ["stgpt"],
                    "finetuning_threshold": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    from spatho.schema import validate_workflow_config

    cfg = validate_workflow_config(cfg_path)
    assert cfg.human_review_policy.min_qc_status == "ok"
    assert cfg.human_review_policy.conflict_resolution == "block"
    assert cfg.human_review_policy.required_evidence_channels == ["stgpt"]
    assert cfg.human_review_policy.finetuning_threshold == 10
