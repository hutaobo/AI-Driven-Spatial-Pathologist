"""Tests for the Planner / Executor / Critic / Reporter workbench pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from spatho.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    EvidenceBundle,
    ExecutionPlan,
)
from spatho.schema import validate_workflow_config
from spatho.workbench import (
    build_workbench_summary,
    execute_plan,
    plan_evidence_run,
    run_critic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch_json(path: Path, payload: object = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({} if payload is None else payload), encoding="utf-8")


def _basic_config(tmp_path: Path, **extra: object) -> Path:
    base = tmp_path / "base.json"
    _touch_json(base, {"dataset_root": str(tmp_path)})
    cfg_path = tmp_path / "workflow.json"
    _touch_json(
        cfg_path,
        {
            "case_name": "wb_test",
            "study_context": "Workbench unit tests",
            "base_pipeline_config": str(base),
            "output_root": str(tmp_path / "out"),
            "annotation_taxonomy": "breast",
            "openai_enabled": False,
            **extra,
        },
    )
    return cfg_path


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def test_planner_writes_execution_plan(tmp_path: Path) -> None:
    cfg = validate_workflow_config(_basic_config(tmp_path))
    plan = plan_evidence_run(cfg, output_root=tmp_path / "out")

    plan_path = tmp_path / "out" / "workbench" / "execution_plan.json"
    assert plan_path.exists()
    loaded = ExecutionPlan.load(plan_path)
    assert loaded.plan_id == plan.plan_id
    assert loaded.case_name == "wb_test"
    assert loaded.schema_version == EVIDENCE_SCHEMA_VERSION


def test_planner_includes_stgpt_nodes_when_enabled(tmp_path: Path) -> None:
    cfg = validate_workflow_config(
        _basic_config(tmp_path, stgpt_enabled=True, stgpt_artifact_dir=str(tmp_path / "stgpt"))
    )
    plan = plan_evidence_run(cfg, output_root=tmp_path / "out")
    node_ids = [n.node_id for n in plan.nodes]
    assert "stgpt_embed_regions" in node_ids
    assert "stgpt_compare_regions" in node_ids
    assert "stgpt_export_artifacts" in node_ids


def test_planner_excludes_stgpt_nodes_when_disabled(tmp_path: Path) -> None:
    cfg = validate_workflow_config(_basic_config(tmp_path))
    plan = plan_evidence_run(cfg, output_root=tmp_path / "out")
    node_ids = [n.node_id for n in plan.nodes]
    assert "stgpt_embed_regions" not in node_ids


def test_planner_includes_pyxenium_mtm_node_when_enabled(tmp_path: Path) -> None:
    summary = tmp_path / "mtm" / "morphomolecular_summary.csv"
    summary.parent.mkdir()
    summary.write_text("structure_label,score\nstroma,0.8\n", encoding="utf-8")
    cfg = validate_workflow_config(
        _basic_config(
            tmp_path,
            pyxenium_mtm_enabled=True,
            pyxenium_mtm_summary_path=str(summary),
        )
    )
    plan = plan_evidence_run(cfg, output_root=tmp_path / "out")
    node_ids = [node.node_id for node in plan.nodes]
    assert "pyxenium_mtm_evidence" in node_ids


def test_executor_reads_pyxenium_mtm_artifacts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "mtm"
    artifact_dir.mkdir()
    summary = artifact_dir / "morphomolecular_summary.csv"
    qc = artifact_dir / "qc_report.json"
    summary.write_text("structure_label,score\nstroma,0.8\n", encoding="utf-8")
    qc.write_text(json.dumps({"status": "pass", "warnings": []}), encoding="utf-8")
    cfg = validate_workflow_config(
        _basic_config(
            tmp_path,
            pyxenium_mtm_enabled=True,
            pyxenium_mtm_artifact_dir=str(artifact_dir),
            pyxenium_mtm_summary_path=str(summary),
            pyxenium_mtm_qc_report_path=str(qc),
        )
    )
    plan = plan_evidence_run(cfg, output_root=tmp_path / "out")

    bundles = execute_plan(plan, cfg, output_root=tmp_path / "out", heuristic_only=True)
    mtm = next(bundle for bundle in bundles if bundle.source == "pyXenium.mtm")

    assert mtm.qc_status == "ok"
    assert str(summary.resolve()) in mtm.supporting_artifacts
    assert mtm.artifact_hashes


def test_planner_stgpt_compare_depends_on_embed(tmp_path: Path) -> None:
    cfg = validate_workflow_config(
        _basic_config(tmp_path, stgpt_enabled=True, stgpt_artifact_dir=str(tmp_path / "stgpt"))
    )
    plan = plan_evidence_run(cfg, output_root=tmp_path / "out")
    compare_node = next(n for n in plan.nodes if n.node_id == "stgpt_compare_regions")
    assert "stgpt_embed_regions" in compare_node.depends_on


def test_planner_topological_order_respects_deps(tmp_path: Path) -> None:
    cfg = validate_workflow_config(
        _basic_config(tmp_path, stgpt_enabled=True, stgpt_artifact_dir=str(tmp_path / "stgpt"))
    )
    plan = plan_evidence_run(cfg, output_root=tmp_path / "out")
    order = [n.node_id for n in plan.topological_order()]
    assert order.index("run_workflow") < order.index("stgpt_embed_regions")
    assert order.index("stgpt_embed_regions") < order.index("stgpt_compare_regions")


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


def _make_bundles(*statuses: str) -> list[EvidenceBundle]:
    bundles = []
    for i, status in enumerate(statuses):
        bundles.append(
            EvidenceBundle(
                evidence_id=f"test.bundle.{i}",
                unit="case",
                unit_id="wb_test",
                source="test_tool",
                evidence_type="synthetic",
                qc_status=status,  # type: ignore[arg-type]
            )
        )
    return bundles


def test_critic_writes_report(tmp_path: Path) -> None:
    cfg = validate_workflow_config(_basic_config(tmp_path))
    bundles = _make_bundles("ok", "ok", "warning")
    report = run_critic(bundles, cfg, output_root=tmp_path / "out")

    report_path = tmp_path / "out" / "workbench" / "critic_report.json"
    assert report_path.exists()
    assert report.passed == 2
    assert report.warned == 1
    assert report.failed == 0
    assert report.overall_status == "warning"


def test_critic_flags_fail_bundles(tmp_path: Path) -> None:
    cfg = validate_workflow_config(_basic_config(tmp_path))
    bundles = _make_bundles("ok", "fail")
    report = run_critic(bundles, cfg, output_root=tmp_path / "out")
    assert "test.bundle.1" in report.flagged_for_human_review


def test_critic_detects_stgpt_coverage_gap(tmp_path: Path) -> None:
    cfg = validate_workflow_config(
        _basic_config(tmp_path, stgpt_enabled=True, stgpt_artifact_dir=str(tmp_path / "missing"))
    )
    # No stgpt bundle in list → coverage gap
    bundles = _make_bundles("ok")
    report = run_critic(bundles, cfg, output_root=tmp_path / "out")
    assert "stgpt" in report.coverage_gaps


def test_critic_detects_pyxenium_mtm_coverage_gap(tmp_path: Path) -> None:
    cfg = validate_workflow_config(_basic_config(tmp_path, pyxenium_mtm_enabled=True))
    bundles = _make_bundles("ok")
    report = run_critic(bundles, cfg, output_root=tmp_path / "out")
    assert "pyXenium.mtm" in report.coverage_gaps


def test_critic_uses_human_review_policy_min_qc_ok(tmp_path: Path) -> None:
    """With min_qc_status='ok', even 'warning' bundles are flagged."""
    cfg = validate_workflow_config(
        _basic_config(
            tmp_path,
            human_review_policy={"min_qc_status": "ok", "conflict_resolution": "flag_only"},
        )
    )
    bundles = _make_bundles("ok", "warning")
    report = run_critic(bundles, cfg, output_root=tmp_path / "out")
    # 'warning' is worse than 'ok' → should be flagged
    assert "test.bundle.1" in report.flagged_for_human_review


def test_critic_uses_human_review_policy_required_channels(tmp_path: Path) -> None:
    cfg = validate_workflow_config(
        _basic_config(
            tmp_path,
            human_review_policy={"required_evidence_channels": ["plip"]},
        )
    )
    bundles = _make_bundles("ok")  # source is "test_tool", not "plip"
    report = run_critic(bundles, cfg, output_root=tmp_path / "out")
    assert "plip" in report.coverage_gaps


def test_critic_report_schema_version(tmp_path: Path) -> None:
    cfg = validate_workflow_config(_basic_config(tmp_path))
    run_critic([], cfg, output_root=tmp_path / "out")
    payload = json.loads((tmp_path / "out" / "workbench" / "critic_report.json").read_text())
    assert payload["schema_version"] == EVIDENCE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Reporter / build_workbench_summary
# ---------------------------------------------------------------------------


def test_build_workbench_summary_splits_approved_and_pending(tmp_path: Path) -> None:
    from spatho.evidence import CriticReport

    bundles = [
        EvidenceBundle(
            evidence_id="ok.bundle",
            unit="case",
            unit_id="c",
            source="s",
            evidence_type="t",
            qc_status="ok",
        ),
        EvidenceBundle(
            evidence_id="fail.bundle",
            unit="case",
            unit_id="c",
            source="s",
            evidence_type="t",
            qc_status="fail",
        ),
    ]
    critic = CriticReport(
        case_name="wb_test",
        total_bundles=2,
        passed=1,
        warned=0,
        failed=1,
        flagged_for_human_review=["fail.bundle"],
        overall_status="fail",
    )
    out = tmp_path / "out"
    summary_path = build_workbench_summary(bundles, critic, output_root=out)
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["approved_count"] == 1
    assert summary["pending_human_review_count"] == 1
    assert summary["schema_version"] == EVIDENCE_SCHEMA_VERSION
    approved_ids = [b["evidence_id"] for b in summary["approved_evidence"]]
    pending_ids = [b["evidence_id"] for b in summary["pending_human_review"]]
    assert "ok.bundle" in approved_ids
    assert "fail.bundle" in pending_ids


# ---------------------------------------------------------------------------
# doctor: schema version compat check
# ---------------------------------------------------------------------------


def test_doctor_reports_schema_version(tmp_path: Path) -> None:
    from spatho.api import workflow_doctor_report

    cfg_path = _basic_config(tmp_path)
    report = workflow_doctor_report(cfg_path)
    assert report["evidence_schema_version"] == EVIDENCE_SCHEMA_VERSION


def test_doctor_detects_stale_execution_plan(tmp_path: Path) -> None:
    from spatho.api import workflow_doctor_report
    from spatho.schema import validate_workflow_config

    cfg_path = _basic_config(tmp_path)
    cfg = validate_workflow_config(cfg_path)
    # Write an execution_plan.json with an old schema version
    stale_plan = cfg.output_root / "workbench" / "execution_plan.json"
    stale_plan.parent.mkdir(parents=True, exist_ok=True)
    stale_plan.write_text(json.dumps({"schema_version": "0.0.1", "plan_id": "old"}), encoding="utf-8")

    report = workflow_doctor_report(cfg_path)
    assert any("execution_plan.json" in issue for issue in report["issues"])
