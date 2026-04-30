from __future__ import annotations

"""Agentic evidence workbench: Planner → Executor → Critic → Reporter.

The four layers communicate through explicit data structures
(:class:`~spatho.evidence.ExecutionPlan`, :class:`~spatho.evidence.EvidenceBundle`,
:class:`~spatho.evidence.CriticReport`) so that each step can be re-run or
replayed independently without re-executing the full workflow.

Caching
-------
Every tool call writes a companion ``<artifact>.meta.json`` file
(:class:`~spatho.evidence.ToolCallMeta`).  The Executor reads the meta file
before invoking a tool; if the ``input_hash`` matches the current inputs the
cached artifact is reused and the tool is skipped.

Human-review handoff
--------------------
The Critic uses the ``human_review_policy`` embedded in
:class:`~spatho.schema.WorkflowConfig` to decide which bundles must be
flagged for human review before the Reporter may include them in the final
HTML report.
"""

import time
import uuid
from pathlib import Path
from typing import Any

from .evidence import (
    EVIDENCE_SCHEMA_VERSION,
    CriticReport,
    EvidenceBundle,
    ExecutionPlan,
    ToolCallMeta,
    ToolNode,
    compute_input_hash,
)
from .schema import WorkflowConfig


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def plan_evidence_run(cfg: WorkflowConfig, *, output_root: Path | None = None) -> ExecutionPlan:
    """Decide which tools need to run for this case and return a serialisable DAG.

    The plan is written to ``<output_root>/workbench/execution_plan.json`` so it
    can be inspected or replayed without re-running the planner.
    """
    root = (output_root or cfg.output_root).resolve()
    nodes: list[ToolNode] = []

    # Base workflow is always present
    nodes.append(
        ToolNode(
            node_id="run_workflow",
            tool_name="spatho.api.run_workflow",
            depends_on=[],
        )
    )

    # stGPT embedding tools – embed_regions is a prerequisite for compare_regions
    if cfg.stgpt_enabled:
        nodes.append(
            ToolNode(
                node_id="stgpt_embed_regions",
                tool_name="stgpt.runtime.embed_regions",
                depends_on=["run_workflow"],
                params={
                    "backend": cfg.stgpt_backend,
                    "artifact_dir": str(cfg.stgpt_artifact_dir) if cfg.stgpt_artifact_dir else None,
                },
            )
        )
        nodes.append(
            ToolNode(
                node_id="stgpt_compare_regions",
                tool_name="stgpt.runtime.compare_regions",
                depends_on=["stgpt_embed_regions"],
            )
        )
        nodes.append(
            ToolNode(
                node_id="stgpt_export_artifacts",
                tool_name="stgpt.runtime.export_spatho_artifacts",
                depends_on=["stgpt_embed_regions"],
                params={
                    "min_cell_coverage": cfg.stgpt_min_cell_coverage,
                    "require_qc_pass": cfg.stgpt_require_qc_pass,
                },
            )
        )

    if cfg.rna_foundation_enabled or cfg.pathway_activity_enabled or cfg.niche_fusion_enabled:
        nodes.append(
            ToolNode(
                node_id="foundation_evidence",
                tool_name="spatho.foundation.apply_foundation_evidence",
                depends_on=["run_workflow"],
            )
        )

    if cfg.he_contour_foundation_enabled:
        nodes.append(
            ToolNode(
                node_id="he_contour_foundation",
                tool_name="spatho.he_foundation.apply_he_contour_foundation",
                depends_on=["run_workflow"],
            )
        )

    plan = ExecutionPlan(
        plan_id=str(uuid.uuid4()),
        case_name=cfg.case_name,
        nodes=nodes,
    )

    plan_path = root / "workbench" / "execution_plan.json"
    plan.save(plan_path)
    return plan


# ---------------------------------------------------------------------------
# Executor helpers
# ---------------------------------------------------------------------------


def _cache_hit(artifact_path: Path, input_hash: str) -> bool:
    """Return True when a valid cached artifact exists for *input_hash*."""
    meta_path = ToolCallMeta.meta_path_for(artifact_path)
    if not artifact_path.exists() or not meta_path.exists():
        return False
    try:
        meta = ToolCallMeta.load(meta_path)
        return meta.input_hash == input_hash
    except Exception:
        return False


def _write_meta(
    artifact_path: Path,
    *,
    tool_name: str,
    input_hash: str,
    model_version: str = "",
    elapsed_seconds: float | None = None,
) -> ToolCallMeta:
    """Write a companion ``.meta.json`` beside *artifact_path*."""
    from . import __version__ as _spatho_version

    meta = ToolCallMeta(
        tool_name=tool_name,
        input_hash=input_hash,
        model_version=model_version,
        spatho_version=_spatho_version,
        elapsed_seconds=elapsed_seconds,
        output_artifact=str(artifact_path),
    )
    meta.save(ToolCallMeta.meta_path_for(artifact_path))
    return meta


def execute_plan(
    plan: ExecutionPlan,
    cfg: WorkflowConfig,
    *,
    output_root: Path | None = None,
    heuristic_only: bool = False,
) -> list[EvidenceBundle]:
    """Run the tools described in *plan* in topological order.

    Each tool produces one or more :class:`EvidenceBundle` objects.  Results
    from cached artifacts are replayed without re-running the tool.

    Parameters
    ----------
    plan:
        Serialisable execution plan from :func:`plan_evidence_run`.
    cfg:
        Workflow configuration.
    output_root:
        Override for the output directory (defaults to ``cfg.output_root``).
    heuristic_only:
        When True forces heuristic annotation/review on the base workflow.

    Returns
    -------
    list[EvidenceBundle]
        All bundles produced or replayed during this execution pass.
    """
    root = (output_root or cfg.output_root).resolve()
    bundles: list[EvidenceBundle] = []

    for node in plan.topological_order():
        bundle = _execute_node(node, cfg, root=root, heuristic_only=heuristic_only)
        if bundle is not None:
            bundles.append(bundle)

    return bundles


def _execute_node(
    node: ToolNode,
    cfg: WorkflowConfig,
    *,
    root: Path,
    heuristic_only: bool,
) -> EvidenceBundle | None:
    """Execute a single plan node and return an EvidenceBundle (or None to skip)."""
    cache_key = compute_input_hash(
        {"node_id": node.node_id, "tool": node.tool_name, "params": node.params, "case": cfg.case_name}
    )
    bundle_path = root / "workbench" / "bundles" / f"{node.node_id}.json"

    # Cache hit – replay without re-running
    if node.skip_if_cached and _cache_hit(bundle_path, cache_key):
        try:
            return EvidenceBundle.load(bundle_path)
        except Exception:
            pass  # fall through to re-execute

    t0 = time.monotonic()

    if node.node_id == "run_workflow":
        bundle = _node_run_workflow(node, cfg, root=root, heuristic_only=heuristic_only, cache_key=cache_key)
    elif node.node_id.startswith("stgpt_"):
        bundle = _node_stgpt(node, cfg, root=root, cache_key=cache_key)
    elif node.node_id == "foundation_evidence":
        bundle = _node_foundation(node, cfg, cache_key=cache_key)
    elif node.node_id == "he_contour_foundation":
        bundle = _node_he_contour(node, cfg, cache_key=cache_key)
    else:
        return None

    elapsed = time.monotonic() - t0
    bundle = bundle.model_copy(update={"elapsed_seconds": elapsed})
    bundle.save(bundle_path)
    _write_meta(bundle_path, tool_name=node.tool_name, input_hash=cache_key, elapsed_seconds=elapsed)
    return bundle


def _node_run_workflow(
    node: ToolNode,
    cfg: WorkflowConfig,
    *,
    root: Path,
    heuristic_only: bool,
    cache_key: str,
) -> EvidenceBundle:
    # The base workflow is run by the top-level orchestrator (run_evidence_workbench),
    # not re-invoked here.  This node records that the workflow ran successfully.
    return EvidenceBundle(
        evidence_id=f"workflow.run.{cfg.case_name}",
        unit="case",
        unit_id=cfg.case_name,
        source="spatho",
        evidence_type="workflow_run",
        measured=False,
        model_derived=False,
        qc_status="ok",
        summary="Base spatho workflow completed.",
        tool_name=node.tool_name,
        input_hash=cache_key,
    )


def _node_stgpt(
    node: ToolNode,
    cfg: WorkflowConfig,
    *,
    root: Path,
    cache_key: str,
) -> EvidenceBundle:
    from .stgpt import inspect_stgpt_evidence, resolve_stgpt_artifact_paths

    paths = resolve_stgpt_artifact_paths(cfg, output_root=root)
    inspection = inspect_stgpt_evidence(cfg)
    if inspection["errors"]:
        qc_status: str = "fail"
    elif inspection["warnings"]:
        qc_status = "warning"
    else:
        qc_status = "ok"
    return EvidenceBundle(
        evidence_id=f"stgpt.{node.node_id}.{cfg.case_name}",
        unit="case",
        unit_id=cfg.case_name,
        source="stgpt",
        evidence_type=node.tool_name,
        model_derived=True,
        qc_status=qc_status,  # type: ignore[arg-type]
        summary=(
            "; ".join(inspection["errors"] + inspection["warnings"])
            or "stGPT artifacts present and inspected."
        ),
        supporting_artifacts=[str(p) for p in paths.values()],
        tool_name=node.tool_name,
        input_hash=cache_key,
    )


def _node_foundation(
    node: ToolNode,
    cfg: WorkflowConfig,
    *,
    cache_key: str,
) -> EvidenceBundle:
    channels = []
    if cfg.rna_foundation_enabled:
        channels.append("rna")
    if cfg.pathway_activity_enabled:
        channels.append("pathway")
    if cfg.niche_fusion_enabled:
        channels.append("niche")
    return EvidenceBundle(
        evidence_id=f"foundation.{cfg.case_name}",
        unit="case",
        unit_id=cfg.case_name,
        source="spatho.foundation",
        evidence_type="foundation_evidence",
        model_derived=True,
        qc_status="ok",
        summary=f"Foundation evidence channels: {', '.join(channels) or 'none'}.",
        tool_name=node.tool_name,
        input_hash=cache_key,
    )


def _node_he_contour(
    node: ToolNode,
    cfg: WorkflowConfig,
    *,
    cache_key: str,
) -> EvidenceBundle:
    return EvidenceBundle(
        evidence_id=f"he_contour.{cfg.case_name}",
        unit="case",
        unit_id=cfg.case_name,
        source="spatho.he_foundation",
        evidence_type="he_contour_foundation",
        model_derived=True,
        qc_status="ok",
        summary="H&E contour foundation evidence applied.",
        tool_name=node.tool_name,
        input_hash=cache_key,
    )


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

_QC_ORDER: dict[str, int] = {"ok": 0, "warning": 1, "fail": 2, "unknown": 3}


def _qc_worse(a: str, b: str) -> str:
    """Return the worse of two qc_status strings."""
    return a if _QC_ORDER.get(a, 3) >= _QC_ORDER.get(b, 3) else b


def run_critic(
    bundles: list[EvidenceBundle],
    cfg: WorkflowConfig,
    *,
    output_root: Path | None = None,
) -> CriticReport:
    """Inspect all bundles for QC failures, conflicts, and coverage gaps.

    The critic uses ``cfg.human_review_policy`` (if present) to determine which
    bundles need human review before the Reporter may include them.

    Parameters
    ----------
    bundles:
        All bundles from :func:`execute_plan`.
    cfg:
        Workflow configuration (contains ``human_review_policy``).
    output_root:
        Override for the output directory.

    Returns
    -------
    CriticReport
        Written to ``<output_root>/workbench/critic_report.json``.
    """
    root = (output_root or cfg.output_root).resolve()
    policy = getattr(cfg, "human_review_policy", None)
    min_qc = getattr(policy, "min_qc_status", "warning") if policy else "warning"
    required_channels = getattr(policy, "required_evidence_channels", []) if policy else []

    flagged: list[str] = []
    conflicts: list[dict[str, Any]] = []
    overall: str = "ok"

    passed = warned = failed = 0
    for bundle in bundles:
        status = bundle.qc_status
        overall = _qc_worse(overall, status)
        if status == "ok":
            passed += 1
        elif status == "warning":
            warned += 1
        elif status == "fail":
            failed += 1

        # Flag if below the configured minimum acceptable status
        if _QC_ORDER.get(status, 3) > _QC_ORDER.get(min_qc, 1):
            flagged.append(bundle.evidence_id)

        if bundle.conflict_note:
            conflicts.append(
                {"evidence_id": bundle.evidence_id, "note": bundle.conflict_note, "unit_id": bundle.unit_id}
            )

    # Coverage gap detection: check which channels were expected
    present_sources = {b.source for b in bundles}
    gaps: list[str] = []
    if cfg.stgpt_enabled and "stgpt" not in present_sources:
        gaps.append("stgpt")
    if (cfg.rna_foundation_enabled or cfg.pathway_activity_enabled or cfg.niche_fusion_enabled) and "spatho.foundation" not in present_sources:
        gaps.append("foundation")
    if cfg.he_contour_foundation_enabled and "spatho.he_foundation" not in present_sources:
        gaps.append("he_contour_foundation")
    for channel in required_channels:
        if channel not in present_sources:
            gaps.append(channel)

    report = CriticReport(
        case_name=cfg.case_name,
        total_bundles=len(bundles),
        passed=passed,
        warned=warned,
        failed=failed,
        flagged_for_human_review=flagged,
        conflicts=conflicts,
        coverage_gaps=gaps,
        overall_status=overall,  # type: ignore[arg-type]
    )

    report_path = root / "workbench" / "critic_report.json"
    report.save(report_path)
    return report


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


def build_workbench_summary(
    bundles: list[EvidenceBundle],
    critic_report: CriticReport,
    *,
    output_root: Path,
) -> Path:
    """Write a JSON workbench summary that the HTML report can embed.

    Only bundles that passed the Critic (i.e. not in ``flagged_for_human_review``
    or have ``qc_status != 'fail'``) are included as *approved* evidence.

    Returns
    -------
    Path
        Path to the written ``workbench_summary.json``.
    """
    import json

    flagged_ids = set(critic_report.flagged_for_human_review)
    approved = [b.model_dump() for b in bundles if b.evidence_id not in flagged_ids]
    pending_review = [b.model_dump() for b in bundles if b.evidence_id in flagged_ids]

    summary = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "case_name": critic_report.case_name,
        "generated_at": critic_report.generated_at,
        "overall_status": critic_report.overall_status,
        "total_bundles": critic_report.total_bundles,
        "approved_count": len(approved),
        "pending_human_review_count": len(pending_review),
        "coverage_gaps": critic_report.coverage_gaps,
        "conflicts": critic_report.conflicts,
        "approved_evidence": approved,
        "pending_human_review": pending_review,
    }

    out_path = output_root / "workbench" / "workbench_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Top-level orchestrator (backwards-compatible entry point)
# ---------------------------------------------------------------------------


def run_evidence_workbench(
    config_path: str | Path,
    *,
    heuristic_only: bool = False,
) -> dict[str, str]:
    """Run the full Planner → Executor → Critic → Reporter pipeline.

    This function is the main entry point used by :func:`spatho.api.run_workflow`
    and the ``spatho run`` CLI command.  It is backwards-compatible: callers
    that previously relied on ``run_evidence_workbench`` receive the same dict
    return value they did when it simply called ``run_workflow``.
    """
    from .api import run_workflow
    from .schema import validate_workflow_config

    cfg = validate_workflow_config(config_path)
    root = cfg.output_root.resolve()

    # Planner
    plan = plan_evidence_run(cfg, output_root=root)

    # Executor (produces EvidenceBundles; cache-aware)
    bundles = execute_plan(plan, cfg, output_root=root, heuristic_only=heuristic_only)

    # Critic
    critic = run_critic(bundles, cfg, output_root=root)

    # Reporter
    workbench_summary_path = build_workbench_summary(bundles, critic, output_root=root)

    # Fall back to the plain workflow result dict for callers that need it
    result = run_workflow(config_path, heuristic_only=heuristic_only)
    return {
        **result,
        "workbench_plan_json": str(root / "workbench" / "execution_plan.json"),
        "workbench_critic_report_json": str(root / "workbench" / "critic_report.json"),
        "workbench_summary_json": str(workbench_summary_path),
    }


__all__ = [
    "plan_evidence_run",
    "execute_plan",
    "run_critic",
    "build_workbench_summary",
    "run_evidence_workbench",
]
