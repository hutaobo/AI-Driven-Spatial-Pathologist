from __future__ import annotations

from .api import (
    build_manifest,
    init_workflow,
    list_available_organ_packs,
    run_workflow,
    workflow_doctor_report,
    write_schema,
    write_xenium_alignment_fixtures,
)
from .agentic import DEMO_QUESTION, build_agentic_spatial_pathologist_demo
from .evidence import (
    CorrectionBundle,
    CriticReport,
    EvidenceBundle,
    ExecutionPlan,
    ToolCallMeta,
    compute_input_hash,
    export_evidence_schema,
    should_trigger_finetuning,
)
from .schema import HumanReviewPolicy
from .reports import build_evidence_report_section
from .workbench import (
    build_workbench_summary,
    execute_plan,
    plan_evidence_run,
    run_critic,
    run_evidence_workbench,
)

__all__ = [
    "build_manifest",
    "run_workflow",
    "workflow_doctor_report",
    "init_workflow",
    "list_available_organ_packs",
    "write_schema",
    "write_xenium_alignment_fixtures",
    "DEMO_QUESTION",
    "build_agentic_spatial_pathologist_demo",
    # Evidence schema
    "EvidenceBundle",
    "ToolCallMeta",
    "CorrectionBundle",
    "ExecutionPlan",
    "CriticReport",
    "HumanReviewPolicy",
    "compute_input_hash",
    "export_evidence_schema",
    "should_trigger_finetuning",
    # Workbench pipeline
    "plan_evidence_run",
    "execute_plan",
    "run_critic",
    "build_workbench_summary",
    "run_evidence_workbench",
    # Reports
    "build_evidence_report_section",
]

__version__ = "0.1.3"
