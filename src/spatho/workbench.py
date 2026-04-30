from __future__ import annotations

from pathlib import Path


def run_evidence_workbench(config_path: str | Path, *, heuristic_only: bool = False) -> dict[str, str]:
    """Run the agentic evidence workbench over the existing spatho workflow."""
    from .api import run_workflow

    return run_workflow(config_path, heuristic_only=heuristic_only)


__all__ = ["run_evidence_workbench"]
