from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import hashlib
import json

from .reports import build_evidence_report_section
from .schema import WorkflowConfig


def stgpt_evidence_requested(cfg: WorkflowConfig) -> bool:
    return bool(cfg.stgpt_enabled)


def inspect_stgpt_evidence(cfg: WorkflowConfig, *, allow_local_pending: bool = True) -> dict[str, Any]:
    """Inspect stGPT artifact readiness without importing stGPT for precomputed mode."""
    errors: list[str] = []
    warnings: list[str] = []
    paths = resolve_stgpt_artifact_paths(cfg)

    if not cfg.stgpt_enabled:
        return {"enabled": False, "ready": True, "errors": errors, "warnings": warnings, "paths": paths}

    if cfg.stgpt_backend == "local_stgpt":
        if cfg.stgpt_model_path is None or not cfg.stgpt_model_path.exists():
            errors.append("stGPT local backend is enabled but stgpt_model_path is missing or does not exist.")
        if cfg.stgpt_config_path is None or not cfg.stgpt_config_path.exists():
            errors.append("stGPT local backend is enabled but stgpt_config_path is missing or does not exist.")
        if allow_local_pending:
            if not errors:
                warnings.append("stGPT local backend artifacts will be generated during spatho run.")
            return {"enabled": True, "ready": not errors, "errors": errors, "warnings": warnings, "paths": paths}

    for key in ("cell_embeddings", "structure_summary", "qc_report"):
        path = paths.get(key)
        if path is None or not path.exists():
            errors.append(f"stGPT evidence is enabled but {key} artifact is missing: {path}")

    qc_report = _load_qc_report(paths.get("qc_report"))
    fatal_errors = _qc_fatal_errors(qc_report)
    qc_warnings = _qc_warnings(qc_report)
    coverage = _qc_coverage(qc_report)
    if fatal_errors and cfg.stgpt_require_qc_pass:
        errors.append("stGPT QC reported fatal errors: " + "; ".join(fatal_errors))
    elif fatal_errors:
        warnings.append("stGPT QC fatal errors were present but stgpt_require_qc_pass is false.")
    warnings.extend(qc_warnings)
    if coverage is not None and coverage < cfg.stgpt_min_cell_coverage:
        warnings.append(
            f"stGPT image/cell coverage {coverage:.3f} is below configured minimum {cfg.stgpt_min_cell_coverage:.3f}."
        )

    return {
        "enabled": True,
        "ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "paths": paths,
        "qc_report": qc_report,
        "cell_coverage": coverage,
    }


def assert_stgpt_ready(cfg: WorkflowConfig) -> None:
    inspection = inspect_stgpt_evidence(cfg, allow_local_pending=True)
    if inspection["errors"]:
        raise ValueError("stGPT evidence guardrail failed: " + " ".join(inspection["errors"]))


def prepare_stgpt_evidence(cfg: WorkflowConfig) -> dict[str, Path]:
    """Prepare or validate stGPT evidence before biological review starts."""
    if cfg.stgpt_backend == "local_stgpt":
        paths = _run_local_stgpt(cfg, output_root=cfg.output_root)
        inspection = inspect_stgpt_evidence_for_paths(cfg, paths)
        if inspection["errors"]:
            raise ValueError("stGPT evidence guardrail failed: " + " ".join(inspection["errors"]))
        return paths
    assert_stgpt_ready(cfg)
    return resolve_stgpt_artifact_paths(cfg)


def apply_stgpt_evidence(cfg: WorkflowConfig, workflow_result: dict[str, str]) -> dict[str, str]:
    workflow_summary_path = Path(workflow_result["workflow_summary_json"]).resolve()
    workflow_summary = json.loads(workflow_summary_path.read_text(encoding="utf-8"))
    output_root = Path(workflow_summary["output_root"]).resolve()
    foundation_dir = output_root / "foundation"
    foundation_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_stgpt_artifact_paths(cfg, output_root=output_root)
    if cfg.stgpt_backend == "local_stgpt":
        if not all(paths[key].exists() for key in ("cell_embeddings", "structure_summary", "qc_report")):
            paths = _run_local_stgpt(cfg, output_root=output_root)

    inspection = inspect_stgpt_evidence_for_paths(cfg, paths)
    if inspection["errors"]:
        raise ValueError("stGPT evidence guardrail failed: " + " ".join(inspection["errors"]))

    summary_rows = _build_stgpt_summary_rows(paths, inspection, cfg)
    summary_csv = foundation_dir / "stgpt_evidence_summary.csv"
    summary_json = foundation_dir / "stgpt_evidence_summary.json"
    _write_csv(summary_csv, summary_rows)
    summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    outputs = {
        "stgpt_cell_embeddings_parquet": str(paths["cell_embeddings"]),
        "stgpt_structure_embedding_summary_csv": str(paths["structure_summary"]),
        "stgpt_qc_report_json": str(paths["qc_report"]),
        "stgpt_evidence_summary_json": str(summary_json),
        "stgpt_evidence_summary_csv": str(summary_csv),
    }
    foundation_outputs = dict(workflow_summary.get("foundation_outputs", {}))
    foundation_outputs.update(outputs)
    workflow_summary["foundation_outputs"] = foundation_outputs
    workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _insert_stgpt_report_section(
        workflow_summary.get("pathology_outputs", {}).get("report_html"),
        summary_rows=summary_rows,
        warnings=inspection["warnings"],
        outputs=outputs,
    )
    return {
        **workflow_result,
        "stgpt_evidence_summary_csv": str(summary_csv),
        "stgpt_evidence_summary_json": str(summary_json),
        "workflow_summary_json": str(workflow_summary_path),
    }


def resolve_stgpt_artifact_paths(cfg: WorkflowConfig, *, output_root: Path | None = None) -> dict[str, Path]:
    artifact_dir = cfg.stgpt_artifact_dir
    if artifact_dir is None and output_root is not None:
        artifact_dir = output_root / "foundation" / "stgpt_runtime"
    if artifact_dir is None:
        artifact_dir = cfg.output_root / "stgpt"
    artifact_dir = artifact_dir.resolve()
    return {
        "cell_embeddings": (cfg.stgpt_cell_embeddings_path or artifact_dir / "cell_embeddings.parquet").resolve(),
        "structure_summary": (
            cfg.stgpt_structure_summary_path
            or _first_existing(
                artifact_dir,
                ("structure_embedding_summary.csv", "structure_summary.csv", "structure_summary.parquet"),
            )
        ).resolve(),
        "qc_report": (cfg.stgpt_qc_report_path or artifact_dir / "qc_report.json").resolve(),
    }


def inspect_stgpt_evidence_for_paths(cfg: WorkflowConfig, paths: dict[str, Path]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    for key in ("cell_embeddings", "structure_summary", "qc_report"):
        path = paths.get(key)
        if path is None or not path.exists():
            errors.append(f"stGPT evidence is enabled but {key} artifact is missing: {path}")
    qc_report = _load_qc_report(paths.get("qc_report"))
    fatal_errors = _qc_fatal_errors(qc_report)
    if fatal_errors and cfg.stgpt_require_qc_pass:
        errors.append("stGPT QC reported fatal errors: " + "; ".join(fatal_errors))
    elif fatal_errors:
        warnings.append("stGPT QC fatal errors were present but stgpt_require_qc_pass is false.")
    warnings.extend(_qc_warnings(qc_report))
    coverage = _qc_coverage(qc_report)
    if coverage is not None and coverage < cfg.stgpt_min_cell_coverage:
        warnings.append(
            f"stGPT image/cell coverage {coverage:.3f} is below configured minimum {cfg.stgpt_min_cell_coverage:.3f}."
        )
    return {"ready": not errors, "errors": errors, "warnings": warnings, "qc_report": qc_report, "cell_coverage": coverage}


def _run_local_stgpt(cfg: WorkflowConfig, *, output_root: Path) -> dict[str, Path]:
    if cfg.stgpt_model_path is None or cfg.stgpt_config_path is None:
        raise ValueError("local_stgpt backend requires stgpt_model_path and stgpt_config_path.")
    try:
        from stgpt.runtime import export_spatho_artifacts
    except ImportError as exc:
        raise ImportError("local_stgpt backend requires the stgpt package to be installed.") from exc

    out = cfg.stgpt_artifact_dir or output_root / "foundation" / "stgpt_runtime"
    result = export_spatho_artifacts(
        config=cfg.stgpt_config_path,
        checkpoint=cfg.stgpt_model_path,
        output_dir=out,
    )
    return {
        "cell_embeddings": Path(result["cell_embeddings"]).resolve(),
        "structure_summary": Path(
            result.get("structure_embedding_summary") or result.get("structure_summary")
        ).resolve(),
        "qc_report": Path(result["qc_report"]).resolve(),
    }


def _first_existing(base: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = base / name
        if path.exists():
            return path
    return base / names[0]


def _load_qc_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"fatal_errors": ["stGPT QC report is not valid JSON."]}
    return payload if isinstance(payload, dict) else {"fatal_errors": ["stGPT QC report is not a JSON object."]}


def _qc_fatal_errors(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("fatal_errors") or payload.get("errors") or []
    if payload.get("status") == "fail" and not raw:
        raw = ["QC status is fail."]
    return [str(item) for item in raw if str(item).strip()]


def _qc_warnings(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("warnings") or []
    return [str(item) for item in raw if str(item).strip()]


def _qc_coverage(payload: dict[str, Any]) -> float | None:
    for key in ("image_coverage", "cell_coverage", "patch_cell_coverage", "patch_cell_coverage_fraction"):
        if key in payload:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                return None
    total = payload.get("n_cells_total")
    covered = payload.get("n_cells_with_image")
    try:
        if total:
            return float(covered) / float(total)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def _sha256(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stgpt_artifact_ids() -> list[str]:
    return [
        "foundation.stgpt_structure_embedding_summary_csv",
        "foundation.stgpt_qc_report_json",
        "foundation.stgpt_evidence_summary_csv",
    ]


def _build_stgpt_summary_rows(
    paths: dict[str, Path],
    inspection: dict[str, Any],
    cfg: WorkflowConfig,
) -> list[dict[str, Any]]:
    structure_path = paths["structure_summary"]
    rows = _read_csv_rows(structure_path) if structure_path.suffix.lower() == ".csv" else []
    qc_status = "warning" if inspection["warnings"] else "ok"
    checkpoint_sha256 = _sha256(cfg.stgpt_model_path)
    artifact_ids = _stgpt_artifact_ids()
    artifact_paths = [str(paths["structure_summary"]), str(paths["qc_report"])]
    if rows:
        summary = []
        for row in rows:
            label = row.get("structure_label") or row.get("structure_id") or row.get("structure") or "unknown"
            summary.append(
                {
                    "evidence_id": f"stgpt.structure.{label}",
                    "evidence_type": "stgpt_structure_embedding",
                    "structure_label": label,
                    "n_cells": row.get("n_cells", ""),
                    "qc_status": qc_status,
                    "qc_flag": "model-derived",
                    "human_review_state": "pending",
                    "model_derived": True,
                    "measured_expression": False,
                    "artifact_ids": artifact_ids,
                    "artifact_paths": artifact_paths,
                    "checkpoint_path": str(cfg.stgpt_model_path) if cfg.stgpt_model_path else "",
                    "checkpoint_sha256": checkpoint_sha256,
                    "claim_guardrail": "Model-derived evidence; do not report as measured expression or diagnosis.",
                    "interpretation": (
                        "stGPT embedding centroid is available for human-reviewed structure evidence."
                    ),
                    "source": str(structure_path),
                }
            )
        return summary
    return [
        {
            "evidence_id": f"stgpt.case.{cfg.case_name}",
            "evidence_type": "stgpt_case_embedding",
            "structure_label": "case",
            "n_cells": "",
            "qc_status": qc_status,
            "qc_flag": "model-derived",
            "human_review_state": "pending",
            "model_derived": True,
            "measured_expression": False,
            "artifact_ids": artifact_ids,
            "artifact_paths": artifact_paths,
            "checkpoint_path": str(cfg.stgpt_model_path) if cfg.stgpt_model_path else "",
            "checkpoint_sha256": checkpoint_sha256,
            "claim_guardrail": "Model-derived evidence; do not report as measured expression or diagnosis.",
            "interpretation": "stGPT embeddings are available; structure summary could not be parsed as CSV.",
            "source": str(structure_path),
        }
    ]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def _insert_stgpt_report_section(
    report_path: str | Path | None,
    *,
    summary_rows: list[dict[str, Any]],
    warnings: list[str],
    outputs: dict[str, str],
) -> None:
    if not report_path:
        return
    path = Path(report_path)
    if not path.exists():
        return
    html = path.read_text(encoding="utf-8")
    start = "<!-- spatho-stgpt-evidence:start -->"
    end = "<!-- spatho-stgpt-evidence:end -->"
    section = build_evidence_report_section(summary_rows=summary_rows, warnings=warnings, outputs=outputs)
    if start in html and end in html:
        before = html.split(start, 1)[0]
        after = html.split(end, 1)[1]
        html = before + section + after
    elif "</main>" in html:
        html = html.replace("</main>", section + "\n</main>")
    else:
        html += "\n" + section
    path.write_text(html, encoding="utf-8")


__all__ = [
    "apply_stgpt_evidence",
    "assert_stgpt_ready",
    "inspect_stgpt_evidence",
    "prepare_stgpt_evidence",
    "resolve_stgpt_artifact_paths",
    "stgpt_evidence_requested",
]
