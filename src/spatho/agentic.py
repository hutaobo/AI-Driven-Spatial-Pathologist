from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEMO_QUESTION = (
    "Which H&E-defined structures in this Xenium case show reproducible "
    "morpho-molecular programs, and do those findings pass QC?"
)

KNOWN_STGPT_ARTIFACTS = (
    "evidence_manifest.json",
    "contour_evidence_chains.jsonl",
    "region_qc_report.json",
    "qc_report.json",
    "region_embeddings.parquet",
    "region_molecular_summary.parquet",
    "region_image_manifest.json",
    "prototype_assignments.parquet",
    "structure_summary.parquet",
    "structure_embedding_summary.csv",
    "structure_embedding_summary.json",
)


def build_agentic_spatial_pathologist_demo(
    *,
    stgpt_evidence_dir: str | Path,
    output_dir: str | Path,
    case_name: str,
    metrics_path: str | Path | None = None,
    checkpoint_card_path: str | Path | None = None,
    pyxenium_summary_path: str | Path | None = None,
    question: str = DEMO_QUESTION,
    max_records: int = 100,
) -> dict[str, str]:
    """Build an artifact-first Agentic Spatial Pathologist v0.1 demo bundle.

    This reads existing stGPT/pyXenium artifacts only. It does not import stGPT,
    launch inference, or package raw image/Zarr matrices.
    """
    evidence_dir = Path(stgpt_evidence_dir).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    qc_payload = _load_first_json(evidence_dir, ("region_qc_report.json", "qc_report.json"))
    fatal_errors = _fatal_errors(qc_payload)
    warnings = _warnings(qc_payload)
    qc_status = "fail" if fatal_errors else ("warning" if warnings else "ok")

    metrics = _load_optional_json(metrics_path)
    checkpoint_card = _load_optional_json(checkpoint_card_path)
    pyxenium_summary = _load_optional_text(pyxenium_summary_path)
    artifacts = _collect_artifacts(
        evidence_dir=evidence_dir,
        metrics_path=metrics_path,
        checkpoint_card_path=checkpoint_card_path,
        pyxenium_summary_path=pyxenium_summary_path,
    )
    artifact_by_path = {record["path"]: record for record in artifacts}
    records = _read_evidence_records(evidence_dir / "contour_evidence_chains.jsonl", max_records=max_records)

    claims = _build_claims(
        case_name=case_name,
        records=records,
        qc_status=qc_status,
        fatal_errors=fatal_errors,
        warnings=warnings,
        artifact_by_path=artifact_by_path,
        checkpoint_hash=_checkpoint_hash(records, checkpoint_card),
    )
    structure_rows = _structure_rows_from_claims(claims)
    guardrail = _guardrail_payload(qc_status=qc_status, fatal_errors=fatal_errors, warnings=warnings)
    review_rows = _human_review_rows(claims, guardrail)

    manifest_payload = {
        "schema_version": "agentic_spatial_pathologist_v0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_name": case_name,
        "question": question,
        "stgpt_evidence_dir": str(evidence_dir),
        "artifact_policy": "Pointer artifacts only; raw Zarr, OME-TIF, H&E PNG, and raw matrices are not bundled.",
        "artifacts": artifacts,
        "guardrail": guardrail,
    }
    report_payload = {
        "schema_version": "agentic_spatial_pathologist_v0.1",
        "case_name": case_name,
        "question": question,
        "qc_status": qc_status,
        "conclusion_allowed": not fatal_errors,
        "claims": claims,
        "metrics_summary": _metrics_summary(metrics),
        "pyxenium_summary_present": pyxenium_summary is not None,
    }

    manifest_json = out / "artifact_manifest.json"
    report_json = out / "agentic_spatial_pathologist_report.json"
    report_md = out / "agentic_spatial_pathologist_report.md"
    table_csv = out / "structure_level_evidence_table.csv"
    guardrail_md = out / "failure_guardrail_section.md"
    checklist_md = out / "human_review_checklist.md"

    manifest_json.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_json.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(table_csv, structure_rows)
    guardrail_md.write_text(_guardrail_markdown(guardrail), encoding="utf-8")
    checklist_md.write_text(_review_markdown(review_rows), encoding="utf-8")
    report_md.write_text(
        _report_markdown(
            case_name=case_name,
            question=question,
            guardrail=guardrail,
            claims=claims,
            metrics_summary=report_payload["metrics_summary"],
            pyxenium_summary=pyxenium_summary,
        ),
        encoding="utf-8",
    )

    return {
        "report_json": str(report_json),
        "report_md": str(report_md),
        "artifact_manifest_json": str(manifest_json),
        "structure_level_evidence_table_csv": str(table_csv),
        "failure_guardrail_section_md": str(guardrail_md),
        "human_review_checklist_md": str(checklist_md),
    }


def _collect_artifacts(
    *,
    evidence_dir: Path,
    metrics_path: str | Path | None,
    checkpoint_card_path: str | Path | None,
    pyxenium_summary_path: str | Path | None,
) -> list[dict[str, Any]]:
    paths = [evidence_dir / name for name in KNOWN_STGPT_ARTIFACTS]
    for optional in (metrics_path, checkpoint_card_path, pyxenium_summary_path):
        if optional is not None:
            paths.append(Path(optional))
    seen: set[Path] = set()
    records: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        exists = resolved.exists()
        size = resolved.stat().st_size if exists and resolved.is_file() else None
        records.append(
            {
                "id": f"artifact.{resolved.name}",
                "path": str(resolved),
                "exists": exists,
                "size_bytes": size,
                "sha256": _sha256_if_small(resolved),
            }
        )
    return records


def _read_evidence_records(path: Path, *, max_records: int) -> list[dict[str, Any]]:
    if not path.exists() or max_records <= 0:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= max_records:
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _build_claims(
    *,
    case_name: str,
    records: list[dict[str, Any]],
    qc_status: str,
    fatal_errors: list[str],
    warnings: list[str],
    artifact_by_path: dict[str, dict[str, Any]],
    checkpoint_hash: str,
) -> list[dict[str, Any]]:
    if fatal_errors:
        return [
            {
                "claim_id": f"claim.blocked.{case_name}",
                "claim_text": "Biological conclusions are blocked because fatal QC errors are present.",
                "evidence_ids": [],
                "qc_status": "fail",
                "artifact_ids": [],
                "artifact_paths": [],
                "checkpoint_hash": checkpoint_hash,
                "human_review_state": "pending",
                "model_derived": True,
                "measured_expression": False,
                "cautionary": True,
            }
        ]

    claims: list[dict[str, Any]] = []
    for idx, record in enumerate(records[:20]):
        evidence_id = str(record.get("evidence_id") or f"stgpt.region.{idx}")
        prototype_id = _prototype_id(record)
        artifact_paths = _artifact_paths(record)
        artifact_ids = [
            artifact_by_path.get(str(Path(path).resolve()), {}).get("id", f"artifact.{Path(path).name}")
            for path in artifact_paths
        ]
        claims.append(
            {
                "claim_id": f"claim.{evidence_id}",
                "claim_text": (
                    f"Region-level stGPT evidence for {case_name} maps to prototype {prototype_id}. "
                    "This is model-derived morpho-molecular evidence, not measured expression."
                ),
                "evidence_ids": [evidence_id],
                "qc_status": qc_status,
                "artifact_ids": artifact_ids,
                "artifact_paths": artifact_paths,
                "checkpoint_hash": checkpoint_hash,
                "human_review_state": "pending",
                "model_derived": True,
                "measured_expression": False,
                "cautionary": bool(warnings or qc_status == "warning"),
            }
        )
    if not claims:
        claims.append(
            {
                "claim_id": f"claim.no_records.{case_name}",
                "claim_text": "No sampled stGPT evidence-chain records were available for biological interpretation.",
                "evidence_ids": [],
                "qc_status": qc_status,
                "artifact_ids": [],
                "artifact_paths": [],
                "checkpoint_hash": checkpoint_hash,
                "human_review_state": "pending",
                "model_derived": True,
                "measured_expression": False,
                "cautionary": True,
            }
        )
    return claims


def _artifact_paths(record: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for section in ("measured_evidence", "model_derived_evidence"):
        payload = record.get(section)
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, dict) and value.get("artifact"):
                    paths.append(str(value["artifact"]))
    return sorted(set(paths))


def _prototype_id(record: dict[str, Any]) -> str:
    model = record.get("model_derived_evidence")
    if isinstance(model, dict):
        proto = model.get("prototype_ref") or model.get("prototype")
        if isinstance(proto, dict):
            return str(proto.get("prototype_id", proto.get("id", "unknown")))
    return "unknown"


def _checkpoint_hash(records: list[dict[str, Any]], checkpoint_card: dict[str, Any] | None) -> str:
    for record in records:
        provenance = record.get("provenance")
        if isinstance(provenance, dict):
            value = provenance.get("checkpoint_hash") or provenance.get("model_hash")
            if value:
                return str(value)
    if isinstance(checkpoint_card, dict):
        for key in ("checkpoint_hash", "model_hash", "sha256"):
            if checkpoint_card.get(key):
                return str(checkpoint_card[key])
    return ""


def _guardrail_payload(*, qc_status: str, fatal_errors: list[str], warnings: list[str]) -> dict[str, Any]:
    return {
        "qc_status": qc_status,
        "conclusion_allowed": not fatal_errors,
        "fatal_errors": fatal_errors,
        "warnings": warnings,
        "rules": [
            "QC fatal blocks biological conclusions.",
            "Warning-only evidence must be reported as cautionary.",
            "Model-derived evidence must not be described as measured expression.",
            "Every claim must link to evidence IDs and artifact manifest entries.",
        ],
    }


def _structure_rows_from_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": claim["claim_id"],
            "evidence_ids": ";".join(claim.get("evidence_ids", [])),
            "qc_status": claim.get("qc_status", ""),
            "human_review_state": claim.get("human_review_state", ""),
            "artifact_ids": ";".join(claim.get("artifact_ids", [])),
            "checkpoint_hash": claim.get("checkpoint_hash", ""),
            "model_derived": claim.get("model_derived", True),
            "measured_expression": claim.get("measured_expression", False),
            "cautionary": claim.get("cautionary", False),
            "claim_text": claim.get("claim_text", ""),
        }
        for claim in claims
    ]


def _human_review_rows(claims: list[dict[str, Any]], guardrail: dict[str, Any]) -> list[dict[str, str]]:
    if not guardrail["conclusion_allowed"]:
        return [
            {
                "item": "QC fatal review",
                "state": "pending",
                "evidence_ids": "",
                "required_action": "Resolve fatal QC before accepting biological conclusions.",
            }
        ]
    return [
        {
            "item": claim["claim_id"],
            "state": str(claim.get("human_review_state", "pending")),
            "evidence_ids": ";".join(claim.get("evidence_ids", [])),
            "required_action": "Pathologist review required before clinical or biological assertion.",
        }
        for claim in claims
    ]


def _metrics_summary(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    prediction = metrics.get("overall_prediction") or {}
    retrieval = metrics.get("overall_retrieval") or []
    top5 = next((row for row in retrieval if isinstance(row, dict) and int(row.get("k", -1)) == 5), {})
    return {
        "gene_mse": prediction.get("gene_mse"),
        "gene_correlation": prediction.get("gene_correlation"),
        "image_to_gene_top5": top5.get("image_to_gene_topk"),
        "gene_to_image_top5": top5.get("gene_to_image_topk"),
    }


def _report_markdown(
    *,
    case_name: str,
    question: str,
    guardrail: dict[str, Any],
    claims: list[dict[str, Any]],
    metrics_summary: dict[str, Any],
    pyxenium_summary: str | None,
) -> str:
    lines = [
        f"# Agentic Spatial Pathologist v0.1: {case_name}",
        "",
        f"**Question:** {question}",
        "",
        "## Guardrail Verdict",
        "",
        f"- QC status: `{guardrail['qc_status']}`",
        f"- Biological conclusions allowed: `{guardrail['conclusion_allowed']}`",
        "- Evidence type: model-derived unless explicitly marked measured.",
        "",
    ]
    if guardrail["fatal_errors"]:
        lines.extend(["## Fatal QC", "", *[f"- {item}" for item in guardrail["fatal_errors"]], ""])
    if guardrail["warnings"]:
        lines.extend(["## Cautionary Warnings", "", *[f"- {item}" for item in guardrail["warnings"]], ""])
    if metrics_summary:
        lines.extend(["## stGPT Metrics Summary", "", *[f"- {key}: `{value}`" for key, value in metrics_summary.items()], ""])
    lines.extend(["## Claims", ""])
    for claim in claims:
        lines.extend(
            [
                f"### {claim['claim_id']}",
                "",
                claim["claim_text"],
                "",
                f"- Evidence IDs: `{'; '.join(claim.get('evidence_ids', []))}`",
                f"- QC status: `{claim.get('qc_status', '')}`",
                f"- Human review: `{claim.get('human_review_state', '')}`",
                f"- Artifact IDs: `{'; '.join(claim.get('artifact_ids', []))}`",
                f"- Checkpoint hash: `{claim.get('checkpoint_hash', '')}`",
                "",
            ]
        )
    if pyxenium_summary is not None:
        lines.extend(["## Optional pyXenium Evidence", "", "pyXenium summary artifact was linked for review.", ""])
    return "\n".join(lines)


def _guardrail_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Failure And Guardrail Section", ""]
    lines.append(f"- QC status: `{payload['qc_status']}`")
    lines.append(f"- Conclusion allowed: `{payload['conclusion_allowed']}`")
    lines.extend(f"- Rule: {rule}" for rule in payload["rules"])
    if payload["fatal_errors"]:
        lines.extend(["", "## Fatal Errors", *[f"- {item}" for item in payload["fatal_errors"]]])
    if payload["warnings"]:
        lines.extend(["", "## Warnings", *[f"- {item}" for item in payload["warnings"]]])
    lines.append("")
    return "\n".join(lines)


def _review_markdown(rows: list[dict[str, str]]) -> str:
    lines = ["# Human Review Checklist", ""]
    for row in rows:
        lines.extend(
            [
                f"- [ ] {row['item']}",
                f"  - State: `{row['state']}`",
                f"  - Evidence IDs: `{row['evidence_ids']}`",
                f"  - Required action: {row['required_action']}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _load_first_json(base: Path, names: tuple[str, ...]) -> dict[str, Any]:
    for name in names:
        payload = _load_optional_json(base / name)
        if payload is not None:
            return payload
    return {}


def _load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _load_optional_text(path: str | Path | None) -> str | None:
    if path is None:
        return None
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    return target.read_text(encoding="utf-8", errors="replace")


def _fatal_errors(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("fatal_errors") or payload.get("errors") or []
    if payload.get("status") in {"fail", "fatal"} and not raw:
        raw = ["QC status is fail."]
    return [str(item) for item in raw if str(item).strip()]


def _warnings(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("warnings") or []
    return [str(item) for item in raw if str(item).strip()]


def _sha256_if_small(path: Path, *, max_bytes: int = 50_000_000) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    if path.stat().st_size > max_bytes:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


__all__ = ["DEMO_QUESTION", "build_agentic_spatial_pathologist_demo"]
