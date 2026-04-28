from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import mimetypes

from .organ_packs import get_organ_pack
from .schema import WorkflowConfig


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(*, artifact_id: str, label: str, category: str, path: Path, output_root: Path) -> dict[str, Any]:
    media_type, _ = mimetypes.guess_type(str(path))
    try:
        relative_path = str(path.resolve().relative_to(output_root.resolve()))
    except ValueError:
        relative_path = path.name
    return {
        "id": artifact_id,
        "label": label,
        "category": category,
        "path": str(path),
        "relative_path": relative_path,
        "exists": path.exists(),
        "media_type": media_type or "application/octet-stream",
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": _sha256(path),
    }


def build_artifact_manifest(
    *,
    workflow_config: WorkflowConfig,
    workflow_summary_path: str | Path,
) -> dict[str, Any]:
    workflow_summary_file = Path(workflow_summary_path).resolve()
    summary: dict[str, Any] = json.loads(workflow_summary_file.read_text(encoding="utf-8"))
    output_root = Path(summary["output_root"]).resolve()
    pack = get_organ_pack(summary.get("annotation_taxonomy", workflow_config.annotation_taxonomy))

    candidates: list[tuple[str, str, str, Path]] = [
        ("workflow.workflow_summary_json", "Workflow summary", "workflow", workflow_summary_file),
        ("workflow.runtime_config_json", "Generated runtime config", "workflow", Path(summary["runtime_base_pipeline_config"]).resolve()),
        ("annotation.cluster_evidence_json", "Cluster evidence", "annotation", Path(summary["annotation_outputs"]["cluster_evidence_json"]).resolve()),
        ("annotation.cluster_annotations_json", "Cluster annotations JSON", "annotation", Path(summary["annotation_outputs"]["cluster_annotations_json"]).resolve()),
        ("annotation.cluster_annotations_csv", "Cluster annotations CSV", "annotation", Path(summary["annotation_outputs"]["cluster_annotations_csv"]).resolve()),
        ("annotation.compatibility_csv", "Compatibility annotation CSV", "annotation", Path(summary["annotation_outputs"]["compatibility_csv"]).resolve()),
        ("annotation.case_review_json", "Annotation case review", "annotation", Path(summary["annotation_outputs"]["case_review_json"]).resolve()),
        ("annotation.report_html", "Annotation report", "annotation", Path(summary["annotation_outputs"]["report_html"]).resolve()),
        ("pathology.report_html", "Pathology review report", "pathology", Path(summary["pathology_outputs"]["report_html"]).resolve()),
        ("pathology.cluster_reviews_json", "Cluster pathology reviews", "pathology", Path(summary["pathology_outputs"]["cluster_reviews_json"]).resolve()),
        ("pathology.structure_reviews_json", "Structure pathology reviews", "pathology", Path(summary["pathology_outputs"]["structure_reviews_json"]).resolve()),
        ("pathology.case_summary_json", "Case summary", "pathology", Path(summary["pathology_outputs"]["case_summary_json"]).resolve()),
        ("pipeline.structure_clustermap_pdf", "Structure clustermap", "pipeline", output_root / "pipeline" / "structure_assignment" / "structure_clustermap.pdf"),
        ("pipeline.cluster_structure_lookup_csv", "Cluster to structure lookup", "pipeline", output_root / "pipeline" / "structure_assignment" / "cluster_structure_lookup.csv"),
        ("pipeline.run_summary_json", "Structure assignment summary", "pipeline", output_root / "pipeline" / "structure_assignment" / "run_summary.json"),
        ("pipeline.structure_assignments_csv", "Structure assignments", "pipeline", output_root / "pipeline" / "structure_assignment" / "structure_assignments.csv"),
        ("pipeline.overlay_summary_csv", "Explorer annotation summary", "pipeline", output_root / "pipeline" / "validation" / "xenium_explorer_annotations_summary.csv"),
        ("pipeline.xenium_alignment_note_md", "Xenium RNA+protein alignment note", "pipeline", output_root / "pipeline" / "validation" / "xenium_rna_protein_alignment_note.md"),
        ("pipeline.xenium_alignment_fixture_manifest_json", "Xenium RNA+protein fixture manifest", "pipeline", output_root / "pipeline" / "validation" / "xenium_rna_protein_fixture_manifest.json"),
    ]

    artifacts = [
        _artifact_record(
            artifact_id=artifact_id,
            label=label,
            category=category,
            path=path,
            output_root=output_root,
        )
        for artifact_id, label, category, path in candidates
    ]

    required_artifact_ids = set(pack.artifact_contract.get("required", []))
    missing_required = [
        artifact["id"]
        for artifact in artifacts
        if artifact["id"] in required_artifact_ids and not artifact["exists"]
    ]

    return {
        "manifest_version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_name": workflow_config.case_name,
        "output_root": str(output_root),
        "organ_pack": {
            "id": pack.id,
            "display_name": pack.display_name,
            "annotation_taxonomy": pack.annotation_taxonomy,
        },
        "provider": {
            "openai_enabled": workflow_config.openai_enabled,
            "openai_api_key_env": workflow_config.openai_api_key_env,
            "openai_model": workflow_config.openai_model,
            "openai_reasoning_effort": workflow_config.openai_reasoning_effort,
            "openai_store": workflow_config.openai_store,
        },
        "dataset": {
            "modality": workflow_config.dataset_modality,
            "canonical_space": workflow_config.canonical_space,
            "export_space": workflow_config.export_space,
            "xenium_pixel_size_um": workflow_config.xenium_pixel_size_um,
            "segmentation_source": workflow_config.segmentation_source,
        },
        "artifact_counts": {
            "total": len(artifacts),
            "existing": sum(1 for artifact in artifacts if artifact["exists"]),
            "missing_required": len(missing_required),
        },
        "missing_required_artifacts": missing_required,
        "artifacts": artifacts,
    }


def write_artifact_manifest(
    *,
    workflow_config: WorkflowConfig,
    workflow_summary_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    workflow_summary_file = Path(workflow_summary_path).resolve()
    manifest = build_artifact_manifest(
        workflow_config=workflow_config,
        workflow_summary_path=workflow_summary_file,
    )
    output_root = Path(manifest["output_root"]).resolve()
    path = Path(output_path).resolve() if output_path is not None else output_root / "artifact_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
