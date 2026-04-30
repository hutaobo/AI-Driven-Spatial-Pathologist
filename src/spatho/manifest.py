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
    optional_annotation_artifacts = [
        ("annotation.heuristic_annotations_json", "Heuristic cluster annotations JSON", "annotation", "heuristic_annotations_json"),
        ("annotation.heuristic_annotations_csv", "Heuristic cluster annotations CSV", "annotation", "heuristic_annotations_csv"),
        ("annotation.pathology_ai_annotations_json", "Local pathology-ai cluster annotations JSON", "annotation", "pathology_ai_annotations_json"),
        ("annotation.pathology_ai_annotations_csv", "Local pathology-ai cluster annotations CSV", "annotation", "pathology_ai_annotations_csv"),
        ("annotation.consensus_annotations_json", "Consensus cluster annotations JSON", "annotation", "consensus_annotations_json"),
        ("annotation.consensus_annotations_csv", "Consensus cluster annotations CSV", "annotation", "consensus_annotations_csv"),
        ("annotation.refinement_metadata_json", "Annotation refinement metadata", "annotation", "annotation_refinement_metadata_json"),
    ]
    annotation_outputs = summary.get("annotation_outputs", {})
    for artifact_id, label, category, key in optional_annotation_artifacts:
        if key in annotation_outputs:
            candidates.append((artifact_id, label, category, Path(annotation_outputs[key]).resolve()))
    he_outputs = summary.get("he_foundation_outputs", {})
    optional_he_artifacts = [
        ("he.patch_manifest_json", "H&E contour patch manifest", "he_foundation", "patch_manifest_json"),
        ("he.classification_json", "H&E contour classification JSON", "he_foundation", "classification_json"),
        ("he.classification_csv", "H&E contour classification CSV", "he_foundation", "classification_csv"),
        ("he.structure_summary_json", "H&E contour structure summary JSON", "he_foundation", "structure_summary_json"),
        ("he.structure_summary_csv", "H&E contour structure summary CSV", "he_foundation", "structure_summary_csv"),
        ("he.multimodal_names_json", "Multimodal structure names JSON", "he_foundation", "structure_multimodal_names_json"),
        ("he.multimodal_names_csv", "Multimodal structure names CSV", "he_foundation", "structure_multimodal_names_csv"),
        ("he.metadata_json", "H&E foundation metadata", "he_foundation", "metadata_json"),
    ]
    for artifact_id, label, category, key in optional_he_artifacts:
        if key in he_outputs:
            candidates.append((artifact_id, label, category, Path(he_outputs[key]).resolve()))
    foundation_outputs = summary.get("foundation_outputs", {})
    optional_foundation_artifacts = [
        ("foundation.rna_cluster_summary_json", "RNA foundation cluster summary JSON", "foundation", "rna_foundation_cluster_summary_json"),
        ("foundation.rna_cluster_summary_csv", "RNA foundation cluster summary CSV", "foundation", "rna_foundation_cluster_summary_csv"),
        ("foundation.rna_structure_summary_json", "RNA foundation structure summary JSON", "foundation", "rna_foundation_structure_summary_json"),
        ("foundation.rna_structure_summary_csv", "RNA foundation structure summary CSV", "foundation", "rna_foundation_structure_summary_csv"),
        ("foundation.pathway_cluster_summary_json", "Pathway activity cluster summary JSON", "foundation", "pathway_activity_cluster_summary_json"),
        ("foundation.pathway_cluster_summary_csv", "Pathway activity cluster summary CSV", "foundation", "pathway_activity_cluster_summary_csv"),
        ("foundation.pathway_structure_summary_json", "Pathway activity structure summary JSON", "foundation", "pathway_activity_structure_summary_json"),
        ("foundation.pathway_structure_summary_csv", "Pathway activity structure summary CSV", "foundation", "pathway_activity_structure_summary_csv"),
        ("foundation.he_morphology_summary_json", "H&E morphology feature summary JSON", "foundation", "he_morphology_feature_summary_json"),
        ("foundation.he_morphology_summary_csv", "H&E morphology feature summary CSV", "foundation", "he_morphology_feature_summary_csv"),
        ("foundation.niche_fusion_summary_json", "Lightweight niche fusion summary JSON", "foundation", "niche_fusion_summary_json"),
        ("foundation.niche_fusion_summary_csv", "Lightweight niche fusion summary CSV", "foundation", "niche_fusion_summary_csv"),
        ("foundation.stgpt_cell_embeddings_parquet", "stGPT cell embeddings", "foundation", "stgpt_cell_embeddings_parquet"),
        (
            "foundation.stgpt_structure_embedding_summary_csv",
            "stGPT structure embedding summary",
            "foundation",
            "stgpt_structure_embedding_summary_csv",
        ),
        ("foundation.stgpt_qc_report_json", "stGPT QC report", "foundation", "stgpt_qc_report_json"),
        ("foundation.stgpt_evidence_summary_json", "stGPT evidence summary JSON", "foundation", "stgpt_evidence_summary_json"),
        ("foundation.stgpt_evidence_summary_csv", "stGPT evidence summary CSV", "foundation", "stgpt_evidence_summary_csv"),
        ("foundation.metadata_json", "Foundation evidence metadata", "foundation", "metadata_json"),
    ]
    for artifact_id, label, category, key in optional_foundation_artifacts:
        if key in foundation_outputs:
            candidates.append((artifact_id, label, category, Path(foundation_outputs[key]).resolve()))

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
            "cluster_annotation_backend": workflow_config.cluster_annotation_backend,
            "cluster_annotation_llm_base_url": workflow_config.cluster_annotation_llm_base_url,
            "cluster_annotation_min_llm_confidence": workflow_config.cluster_annotation_min_llm_confidence,
            "cluster_annotation_override_margin": workflow_config.cluster_annotation_override_margin,
            "cluster_annotation_require_marker_overlap": workflow_config.cluster_annotation_require_marker_overlap,
            "he_contour_foundation_enabled": workflow_config.he_contour_foundation_enabled,
            "he_foundation_model_id": workflow_config.he_foundation_model_id,
            "he_foundation_prompt_set": workflow_config.he_foundation_prompt_set,
            "he_visual_override_enabled": workflow_config.he_visual_override_enabled,
            "rna_foundation_enabled": workflow_config.rna_foundation_enabled,
            "rna_foundation_backend": workflow_config.rna_foundation_backend,
            "pathway_activity_enabled": workflow_config.pathway_activity_enabled,
            "niche_fusion_enabled": workflow_config.niche_fusion_enabled,
            "niche_fusion_backend": workflow_config.niche_fusion_backend,
            "stgpt_enabled": workflow_config.stgpt_enabled,
            "stgpt_backend": workflow_config.stgpt_backend,
            "stgpt_require_qc_pass": workflow_config.stgpt_require_qc_pass,
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
