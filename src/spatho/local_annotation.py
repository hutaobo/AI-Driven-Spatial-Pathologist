from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from urllib import error, request

import pandas as pd

from histoseg.annotation import ClusterAnnotationPipelineConfig, run_cluster_annotation_pipeline
from histoseg.annotation.evidence_pack import infer_differential_expression_csv, infer_projection_csv
from histoseg.annotation.report import write_annotation_report
from histoseg.annotation.taxonomy import get_label_by_id, label_taxonomy_payload, normalize_taxonomy_name
from histoseg.spatial_pathologist.artifact_loader import ensure_base_pipeline_outputs, load_base_pipeline_config
from histoseg.spatial_pathologist.config import SpatialPathologistConfig
from histoseg.spatial_pathologist.runner import run_spatial_pathologist

from .schema import WorkflowConfig


@dataclass(frozen=True)
class ConsensusSettings:
    min_llm_confidence: float
    override_margin: float
    require_marker_overlap: bool


class PathologyAIClusterAnnotationClient:
    def __init__(self, *, base_url: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def health(self) -> dict[str, Any]:
        try:
            return _json_request(url=f"{self.base_url}/health", method="GET", timeout=10.0)
        except Exception as exc:
            return {"ready": False, "error": str(exc)}

    def annotate_cluster(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _json_request(
            url=f"{self.base_url}/annotations/cluster",
            method="POST",
            timeout=self.timeout_seconds,
            payload=payload,
        )


def _json_request(
    *,
    url: str,
    method: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Unable to reach {url}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {url}: {raw[:240]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object from {url}.")
    return parsed


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_annotation_table(path: Path, annotations: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not annotations:
        pd.DataFrame().to_csv(path, index=False)
        return path
    pd.DataFrame(annotations).sort_values("cluster_id").to_csv(path, index=False)
    return path


def _write_compatibility_csv(path: Path, annotations: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(annotations).sort_values("cluster_id")
    compatibility = table[["cluster_id", "detailed_label"]].copy()
    compatibility.columns = ["cluster", "celltype"]
    compatibility.to_csv(path, index=False)
    return path


def _annotation_case_review(cluster_annotations: list[dict[str, Any]]) -> dict[str, Any]:
    high_priority = [
        int(item["cluster_id"])
        for item in cluster_annotations
        if str(item.get("review_priority", "")).lower() == "high"
    ]
    discovery_candidates = [
        int(item["cluster_id"])
        for item in cluster_annotations
        if str(item.get("malignancy_state", "")).lower() == "tumor"
        or "tumor" in str(item.get("label_id", "")).lower()
    ]
    return {
        "headline": f"Consensus annotation completed for {len(cluster_annotations)} clusters.",
        "overall_impression": (
            "Cluster labels combine marker-based heuristic evidence with a local pathology-ai consistency pass."
        ),
        "high_priority_cluster_ids": high_priority[:8],
        "consistency_notes": [
            "Final labels use conservative consensus rules; rejected local LLM suggestions are retained in metadata.",
            "Tumor-vs-normal adjudication remains marker-driven unless morphology or CNV artifacts are provided.",
        ],
        "discovery_candidates": discovery_candidates[:8],
    }


def _marker_genes(cluster_evidence: dict[str, Any], key: str) -> set[str]:
    genes: set[str] = set()
    for marker in cluster_evidence.get(key, []):
        gene = str(marker.get("gene", "")).strip().upper()
        if gene:
            genes.add(gene)
    return genes


def _has_marker_overlap(llm_annotation: dict[str, Any], cluster_evidence: dict[str, Any]) -> bool:
    positive_genes = _marker_genes(cluster_evidence, "top_positive_markers")
    supporting = {str(gene).strip().upper() for gene in llm_annotation.get("supporting_markers", []) if str(gene).strip()}
    return bool(positive_genes & supporting)


def _coerce_llm_annotation(
    *,
    llm_payload: dict[str, Any],
    heuristic_annotation: dict[str, Any],
    taxonomy_name: str,
) -> dict[str, Any]:
    label_by_id = get_label_by_id(taxonomy_name)
    label_id = str(llm_payload["label_id"])
    if label_id not in label_by_id:
        raise ValueError(f"label_id {label_id!r} is not valid for taxonomy {taxonomy_name!r}")
    spec = label_by_id[label_id]
    alternative_ids = [
        str(value)
        for value in llm_payload.get("alternative_label_ids", [])
        if str(value) in label_by_id and str(value) != label_id
    ]
    return {
        "cluster_id": int(heuristic_annotation["cluster_id"]),
        "label_id": spec.id,
        "detailed_label": spec.label,
        "broad_family": spec.broad_family,
        "malignancy_state": spec.malignancy_state,
        "confidence": round(float(llm_payload["confidence"]), 3),
        "supporting_markers": [str(value) for value in llm_payload.get("supporting_markers", [])],
        "conflicting_markers": [str(value) for value in llm_payload.get("conflicting_markers", [])],
        "alternative_label_ids": alternative_ids,
        "alternative_labels": [label_by_id[label_id].label for label_id in alternative_ids],
        "reasoning_summary": str(llm_payload.get("reasoning_summary", "")).strip(),
        "review_priority": str(llm_payload.get("review_priority", "medium")),
        "tumor_evidence": [str(value) for value in llm_payload.get("tumor_evidence", [])],
        "recommended_follow_up": [str(value) for value in llm_payload.get("recommended_follow_up", [])],
        "downstream_cell_type": spec.label,
        "engine": "pathology_ai_api",
        "prompt_version": f"pathology-ai-cluster-{taxonomy_name}-v1",
    }


def choose_consensus_annotation(
    *,
    heuristic_annotation: dict[str, Any],
    llm_annotation: dict[str, Any] | None,
    cluster_evidence: dict[str, Any],
    settings: ConsensusSettings,
) -> tuple[dict[str, Any], dict[str, Any]]:
    heuristic_confidence = float(heuristic_annotation.get("confidence", 0.0))
    decision = {
        "cluster_id": int(heuristic_annotation["cluster_id"]),
        "heuristic_label_id": heuristic_annotation.get("label_id"),
        "heuristic_confidence": heuristic_confidence,
        "pathology_ai_label_id": None,
        "pathology_ai_confidence": None,
        "marker_overlap": False,
        "accepted": False,
        "reason": "pathology_ai_unavailable",
    }
    if llm_annotation is None:
        final = dict(heuristic_annotation)
        final["engine"] = "consensus:pathology_ai_api"
        final["consensus_source"] = "heuristic"
        final["consensus_reason"] = decision["reason"]
        return final, decision

    llm_confidence = float(llm_annotation.get("confidence", 0.0))
    same_label = llm_annotation.get("label_id") == heuristic_annotation.get("label_id")
    marker_overlap = _has_marker_overlap(llm_annotation, cluster_evidence)
    decision.update(
        {
            "pathology_ai_label_id": llm_annotation.get("label_id"),
            "pathology_ai_confidence": llm_confidence,
            "marker_overlap": marker_overlap,
        }
    )
    if llm_confidence < settings.min_llm_confidence:
        reason = "pathology_ai_low_confidence"
        accept = False
    elif same_label:
        reason = "same_label"
        accept = True
    elif settings.require_marker_overlap and not marker_overlap:
        reason = "no_marker_overlap"
        accept = False
    elif heuristic_confidence < 0.65:
        reason = "heuristic_low_confidence"
        accept = True
    elif llm_confidence - heuristic_confidence >= settings.override_margin:
        reason = "pathology_ai_confidence_margin"
        accept = True
    else:
        reason = "heuristic_retained"
        accept = False

    selected = llm_annotation if accept else heuristic_annotation
    final = dict(selected)
    final["engine"] = "consensus:pathology_ai_api"
    final["prompt_version"] = "consensus-pathology-ai-v1"
    final["consensus_source"] = "pathology_ai_api" if accept else "heuristic"
    final["consensus_reason"] = reason
    final["heuristic_label_id"] = heuristic_annotation.get("label_id")
    final["heuristic_confidence"] = heuristic_confidence
    final["pathology_ai_label_id"] = llm_annotation.get("label_id")
    final["pathology_ai_confidence"] = llm_confidence
    decision["accepted"] = accept
    decision["reason"] = reason
    return final, decision


def refine_cluster_annotations_with_pathology_ai(
    *,
    output_dir: Path,
    case_name: str,
    study_context: str,
    annotation_taxonomy: str,
    pathology_ai_base_url: str,
    settings: ConsensusSettings,
) -> dict[str, str]:
    taxonomy_name = normalize_taxonomy_name(annotation_taxonomy)
    output_dir = Path(output_dir)
    evidence_pack = json.loads((output_dir / "cluster_evidence.json").read_text(encoding="utf-8"))
    heuristic_annotations = json.loads((output_dir / "cluster_annotations_openai.json").read_text(encoding="utf-8"))
    heuristic_annotations = sorted(heuristic_annotations, key=lambda item: int(item["cluster_id"]))
    heuristic_by_cluster = {int(item["cluster_id"]): item for item in heuristic_annotations}

    heuristic_json = output_dir / "cluster_annotations_heuristic.json"
    heuristic_csv = output_dir / "cluster_annotations_heuristic.csv"
    _write_json(heuristic_json, heuristic_annotations)
    _write_annotation_table(heuristic_csv, heuristic_annotations)

    client = PathologyAIClusterAnnotationClient(base_url=pathology_ai_base_url)
    vocabulary = label_taxonomy_payload(taxonomy_name)
    service_health = client.health()
    llm_annotations: list[dict[str, Any]] = []
    consensus_annotations: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for cluster_evidence in evidence_pack["clusters"]:
        cluster_id = int(cluster_evidence["cluster_id"])
        heuristic_annotation = heuristic_by_cluster[cluster_id]
        llm_annotation: dict[str, Any] | None = None
        try:
            response = client.annotate_cluster(
                {
                    "case_name": case_name,
                    "study_context": study_context,
                    "annotation_taxonomy": taxonomy_name,
                    "controlled_vocabulary": vocabulary,
                    "cluster_evidence": cluster_evidence,
                    "heuristic_annotation": heuristic_annotation,
                }
            )
            llm_annotation = _coerce_llm_annotation(
                llm_payload=response,
                heuristic_annotation=heuristic_annotation,
                taxonomy_name=taxonomy_name,
            )
            llm_annotations.append(llm_annotation)
        except Exception as exc:
            errors.append({"cluster_id": cluster_id, "error": str(exc)})

        consensus, decision = choose_consensus_annotation(
            heuristic_annotation=heuristic_annotation,
            llm_annotation=llm_annotation,
            cluster_evidence=cluster_evidence,
            settings=settings,
        )
        if llm_annotation is None:
            decision["reason"] = "pathology_ai_error"
            consensus["consensus_reason"] = "pathology_ai_error"
        consensus_annotations.append(consensus)
        decisions.append(decision)

    pathology_ai_json = output_dir / "cluster_annotations_pathology_ai.json"
    pathology_ai_csv = output_dir / "cluster_annotations_pathology_ai.csv"
    consensus_json = output_dir / "cluster_annotations_consensus.json"
    consensus_csv = output_dir / "cluster_annotations_consensus.csv"
    compatibility_csv = output_dir / "cluster_celltype_annotation.csv"
    compatibility_json = output_dir / "cluster_annotations_openai.json"
    compatibility_full_csv = output_dir / "cluster_annotations_openai.csv"
    metadata_json = output_dir / "annotation_refinement_metadata.json"

    _write_json(pathology_ai_json, llm_annotations)
    _write_annotation_table(pathology_ai_csv, llm_annotations)
    _write_json(consensus_json, consensus_annotations)
    _write_annotation_table(consensus_csv, consensus_annotations)
    _write_json(compatibility_json, consensus_annotations)
    _write_annotation_table(compatibility_full_csv, consensus_annotations)
    _write_compatibility_csv(compatibility_csv, consensus_annotations)

    case_review = _annotation_case_review(consensus_annotations)
    _write_json(output_dir / "annotation_case_review.json", case_review)
    report_path = write_annotation_report(
        output_dir=output_dir,
        evidence_pack=evidence_pack,
        cluster_annotations=consensus_annotations,
        case_review=case_review,
    )

    accepted = sum(1 for item in decisions if item.get("accepted") is True)
    rejected = sum(1 for item in decisions if item.get("accepted") is False and item.get("pathology_ai_label_id"))
    fallback = sum(1 for item in decisions if item.get("pathology_ai_label_id") is None)
    metadata = {
        "engine": "consensus:pathology_ai_api",
        "pathology_ai_base_url": pathology_ai_base_url,
        "service_health": service_health,
        "settings": {
            "min_llm_confidence": settings.min_llm_confidence,
            "override_margin": settings.override_margin,
            "require_marker_overlap": settings.require_marker_overlap,
        },
        "cluster_count": len(consensus_annotations),
        "attempted": len(evidence_pack["clusters"]),
        "succeeded": len(llm_annotations),
        "accepted": accepted,
        "rejected": rejected,
        "fallback": fallback,
        "errors": errors,
        "decisions": decisions,
    }
    _write_json(metadata_json, metadata)

    return {
        "cluster_evidence_json": str(output_dir / "cluster_evidence.json"),
        "cluster_annotations_json": str(compatibility_json),
        "cluster_annotations_csv": str(compatibility_full_csv),
        "compatibility_csv": str(compatibility_csv),
        "case_review_json": str(output_dir / "annotation_case_review.json"),
        "report_html": str(report_path),
        "heuristic_annotations_json": str(heuristic_json),
        "heuristic_annotations_csv": str(heuristic_csv),
        "pathology_ai_annotations_json": str(pathology_ai_json),
        "pathology_ai_annotations_csv": str(pathology_ai_csv),
        "consensus_annotations_json": str(consensus_json),
        "consensus_annotations_csv": str(consensus_csv),
        "annotation_refinement_metadata_json": str(metadata_json),
    }


def _write_runtime_config(runtime_cfg: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for key, value in runtime_cfg.items():
        if key == "project_root":
            continue
        payload[key] = str(value) if isinstance(value, Path) else value
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _build_runtime_base_config(
    base_cfg: dict[str, Any],
    *,
    annotation_csv: Path,
    output_root: Path,
) -> dict[str, Any]:
    runtime_cfg = dict(base_cfg)
    runtime_cfg["cluster_annotation_csv"] = annotation_csv.resolve()
    runtime_cfg["analysis_output_dir"] = (output_root / "pipeline" / "structure_assignment").resolve()
    runtime_cfg["validation_output_dir"] = (output_root / "pipeline" / "validation").resolve()
    runtime_cfg["config_path"] = (output_root / "runtime_configs" / "generated_runtime_config.json").resolve()
    return runtime_cfg


def _should_recompute_pipeline(*, annotation_csv: Path, output_root: Path, force_recompute_pipeline: bool) -> bool:
    if force_recompute_pipeline:
        return True
    structure_assignments = output_root / "pipeline" / "structure_assignment" / "structure_assignments.csv"
    validation_summary = output_root / "pipeline" / "validation" / "xenium_explorer_annotations_summary.csv"
    if not structure_assignments.exists() or not validation_summary.exists():
        return True
    annotation_mtime = annotation_csv.stat().st_mtime
    return annotation_mtime > min(structure_assignments.stat().st_mtime, validation_summary.stat().st_mtime)


def run_workflow_with_local_cluster_annotation(cfg: WorkflowConfig) -> dict[str, str]:
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base_cfg = load_base_pipeline_config(cfg.base_pipeline_config)
    differential_expression_csv = Path(
        cfg.differential_expression_csv or infer_differential_expression_csv(base_cfg["cluster_csv"])
    )
    projection_csv = Path(cfg.projection_csv or infer_projection_csv(base_cfg["dataset_root"]))

    initial_outputs = run_cluster_annotation_pipeline(
        ClusterAnnotationPipelineConfig(
            case_name=cfg.case_name,
            study_context=cfg.study_context,
            cluster_csv=Path(base_cfg["cluster_csv"]),
            differential_expression_csv=differential_expression_csv,
            projection_csv=projection_csv,
            output_dir=output_root / "annotation",
            annotation_taxonomy=cfg.annotation_taxonomy,
            openai_enabled=False,
            openai_api_key_env=cfg.openai_api_key_env,
            openai_model=cfg.openai_model,
            openai_reasoning_effort=cfg.openai_reasoning_effort,
            openai_store=cfg.openai_store,
            force_recompute=cfg.force_recompute_annotation,
            top_positive_markers=cfg.top_positive_markers,
            top_negative_markers=cfg.top_negative_markers,
            min_log2fc=cfg.min_log2fc,
            max_adjusted_p_value=cfg.max_adjusted_p_value,
            top_neighbors=cfg.top_neighbors,
        )
    )
    annotation_base_url = cfg.cluster_annotation_llm_base_url or cfg.pathology_ai_api_base_url
    annotation_outputs = {
        **initial_outputs,
        **refine_cluster_annotations_with_pathology_ai(
            output_dir=output_root / "annotation",
            case_name=cfg.case_name,
            study_context=cfg.study_context,
            annotation_taxonomy=cfg.annotation_taxonomy,
            pathology_ai_base_url=annotation_base_url,
            settings=ConsensusSettings(
                min_llm_confidence=cfg.cluster_annotation_min_llm_confidence,
                override_margin=cfg.cluster_annotation_override_margin,
                require_marker_overlap=cfg.cluster_annotation_require_marker_overlap,
            ),
        ),
    }

    runtime_cfg = _build_runtime_base_config(
        base_cfg,
        annotation_csv=Path(annotation_outputs["compatibility_csv"]),
        output_root=output_root,
    )
    runtime_cfg_path = _write_runtime_config(
        runtime_cfg,
        output_root / "runtime_configs" / "generated_runtime_config.json",
    )
    recompute_pipeline = _should_recompute_pipeline(
        annotation_csv=Path(annotation_outputs["compatibility_csv"]),
        output_root=output_root,
        force_recompute_pipeline=cfg.force_recompute_pipeline,
    )
    spatial_cfg = SpatialPathologistConfig(
        case_name=cfg.case_name,
        study_context=cfg.study_context,
        base_pipeline_config=runtime_cfg_path,
        output_dir=output_root / "pathology_review",
        pathology_review_backend=cfg.pathology_review_backend,
        pathology_ai_api_base_url=cfg.pathology_ai_api_base_url,
        pathology_ai_top_k=cfg.pathology_ai_top_k,
        pathology_ai_answer_language=cfg.pathology_ai_answer_language,
        pathology_ai_document_ids=tuple(cfg.pathology_ai_document_ids),
        openai_enabled=cfg.openai_enabled,
        openai_api_key_env=cfg.openai_api_key_env,
        openai_model=cfg.openai_model,
        openai_reasoning_effort=cfg.openai_reasoning_effort,
        openai_store=cfg.openai_store,
        force_recompute_pipeline=recompute_pipeline,
        low_confidence_threshold=cfg.low_confidence_threshold,
        ambiguity_margin_threshold=cfg.ambiguity_margin_threshold,
        top_clusters_per_structure=cfg.top_clusters_per_structure,
    )
    ensure_base_pipeline_outputs(spatial_cfg)
    pathology_outputs = run_spatial_pathologist(
        SpatialPathologistConfig(
            **{
                **spatial_cfg.__dict__,
                "force_recompute_pipeline": False,
            }
        )
    )
    workflow_summary = {
        "case_name": cfg.case_name,
        "study_context": cfg.study_context,
        "output_root": str(output_root),
        "annotation_outputs": annotation_outputs,
        "runtime_base_pipeline_config": str(runtime_cfg_path),
        "pathology_outputs": pathology_outputs,
        "responses_api_ready": True,
        "annotation_taxonomy": cfg.annotation_taxonomy,
        "cluster_annotation_backend": cfg.cluster_annotation_backend,
        "cluster_annotation_llm_base_url": annotation_base_url,
        "cluster_annotation_min_llm_confidence": cfg.cluster_annotation_min_llm_confidence,
        "cluster_annotation_override_margin": cfg.cluster_annotation_override_margin,
        "cluster_annotation_require_marker_overlap": cfg.cluster_annotation_require_marker_overlap,
        "pathology_review_backend": cfg.pathology_review_backend,
        "pathology_ai_api_base_url": cfg.pathology_ai_api_base_url,
        "pathology_ai_top_k": cfg.pathology_ai_top_k,
        "pathology_ai_answer_language": cfg.pathology_ai_answer_language,
        "pathology_ai_document_ids": list(cfg.pathology_ai_document_ids),
        "pipeline_recomputed": recompute_pipeline,
        "openai_enabled": cfg.openai_enabled,
        "openai_api_key_env": cfg.openai_api_key_env,
        "openai_model": cfg.openai_model,
        "openai_reasoning_effort": cfg.openai_reasoning_effort,
        "openai_store": cfg.openai_store,
    }
    workflow_summary_path = output_root / "workflow_summary.json"
    _write_json(workflow_summary_path, workflow_summary)
    return {
        "output_root": str(output_root),
        "annotation_report_html": annotation_outputs["report_html"],
        "annotation_csv": annotation_outputs["compatibility_csv"],
        "runtime_base_pipeline_config": str(runtime_cfg_path),
        "pathology_report_html": pathology_outputs["report_html"],
        "workflow_summary_json": str(workflow_summary_path),
    }
