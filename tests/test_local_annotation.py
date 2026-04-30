from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from spatho.local_annotation import (
    ConsensusSettings,
    choose_consensus_annotation,
    refine_cluster_annotations_with_pathology_ai,
)


def _heuristic_annotation(*, confidence: float = 0.35) -> dict[str, object]:
    return {
        "cluster_id": 3,
        "label_id": "lymphocytes_mixed",
        "detailed_label": "Lymphocytes (T/B mixed)",
        "broad_family": "immune",
        "malignancy_state": "non_tumor",
        "confidence": confidence,
        "supporting_markers": ["TRAC"],
        "conflicting_markers": ["CXCL12"],
        "alternative_label_ids": [],
        "alternative_labels": [],
        "reasoning_summary": "Heuristic lymphocyte call.",
        "review_priority": "high",
        "tumor_evidence": [],
        "recommended_follow_up": [],
        "downstream_cell_type": "Lymphocytes (T/B mixed)",
        "engine": "heuristic",
        "prompt_version": "heuristic-breast-v1",
    }


def _llm_annotation(*, confidence: float = 0.86, supporting_markers: list[str] | None = None) -> dict[str, object]:
    return {
        "cluster_id": 3,
        "label_id": "fibroblasts_stromal",
        "detailed_label": "Fibroblasts / Stromal Cells",
        "broad_family": "stromal",
        "malignancy_state": "microenvironment",
        "confidence": confidence,
        "supporting_markers": supporting_markers or ["CXCL12"],
        "conflicting_markers": [],
        "alternative_label_ids": [],
        "alternative_labels": [],
        "reasoning_summary": "CXCL12 supports fibroblastic stroma.",
        "review_priority": "low",
        "tumor_evidence": [],
        "recommended_follow_up": [],
        "downstream_cell_type": "Fibroblasts / Stromal Cells",
        "engine": "pathology_ai_api",
        "prompt_version": "pathology-ai-cluster-breast-v1",
    }


def _cluster_evidence() -> dict[str, object]:
    return {
        "cluster_id": 3,
        "cluster_size": 42,
        "top_positive_markers": [
            {"gene": "CXCL12", "log2fc": 2.4, "adjusted_p_value": 0.001, "mean_counts": 1.2},
            {"gene": "LAMA2", "log2fc": 1.8, "adjusted_p_value": 0.001, "mean_counts": 0.8},
        ],
        "top_negative_markers": [{"gene": "EPCAM", "log2fc": -1.1}],
        "umap": {"centroid_umap_1": 0.0, "centroid_umap_2": 0.0},
        "nearest_clusters_in_umap": [],
    }


def _settings() -> ConsensusSettings:
    return ConsensusSettings(
        min_llm_confidence=0.60,
        override_margin=0.15,
        require_marker_overlap=True,
    )


def test_consensus_accepts_low_confidence_heuristic_with_marker_overlap() -> None:
    consensus, decision = choose_consensus_annotation(
        heuristic_annotation=_heuristic_annotation(confidence=0.35),
        llm_annotation=_llm_annotation(confidence=0.86),
        cluster_evidence=_cluster_evidence(),
        settings=_settings(),
    )

    assert decision["accepted"] is True
    assert decision["reason"] == "heuristic_low_confidence"
    assert consensus["detailed_label"] == "Fibroblasts / Stromal Cells"
    assert consensus["engine"] == "consensus:pathology_ai_api"


def test_consensus_retains_confident_heuristic_without_marker_overlap() -> None:
    consensus, decision = choose_consensus_annotation(
        heuristic_annotation=_heuristic_annotation(confidence=0.91),
        llm_annotation=_llm_annotation(confidence=0.95, supporting_markers=["NOT_IN_EVIDENCE"]),
        cluster_evidence=_cluster_evidence(),
        settings=_settings(),
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "no_marker_overlap"
    assert consensus["detailed_label"] == "Lymphocytes (T/B mixed)"
    assert consensus["consensus_source"] == "heuristic"


def test_consensus_falls_back_when_llm_annotation_is_missing() -> None:
    consensus, decision = choose_consensus_annotation(
        heuristic_annotation=_heuristic_annotation(confidence=0.72),
        llm_annotation=None,
        cluster_evidence=_cluster_evidence(),
        settings=_settings(),
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "pathology_ai_unavailable"
    assert consensus["detailed_label"] == "Lymphocytes (T/B mixed)"
    assert consensus["consensus_source"] == "heuristic"


class _AnnotationHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _write_json(self, status: int, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        self._write_json(200, {"service": "pathology-ai", "ready": True})

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self._write_json(
            200,
            {
                "label_id": "fibroblasts_stromal",
                "confidence": 0.86,
                "review_priority": "low",
                "supporting_markers": ["CXCL12"],
                "conflicting_markers": [],
                "alternative_label_ids": [],
                "reasoning_summary": "CXCL12 supports fibroblastic stroma.",
                "tumor_evidence": [],
                "recommended_follow_up": [],
            },
        )


def test_refine_cluster_annotations_writes_consensus_outputs(tmp_path) -> None:
    output_dir = tmp_path / "annotation"
    output_dir.mkdir()
    (output_dir / "cluster_evidence.json").write_text(
        json.dumps(
            {
                "case_name": "breast_demo",
                "study_context": "Breast case",
                "cluster_count": 1,
                "clusters": [_cluster_evidence()],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "cluster_annotations_openai.json").write_text(
        json.dumps([_heuristic_annotation(confidence=0.35)]),
        encoding="utf-8",
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), _AnnotationHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = refine_cluster_annotations_with_pathology_ai(
            output_dir=output_dir,
            case_name="breast_demo",
            study_context="Breast case",
            annotation_taxonomy="breast",
            pathology_ai_base_url=f"http://127.0.0.1:{server.server_port}",
            settings=_settings(),
        )
    finally:
        server.shutdown()

    compatibility = (output_dir / "cluster_celltype_annotation.csv").read_text(encoding="utf-8")
    metadata = json.loads((output_dir / "annotation_refinement_metadata.json").read_text(encoding="utf-8"))
    consensus = json.loads((output_dir / "cluster_annotations_consensus.json").read_text(encoding="utf-8"))

    assert "Fibroblasts / Stromal Cells" in compatibility
    assert metadata["attempted"] == 1
    assert metadata["succeeded"] == 1
    assert metadata["accepted"] == 1
    assert consensus[0]["engine"] == "consensus:pathology_ai_api"
    assert result["annotation_refinement_metadata_json"].endswith("annotation_refinement_metadata.json")
