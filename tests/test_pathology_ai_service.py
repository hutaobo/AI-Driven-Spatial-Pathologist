from __future__ import annotations

import json
import threading
from pathlib import Path
import sys
import time
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathology_ai_service.config import ServiceSettings
from pathology_ai_service.core import (
    InMemoryVectorStore,
    PathologyAIService,
    _normalize_structure_multimodal_payload,
)
from pathology_ai_service.models import (
    ClusterAnnotationRequest,
    ClusterAnnotationResponse,
    HEContourClass,
    HEContourClassification,
    HEContourClassifyResponse,
    ReviewModelResponse,
    StructureMultimodalNamingRequest,
    StructureMultimodalNamingResponse,
)
from pathology_ai_service.server import serve


class FakeEmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            normalized = text.lower()
            vectors.append(
                [
                    float(normalized.count("lung")),
                    float(normalized.count("breast")),
                    float(len(normalized.split())),
                ]
            )
        return vectors

    def health(self) -> dict[str, object]:
        return {"ok": True}


class FakeRerankerClient:
    def rerank(self, *, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        tokens = set(query.lower().split())
        scored = []
        for index, text in enumerate(texts):
            overlap = len(tokens.intersection(text.lower().split()))
            scored.append((index, float(overlap)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_n]

    def health(self) -> dict[str, object]:
        return {"ok": True}


class FakeLLMClient:
    def generate_cluster_annotation(self, *, payload: ClusterAnnotationRequest) -> ClusterAnnotationResponse:
        return ClusterAnnotationResponse(
            label_id="fibroblasts_stromal",
            confidence=0.86,
            review_priority="low",
            supporting_markers=["CXCL12"],
            conflicting_markers=[],
            alternative_label_ids=[],
            reasoning_summary="CXCL12 supports a stromal fibroblast annotation.",
            tumor_evidence=[],
            recommended_follow_up=[],
        )

    def generate_review(self, *, review_type: str, question: str, answer_language: str, evidence: dict[str, object], citations):
        citation_ids = [citations[0].citation_id] if citations else []
        return ReviewModelResponse(
            summary=f"{review_type} summary for {question}",
            interpretation="Grounded local interpretation",
            confidence=0.82,
            key_evidence=["Marker profile and retrieved references are concordant."],
            caveats=["Research-use-only pathology decision support."],
            recommended_follow_up=["Review the linked H&E tile with a pathologist."],
            citation_ids=citation_ids,
        )

    def generate_structure_multimodal_name(
        self,
        *,
        payload: StructureMultimodalNamingRequest,
    ) -> StructureMultimodalNamingResponse:
        structure_id = payload.structure.get("structure_id")
        top_label = payload.he_visual_summary.get("top_label") or "tumor-rich contour"
        return StructureMultimodalNamingResponse(
            structure_id=int(structure_id) if structure_id is not None else None,
            pre_visual_name=str(payload.current_review.get("title", "Molecular structure")),
            final_name=f"Visual {top_label}",
            visual_override=True,
            confidence=0.88,
            review_priority="medium",
            reasoning_summary="Local H&E contour evidence supports a visual override.",
            visual_evidence=[f"PLIP top visual label: {top_label}"],
            molecular_evidence=["RNA composition was provided for fusion."],
            conflicts=[],
            recommended_checks=["Review representative H&E contour patches."],
        )

    def health(self) -> dict[str, object]:
        return {"ok": True}


class InvalidAnnotationLLMClient(FakeLLMClient):
    def generate_cluster_annotation(self, *, payload: ClusterAnnotationRequest) -> ClusterAnnotationResponse:
        return ClusterAnnotationResponse(
            label_id="not_in_vocabulary",
            confidence=0.91,
            review_priority="low",
            supporting_markers=["CXCL12"],
            conflicting_markers=[],
            alternative_label_ids=[],
            reasoning_summary="Invalid label for testing.",
            tumor_evidence=[],
            recommended_follow_up=[],
        )


class BrokenVectorStore(InMemoryVectorStore):
    def health(self) -> dict[str, object]:
        return {"ok": False, "error": "offline"}

    def query(self, *, vector: list[float], top_k: int, document_ids: list[str]):
        raise RuntimeError("Vector store offline")


class FakeHEClassifier:
    def classify(self, payload):
        return HEContourClassifyResponse(
            case_name=payload.case_name,
            model_id=payload.model_id,
            prompt_set=payload.prompt_set,
            classifications=[
                HEContourClassification(
                    contour_id=payload.contours[0].contour_id,
                    image_path=payload.contours[0].image_path,
                    structure_id=payload.contours[0].structure_id,
                    structure_name=payload.contours[0].structure_name,
                    top_classes=[
                        HEContourClass(
                            label_id="invasive_tumor_epithelium",
                            label="Invasive tumor epithelium",
                            prompt="H&E histopathology patch showing invasive breast carcinoma tumor epithelium",
                            score=0.72,
                        )
                    ],
                    patch_quality={"width": 224, "height": 224},
                )
            ],
            warnings=[],
        )


def _start_server(service: PathologyAIService, *, host: str = "127.0.0.1", port: int = 8765) -> threading.Thread:
    thread = threading.Thread(target=serve, args=(service,), kwargs={"host": host, "port": port}, daemon=True)
    thread.start()
    deadline = time.time() + 5
    health_url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        try:
            status, _ = _json_request(health_url)
        except error.URLError:
            time.sleep(0.05)
            continue
        if status == 200:
            break
    else:  # pragma: no cover - defensive path
        raise RuntimeError(f"Timed out waiting for test server at {health_url}")
    return thread


def _json_request(url: str, payload: dict[str, object] | None = None) -> tuple[int, dict[str, object]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET")
    try:
        with request.urlopen(req, timeout=5) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        status = exc.code
        body = json.loads(exc.read().decode("utf-8"))
    return status, body


def _build_service(vector_store=None) -> PathologyAIService:
    settings = ServiceSettings(vector_db="memory", qdrant_collection="test_collection")
    return PathologyAIService(
        settings=settings,
        embedding_client=FakeEmbeddingClient(),
        reranker_client=FakeRerankerClient(),
        llm_client=FakeLLMClient(),
        vector_store=vector_store or InMemoryVectorStore(),
    )


def test_health_endpoint_reports_ready() -> None:
    service = _build_service()
    _start_server(service, port=8766)

    status, payload = _json_request("http://127.0.0.1:8766/health")

    assert status == 200
    assert payload["ready"] is True
    assert payload["settings"]["llm_model"] == "openai/gpt-oss-120b"


def test_document_filter_is_applied_during_review() -> None:
    service = _build_service()
    _start_server(service, port=8767)

    upsert_status, upsert_payload = _json_request(
        "http://127.0.0.1:8767/documents/upsert",
        payload={
            "documents": [
                {
                    "document_id": "lung-atlas",
                    "title": "Lung Atlas",
                    "text": "Lung adenocarcinoma often shows lung epithelial markers and gland formation.",
                },
                {
                    "document_id": "breast-atlas",
                    "title": "Breast Atlas",
                    "text": "Breast carcinoma often shows breast lineage markers and ductal features.",
                },
            ]
        },
    )
    assert upsert_status == 200
    assert upsert_payload["chunk_count"] >= 2

    review_status, review_payload = _json_request(
        "http://127.0.0.1:8767/reviews/structure",
        payload={
            "question": "What lung pathology interpretation best matches this structure?",
            "document_ids": ["lung-atlas"],
            "evidence": {"markers": ["EPCAM", "KRT19"]},
        },
    )

    assert review_status == 200
    assert review_payload["review_type"] == "structure"
    assert review_payload["retrieval"]["document_filter_applied"] is True
    assert review_payload["citations"]
    assert {citation["document_id"] for citation in review_payload["citations"]} == {"lung-atlas"}


def test_cluster_annotation_endpoint_returns_structured_label() -> None:
    service = _build_service()
    _start_server(service, port=8770)

    status, payload = _json_request(
        "http://127.0.0.1:8770/annotations/cluster",
        payload={
            "case_name": "breast_demo",
            "study_context": "Breast Xenium case.",
            "annotation_taxonomy": "breast",
            "controlled_vocabulary": [
                {
                    "id": "fibroblasts_stromal",
                    "label": "Fibroblasts / Stromal Cells",
                    "marker_genes": ["CXCL12"],
                }
            ],
            "cluster_evidence": {"cluster_id": 3, "top_positive_markers": [{"gene": "CXCL12"}]},
            "heuristic_annotation": {"cluster_id": 3, "label_id": "lymphocytes_mixed"},
        },
    )

    assert status == 200
    assert payload["label_id"] == "fibroblasts_stromal"
    assert payload["confidence"] == 0.86
    assert payload["supporting_markers"] == ["CXCL12"]


def test_cluster_annotation_endpoint_rejects_unknown_label() -> None:
    settings = ServiceSettings(vector_db="memory", qdrant_collection="test_collection")
    service = PathologyAIService(
        settings=settings,
        embedding_client=FakeEmbeddingClient(),
        reranker_client=FakeRerankerClient(),
        llm_client=InvalidAnnotationLLMClient(),
        vector_store=InMemoryVectorStore(),
    )
    _start_server(service, port=8771)

    status, payload = _json_request(
        "http://127.0.0.1:8771/v1/annotations/cluster",
        payload={
            "case_name": "breast_demo",
            "study_context": "Breast Xenium case.",
            "annotation_taxonomy": "breast",
            "controlled_vocabulary": [{"id": "fibroblasts_stromal", "label": "Fibroblasts / Stromal Cells"}],
            "cluster_evidence": {"cluster_id": 3},
            "heuristic_annotation": {"cluster_id": 3, "label_id": "fibroblasts_stromal"},
        },
    )

    assert status == 422
    assert "not in the controlled vocabulary" in payload["message"]


def test_he_contour_classification_endpoint_uses_local_classifier(tmp_path) -> None:
    service = _build_service()
    service._he_classifier = FakeHEClassifier()
    _start_server(service, port=8772)
    image_path = tmp_path / "patch.png"
    image_path.write_bytes(b"fake")

    status, payload = _json_request(
        "http://127.0.0.1:8772/v1/he/contours/classify",
        payload={
            "case_name": "breast_demo",
            "model_id": "vinid/plip",
            "prompt_set": "breast_contour_v1",
            "contours": [
                {
                    "contour_id": "S1 contour",
                    "image_path": str(image_path),
                    "structure_id": 1,
                    "structure_name": "Tumor",
                }
            ],
        },
    )

    assert status == 200
    result = payload["classifications"][0]
    assert result["top_classes"][0]["label_id"] == "invasive_tumor_epithelium"
    assert result["top_classes"][0]["score"] == 0.72


def test_structure_multimodal_naming_endpoint_returns_visual_override() -> None:
    service = _build_service()
    _start_server(service, port=8773)

    status, payload = _json_request(
        "http://127.0.0.1:8773/v1/annotations/structure-multimodal",
        payload={
            "case_name": "breast_demo",
            "study_context": "Breast Xenium case.",
            "annotation_taxonomy": "breast",
            "structure": {"structure_id": 1, "top_clusters": []},
            "current_review": {"title": "Molecular-only name"},
            "he_visual_summary": {"top_label": "Invasive tumor epithelium"},
            "multimodal_evidence": {},
            "override_policy": {"he_visual_override_enabled": True},
        },
    )

    assert status == 200
    assert payload["visual_override"] is True
    assert payload["final_name"] == "Visual Invasive tumor epithelium"


def test_structure_multimodal_payload_normalizes_llm_dict_evidence() -> None:
    normalized = _normalize_structure_multimodal_payload(
        {
            "final_name": "Tumor-stroma interface",
            "visual_evidence": {"top_visual_labels": [{"label": "Stroma", "score": 0.9}]},
            "molecular_evidence": {"top_candidate": {"label": "CAF", "fraction": 0.4}},
            "conflicts": "Visual and molecular evidence disagree.",
            "recommended_checks": None,
        },
        fallback_structure_id=7,
    )

    assert normalized["structure_id"] == 7
    assert normalized["visual_evidence"][0].startswith("top_visual_labels:")
    assert normalized["molecular_evidence"][0].startswith("top_candidate:")
    assert normalized["conflicts"] == ["Visual and molecular evidence disagree."]
    assert normalized["recommended_checks"] == []


def test_review_returns_error_without_matching_documents() -> None:
    service = _build_service()
    _start_server(service, port=8768)

    _json_request(
        "http://127.0.0.1:8768/documents/upsert",
        payload={
            "document_id": "lung-atlas",
            "title": "Lung Atlas",
            "text": "Lung adenocarcinoma reference text.",
        },
    )

    review_status, review_payload = _json_request(
        "http://127.0.0.1:8768/reviews/case",
        payload={
            "question": "Summarize the breast case.",
            "document_ids": ["breast-atlas"],
            "evidence": {"notes": "No matching references should be returned."},
        },
    )

    assert review_status == 404
    assert "No reference passages matched" in review_payload["message"]


def test_review_returns_clear_error_when_vector_store_is_unavailable() -> None:
    service = _build_service(vector_store=BrokenVectorStore())
    _start_server(service, port=8769)

    review_status, review_payload = _json_request(
        "http://127.0.0.1:8769/reviews/structure",
        payload={
            "question": "What pathology interpretation fits this lung structure?",
            "document_ids": ["lung-atlas"],
            "evidence": {"markers": ["EPCAM"]},
        },
    )

    assert review_status == 503
    assert "Vector store query failed" in review_payload["message"]
