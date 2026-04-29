from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathology_ai_service.tei_compat_server import TEICompatService, serve


class FakeEmbeddingBackend:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(text.lower().count("lung"))] for text in texts]

    def health(self) -> dict[str, object]:
        return {"ok": True, "backend": "fake-embedding"}


class FakeRerankerBackend:
    def rerank(self, *, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        tokens = set(query.lower().split())
        scored = []
        for index, text in enumerate(texts):
            score = float(len(tokens.intersection(text.lower().split())))
            scored.append((index, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_n]

    def health(self) -> dict[str, object]:
        return {"ok": True, "backend": "fake-reranker"}


def _start_server(service: TEICompatService, *, host: str = "127.0.0.1", port: int) -> threading.Thread:
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


def _json_request(url: str, payload: dict[str, object] | None = None) -> tuple[int, object]:
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


def test_embedding_endpoint_returns_tei_compatible_vectors() -> None:
    service = TEICompatService(mode="embed", model_id="BAAI/bge-m3", embedding_backend=FakeEmbeddingBackend())
    _start_server(service, port=8780)

    health_status, health_payload = _json_request("http://127.0.0.1:8780/health")
    embed_status, embed_payload = _json_request("http://127.0.0.1:8780/embed", payload={"inputs": ["lung text", "other"]})
    single_status, single_payload = _json_request("http://127.0.0.1:8780/embed", payload={"inputs": "lung"})

    assert health_status == 200
    assert health_payload["ok"] is True
    assert health_payload["model"] == "BAAI/bge-m3"
    assert embed_status == 200
    assert embed_payload == [[9.0, 1.0], [5.0, 0.0]]
    assert single_status == 200
    assert single_payload == [4.0, 1.0]


def test_rerank_endpoint_returns_ranked_results() -> None:
    service = TEICompatService(mode="rerank", model_id="BAAI/bge-reranker-v2-m3", reranker_backend=FakeRerankerBackend())
    _start_server(service, port=8781)

    status, payload = _json_request(
        "http://127.0.0.1:8781/rerank",
        payload={"query": "lung gland", "texts": ["breast duct", "lung gland formation", "lung"], "top_n": 2},
    )

    assert status == 200
    assert payload == {"results": [{"index": 1, "score": 2.0}, {"index": 2, "score": 1.0}]}


def test_wrong_mode_returns_clear_error() -> None:
    service = TEICompatService(mode="embed", model_id="BAAI/bge-m3", embedding_backend=FakeEmbeddingBackend())
    _start_server(service, port=8782)

    status, payload = _json_request("http://127.0.0.1:8782/rerank", payload={"query": "lung", "texts": ["lung"]})

    assert status == 404
    assert payload["error"] == "service_error"
    assert "not started in reranker mode" in payload["message"]
