from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Callable, Protocol
from urllib import error, parse, request

from .config import ServiceSettings
from .models import (
    Citation,
    DocumentUpsertRequest,
    DocumentUpsertResponse,
    DocumentUpsertResult,
    RetrievalStats,
    ReviewModelResponse,
    ReviewRequest,
    ReviewResponse,
)
from .prompts import build_review_messages


class ServiceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


def _json_request(
    *,
    url: str,
    method: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    body = None
    request_headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)
    req = request.Request(url=url, data=body, headers=request_headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise ServiceError(f"Upstream HTTP {exc.code} from {url}: {details}", status_code=502) from exc
    except error.URLError as exc:
        raise ServiceError(f"Unable to reach upstream service at {url}: {exc}", status_code=503) from exc
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ServiceError(f"Invalid JSON returned from {url}: {raw[:240]}", status_code=502) from exc


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _extract_json_object(raw: str) -> dict[str, Any]:
    normalized = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", normalized, flags=re.DOTALL)
    if fenced:
        normalized = fenced.group(1).strip()
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(normalized[start : end + 1])


def _chunk_text(text: str, *, chunk_size_chars: int, chunk_overlap_chars: int) -> list[tuple[int, int, str]]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    chunks: list[tuple[int, int, str]] = []
    start = 0
    text_length = len(normalized)
    while start < text_length:
        end = min(text_length, start + chunk_size_chars)
        if end < text_length:
            window = normalized[start:end]
            breakpoints = [window.rfind("\n\n"), window.rfind(". "), window.rfind(" ")]
            candidate = max(breakpoints)
            if candidate > chunk_size_chars // 2:
                end = start + candidate + 1
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end >= text_length:
            break
        next_start = max(start + 1, end - chunk_overlap_chars)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


@dataclass(slots=True)
class StoredChunk:
    point_id: str
    document_id: str
    chunk_id: str
    title: str | None
    text: str
    source: str | None
    metadata: dict[str, Any]
    vector: list[float]


@dataclass(slots=True)
class RetrievedChunk:
    citation_id: str
    document_id: str
    chunk_id: str
    title: str | None
    text: str
    source: str | None
    metadata: dict[str, Any]
    score: float


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def health(self) -> dict[str, Any]: ...


class RerankerClient(Protocol):
    def rerank(self, *, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]: ...

    def health(self) -> dict[str, Any]: ...


class LanguageModelClient(Protocol):
    def generate_review(
        self,
        *,
        review_type: str,
        question: str,
        answer_language: str,
        evidence: dict[str, Any],
        citations: list[RetrievedChunk],
    ) -> ReviewModelResponse: ...

    def health(self) -> dict[str, Any]: ...


class VectorStore(Protocol):
    collection_name: str

    def upsert(self, *, chunks: list[StoredChunk]) -> None: ...

    def query(self, *, vector: list[float], top_k: int, document_ids: list[str]) -> list[RetrievedChunk]: ...

    def health(self) -> dict[str, Any]: ...


class TEIEmbeddingClient:
    def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {"inputs": texts[0] if len(texts) == 1 else texts}
        response = _json_request(url=f"{self._base_url}/embed", method="POST", timeout=self._timeout, payload=payload)
        if isinstance(response, list) and response and isinstance(response[0], (int, float)):
            return [[float(value) for value in response]]
        if isinstance(response, list) and (not response or isinstance(response[0], list)):
            return [[float(value) for value in row] for row in response]
        if isinstance(response, dict) and "data" in response:
            return [[float(value) for value in item["embedding"]] for item in response["data"]]
        raise ServiceError("Unexpected embedding response format.", status_code=502)

    def health(self) -> dict[str, Any]:
        try:
            _json_request(url=f"{self._base_url}/health", method="GET", timeout=self._timeout)
            return {"ok": True}
        except ServiceError as exc:
            return {"ok": False, "error": str(exc)}


class TEIRerankerClient:
    def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def rerank(self, *, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        if not texts:
            return []
        payload = {"query": query, "texts": texts, "raw_scores": False}
        response = _json_request(url=f"{self._base_url}/rerank", method="POST", timeout=self._timeout, payload=payload)
        results = response.get("results") if isinstance(response, dict) else response
        if not isinstance(results, list):
            raise ServiceError("Unexpected reranker response format.", status_code=502)
        ordered: list[tuple[int, float]] = []
        for position, item in enumerate(results):
            if isinstance(item, dict):
                index = int(item.get("index", position))
                score = float(item.get("score", 0.0))
            else:
                index = position
                score = float(item)
            ordered.append((index, score))
        ordered.sort(key=lambda item: item[1], reverse=True)
        return ordered[:top_n]

    def health(self) -> dict[str, Any]:
        try:
            _json_request(url=f"{self._base_url}/health", method="GET", timeout=self._timeout)
            return {"ok": True}
        except ServiceError as exc:
            return {"ok": False, "error": str(exc)}


class OpenAICompatibleLLMClient:
    def __init__(self, *, base_url: str, model: str, timeout_seconds: float, strict_json: bool) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._strict_json = strict_json

    def generate_review(
        self,
        *,
        review_type: str,
        question: str,
        answer_language: str,
        evidence: dict[str, Any],
        citations: list[RetrievedChunk],
    ) -> ReviewModelResponse:
        citation_payload = [
            {
                "citation_id": chunk.citation_id,
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "text": chunk.text,
            }
            for chunk in citations
        ]
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": build_review_messages(
                review_type=review_type,
                question=question,
                answer_language=answer_language,
                evidence=evidence,
                citations=citation_payload,
            ),
            "temperature": 0.0,
            "top_p": 1.0,
        }
        if self._strict_json:
            payload["response_format"] = {"type": "json_object"}
        response = _json_request(
            url=f"{self._base_url}/chat/completions",
            method="POST",
            timeout=self._timeout,
            payload=payload,
        )
        try:
            choice = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ServiceError("Unexpected LLM response format.", status_code=502) from exc
        if isinstance(choice, list):
            content = "".join(part.get("text", "") for part in choice if isinstance(part, dict))
        else:
            content = str(choice)
        try:
            parsed = _extract_json_object(content)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ServiceError(f"LLM did not return valid JSON: {content[:240]}", status_code=502) from exc
        return ReviewModelResponse.model_validate(parsed)

    def health(self) -> dict[str, Any]:
        try:
            response = _json_request(url=f"{self._base_url}/models", method="GET", timeout=self._timeout)
            model_ids = [item.get("id") for item in response.get("data", []) if isinstance(item, dict)]
            return {"ok": self._model in model_ids if model_ids else True, "models": model_ids[:10]}
        except ServiceError as exc:
            return {"ok": False, "error": str(exc)}


class InMemoryVectorStore:
    collection_name = "in_memory"

    def __init__(self) -> None:
        self._points: dict[str, StoredChunk] = {}

    def upsert(self, *, chunks: list[StoredChunk]) -> None:
        for chunk in chunks:
            self._points[chunk.point_id] = chunk

    def query(self, *, vector: list[float], top_k: int, document_ids: list[str]) -> list[RetrievedChunk]:
        candidates: list[tuple[float, StoredChunk]] = []
        document_filter = set(document_ids)
        for chunk in self._points.values():
            if document_filter and chunk.document_id not in document_filter:
                continue
            candidates.append((_cosine_similarity(vector, chunk.vector), chunk))
        candidates.sort(key=lambda item: item[0], reverse=True)
        results: list[RetrievedChunk] = []
        for index, (score, chunk) in enumerate(candidates[:top_k], start=1):
            results.append(
                RetrievedChunk(
                    citation_id=f"C{index}",
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    text=chunk.text,
                    source=chunk.source,
                    metadata=chunk.metadata,
                    score=score,
                )
            )
        return results

    def health(self) -> dict[str, Any]:
        return {"ok": True, "points": len(self._points)}


class QdrantVectorStore:
    def __init__(self, *, base_url: str, collection_name: str, timeout_seconds: float, api_key: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self.collection_name = collection_name
        self._timeout = timeout_seconds
        self._api_key = api_key
        self._vector_size: int | None = None

    def _headers(self) -> dict[str, str]:
        return {"api-key": self._api_key} if self._api_key else {}

    def _ensure_collection(self, *, vector_size: int) -> None:
        if self._vector_size == vector_size:
            return
        url = f"{self._base_url}/collections/{parse.quote(self.collection_name, safe='')}"
        try:
            _json_request(url=url, method="GET", timeout=self._timeout, headers=self._headers())
        except ServiceError:
            _json_request(
                url=url,
                method="PUT",
                timeout=self._timeout,
                headers=self._headers(),
                payload={"vectors": {"size": vector_size, "distance": "Cosine"}},
            )
        self._vector_size = vector_size

    def upsert(self, *, chunks: list[StoredChunk]) -> None:
        if not chunks:
            return
        self._ensure_collection(vector_size=len(chunks[0].vector))
        points = [
            {
                "id": chunk.point_id,
                "vector": chunk.vector,
                "payload": {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "source": chunk.source,
                    "metadata": chunk.metadata,
                },
            }
            for chunk in chunks
        ]
        url = f"{self._base_url}/collections/{parse.quote(self.collection_name, safe='')}/points?wait=true"
        _json_request(url=url, method="PUT", timeout=self._timeout, headers=self._headers(), payload={"points": points})

    def query(self, *, vector: list[float], top_k: int, document_ids: list[str]) -> list[RetrievedChunk]:
        url = f"{self._base_url}/collections/{parse.quote(self.collection_name, safe='')}/points/search"
        payload: dict[str, Any] = {
            "vector": vector,
            "limit": top_k,
            "with_payload": True,
        }
        if document_ids:
            payload["filter"] = {
                "must": [
                    {
                        "key": "document_id",
                        "match": {"any": document_ids},
                    }
                ]
            }
        response = _json_request(url=url, method="POST", timeout=self._timeout, headers=self._headers(), payload=payload)
        results = response.get("result", [])
        parsed_results: list[RetrievedChunk] = []
        for index, item in enumerate(results, start=1):
            payload = item.get("payload", {})
            parsed_results.append(
                RetrievedChunk(
                    citation_id=f"C{index}",
                    document_id=str(payload.get("document_id", "")),
                    chunk_id=str(payload.get("chunk_id", "")),
                    title=payload.get("title"),
                    text=str(payload.get("text", "")),
                    source=payload.get("source"),
                    metadata=payload.get("metadata") or {},
                    score=float(item.get("score", 0.0)),
                )
            )
        return parsed_results

    def health(self) -> dict[str, Any]:
        try:
            response = _json_request(url=f"{self._base_url}/collections", method="GET", timeout=self._timeout, headers=self._headers())
            collections = [item.get("name") for item in response.get("result", {}).get("collections", []) if isinstance(item, dict)]
            return {"ok": True, "collections": collections[:20]}
        except ServiceError as exc:
            return {"ok": False, "error": str(exc)}


class PathologyAIService:
    def __init__(
        self,
        *,
        settings: ServiceSettings,
        embedding_client: EmbeddingClient,
        reranker_client: RerankerClient,
        llm_client: LanguageModelClient,
        vector_store: VectorStore,
    ) -> None:
        self._settings = settings
        self._embedding_client = embedding_client
        self._reranker_client = reranker_client
        self._llm_client = llm_client
        self._vector_store = vector_store

    @property
    def settings(self) -> ServiceSettings:
        return self._settings

    def _safe_component_health(self, check: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            return check()
        except Exception as exc:  # pragma: no cover - defensive path
            return {"ok": False, "error": str(exc)}

    def health(self) -> dict[str, Any]:
        llm = self._safe_component_health(self._llm_client.health)
        embedder = self._safe_component_health(self._embedding_client.health)
        reranker = self._safe_component_health(self._reranker_client.health)
        vector_store = self._safe_component_health(self._vector_store.health)
        ready = all(component.get("ok") is True for component in [llm, embedder, reranker, vector_store])
        return {
            "service": "pathology-ai",
            "ready": ready,
            "settings": self._settings.public_dict(),
            "components": {
                "llm": llm,
                "embedder": embedder,
                "reranker": reranker,
                "vector_store": vector_store,
            },
        }

    def upsert_documents(self, payload: DocumentUpsertRequest) -> DocumentUpsertResponse:
        chunk_size = payload.chunk_size_chars or self._settings.max_chunk_chars
        overlap = payload.chunk_overlap_chars if payload.chunk_overlap_chars is not None else self._settings.chunk_overlap_chars
        all_chunks: list[StoredChunk] = []
        per_document: list[DocumentUpsertResult] = []
        for document in payload.documents:
            chunks = _chunk_text(document.text, chunk_size_chars=chunk_size, chunk_overlap_chars=overlap)
            if not chunks:
                raise ServiceError(f"Document {document.document_id!r} produced no chunks.", status_code=400)
            chunk_texts = [chunk_text for _, _, chunk_text in chunks]
            try:
                vectors = self._embedding_client.embed(chunk_texts)
            except ServiceError:
                raise
            except Exception as exc:
                raise ServiceError(f"Embedding request failed: {exc}", status_code=503) from exc
            stored_chunks: list[StoredChunk] = []
            for index, ((_, _, chunk_text), vector) in enumerate(zip(chunks, vectors, strict=True), start=1):
                chunk_id = f"chunk-{index:04d}"
                stored_chunks.append(
                    StoredChunk(
                        point_id=f"{document.document_id}:{chunk_id}",
                        document_id=document.document_id,
                        chunk_id=chunk_id,
                        title=document.title,
                        text=chunk_text,
                        source=document.source,
                        metadata=document.metadata,
                        vector=vector,
                    )
                )
            all_chunks.extend(stored_chunks)
            per_document.append(DocumentUpsertResult(document_id=document.document_id, chunk_count=len(stored_chunks)))
        try:
            self._vector_store.upsert(chunks=all_chunks)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(f"Vector store upsert failed: {exc}", status_code=503) from exc
        return DocumentUpsertResponse(
            collection=self._vector_store.collection_name,
            documents=per_document,
            chunk_count=len(all_chunks),
        )

    def review(self, *, review_type: str, payload: ReviewRequest) -> ReviewResponse:
        top_k = payload.top_k or self._settings.default_top_k
        query_text = payload.question
        try:
            query_vector = self._embedding_client.embed([query_text])[0]
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(f"Embedding request failed: {exc}", status_code=503) from exc
        try:
            candidates = self._vector_store.query(
                vector=query_vector,
                top_k=max(top_k * 2, top_k),
                document_ids=payload.document_ids,
            )
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(f"Vector store query failed: {exc}", status_code=503) from exc
        if not candidates:
            raise ServiceError("No reference passages matched the provided document filter.", status_code=404)
        try:
            reranked_order = self._reranker_client.rerank(
                query=payload.question,
                texts=[chunk.text for chunk in candidates],
                top_n=min(top_k, len(candidates)),
            )
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(f"Reranker request failed: {exc}", status_code=503) from exc
        selected = [candidates[index] for index, _ in reranked_order if 0 <= index < len(candidates)]
        if not selected:
            selected = candidates[:top_k]
        for position, chunk in enumerate(selected, start=1):
            chunk.citation_id = f"C{position}"
        try:
            model_output = self._llm_client.generate_review(
                review_type=review_type,
                question=payload.question,
                answer_language=payload.answer_language,
                evidence=payload.evidence,
                citations=selected,
            )
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(f"LLM request failed: {exc}", status_code=503) from exc
        citations_by_id = {citation.citation_id: citation for citation in selected}
        response_citations: list[Citation] = []
        for citation_id in model_output.citation_ids:
            citation = citations_by_id.get(citation_id)
            if citation is None:
                continue
            response_citations.append(
                Citation(
                    citation_id=citation.citation_id,
                    document_id=citation.document_id,
                    chunk_id=citation.chunk_id,
                    title=citation.title,
                    score=citation.score,
                    text_excerpt=citation.text[:400],
                    source=citation.source,
                    metadata=citation.metadata,
                )
            )
        return ReviewResponse(
            review_type="structure" if review_type == "structure" else "case",
            question=payload.question,
            answer_language=payload.answer_language,
            summary=model_output.summary,
            interpretation=model_output.interpretation,
            confidence=model_output.confidence,
            key_evidence=model_output.key_evidence,
            caveats=model_output.caveats,
            recommended_follow_up=model_output.recommended_follow_up,
            citations=response_citations,
            retrieval=RetrievalStats(
                candidate_count=len(candidates),
                returned_count=len(selected),
                document_filter_applied=bool(payload.document_ids),
            ),
        )


def build_service(
    settings: ServiceSettings | None = None,
    *,
    embedding_client: EmbeddingClient | None = None,
    reranker_client: RerankerClient | None = None,
    llm_client: LanguageModelClient | None = None,
    vector_store: VectorStore | None = None,
) -> PathologyAIService:
    service_settings = settings or ServiceSettings.from_env()
    embedding = embedding_client or TEIEmbeddingClient(
        base_url=service_settings.embed_base_url,
        timeout_seconds=service_settings.request_timeout_seconds,
    )
    reranker = reranker_client or TEIRerankerClient(
        base_url=service_settings.rerank_base_url,
        timeout_seconds=service_settings.request_timeout_seconds,
    )
    llm = llm_client or OpenAICompatibleLLMClient(
        base_url=service_settings.llm_base_url,
        model=service_settings.llm_model,
        timeout_seconds=service_settings.request_timeout_seconds,
        strict_json=service_settings.strict_json,
    )
    if vector_store is not None:
        store = vector_store
    elif service_settings.vector_db == "memory":
        store = InMemoryVectorStore()
    elif service_settings.vector_db == "qdrant":
        store = QdrantVectorStore(
            base_url=service_settings.qdrant_base_url,
            collection_name=service_settings.qdrant_collection,
            timeout_seconds=service_settings.request_timeout_seconds,
            api_key=service_settings.qdrant_api_key,
        )
    else:
        raise ValueError(f"Unsupported VECTOR_DB backend: {service_settings.vector_db}")
    return PathologyAIService(
        settings=service_settings,
        embedding_client=embedding,
        reranker_client=reranker,
        llm_client=llm,
        vector_store=store,
    )
