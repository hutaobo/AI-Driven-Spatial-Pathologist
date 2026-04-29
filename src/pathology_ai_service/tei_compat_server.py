from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any, Protocol


class TEICompatError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def health(self) -> dict[str, Any]: ...


class RerankerBackend(Protocol):
    def rerank(self, *, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]: ...

    def health(self) -> dict[str, Any]: ...


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_length).decode("utf-8") if content_length else "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TEICompatError(f"Request body is not valid JSON: {exc}", status_code=400) from exc
    if not isinstance(payload, dict):
        raise TEICompatError("Request body must be a JSON object.", status_code=400)
    return payload


def _normalize_inputs(raw: Any) -> tuple[list[str], bool]:
    if isinstance(raw, str):
        return [raw], True
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return raw, False
    raise TEICompatError("'inputs' must be a string or a list of strings.", status_code=400)


class TransformerEmbeddingBackend:
    def __init__(self, *, model_id: str, device: str, max_length: int, batch_size: int, dtype: str) -> None:
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._functional = functional
        model_kwargs: dict[str, Any] = {}
        if dtype == "auto" and device.startswith("cuda"):
            model_kwargs["torch_dtype"] = "auto"
        elif dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id, **model_kwargs).to(device)
        self._model.eval()

    def _mean_pool(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        input_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = self._torch.sum(last_hidden_state * input_mask, dim=1)
        counts = self._torch.clamp(input_mask.sum(dim=1), min=1e-9)
        return summed / counts

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self._model(**encoded)
                pooled = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
                normalized = self._functional.normalize(pooled, p=2, dim=1)
                vectors.extend(normalized.float().cpu().tolist())
        return [[float(value) for value in row] for row in vectors]

    def health(self) -> dict[str, Any]:
        return {"ok": True, "mode": "embed", "model": self.model_id, "device": self.device}


class TransformerRerankerBackend:
    def __init__(self, *, model_id: str, device: str, max_length: int, batch_size: int, dtype: str) -> None:
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        model_kwargs: dict[str, Any] = {}
        if dtype == "auto" and device.startswith("cuda"):
            model_kwargs["torch_dtype"] = "auto"
        elif dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id, **model_kwargs).to(device)
        self._model.eval()

    def rerank(self, *, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        scored: list[tuple[int, float]] = []
        pairs = [(query, text) for text in texts]
        with self._torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                queries = [item[0] for item in batch]
                passages = [item[1] for item in batch]
                encoded = self._tokenizer(
                    queries,
                    passages,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self._model(**encoded).logits.reshape(-1).float().cpu().tolist()
                scored.extend((start + offset, float(score)) for offset, score in enumerate(logits))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_n]

    def health(self) -> dict[str, Any]:
        return {"ok": True, "mode": "rerank", "model": self.model_id, "device": self.device}


class TEICompatService:
    def __init__(
        self,
        *,
        mode: str,
        model_id: str,
        embedding_backend: EmbeddingBackend | None = None,
        reranker_backend: RerankerBackend | None = None,
    ) -> None:
        if mode not in {"embed", "rerank"}:
            raise ValueError("mode must be either 'embed' or 'rerank'.")
        self.mode = mode
        self.model_id = model_id
        self._embedding_backend = embedding_backend
        self._reranker_backend = reranker_backend

    def health(self) -> dict[str, Any]:
        backend = self._embedding_backend if self.mode == "embed" else self._reranker_backend
        if backend is None:
            return {"ok": False, "mode": self.mode, "model": self.model_id, "error": "backend is not configured"}
        payload = backend.health()
        payload.setdefault("ok", True)
        payload.setdefault("mode", self.mode)
        payload.setdefault("model", self.model_id)
        return payload

    def embed(self, payload: dict[str, Any]) -> list[float] | list[list[float]]:
        if self.mode != "embed" or self._embedding_backend is None:
            raise TEICompatError("This server was not started in embedding mode.", status_code=404)
        texts, single_input = _normalize_inputs(payload.get("inputs"))
        vectors = self._embedding_backend.embed(texts)
        if single_input:
            return vectors[0] if vectors else []
        return vectors

    def rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.mode != "rerank" or self._reranker_backend is None:
            raise TEICompatError("This server was not started in reranker mode.", status_code=404)
        query = payload.get("query")
        texts = payload.get("texts")
        if not isinstance(query, str) or not query:
            raise TEICompatError("'query' must be a non-empty string.", status_code=400)
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise TEICompatError("'texts' must be a list of strings.", status_code=400)
        top_n = int(payload.get("top_n") or payload.get("top_k") or len(texts))
        top_n = max(0, min(top_n, len(texts)))
        ranked = self._reranker_backend.rerank(query=query, texts=texts, top_n=top_n)
        return {"results": [{"index": index, "score": score} for index, score in ranked]}


def _handler_factory(service: TEICompatService) -> type[BaseHTTPRequestHandler]:
    class TEICompatRequestHandler(BaseHTTPRequestHandler):
        server_version = "pathology-ai-tei-compat/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def _write_json(self, *, status: int, payload: Any) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            try:
                if self.path.rstrip("/") == "/health":
                    self._write_json(status=HTTPStatus.OK, payload=service.health())
                    return
                self._write_json(status=HTTPStatus.NOT_FOUND, payload={"error": "not_found", "message": f"Unknown route: {self.path}"})
            except Exception as exc:  # pragma: no cover - defensive path
                self._write_json(status=HTTPStatus.INTERNAL_SERVER_ERROR, payload={"error": "internal_error", "message": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            try:
                normalized = self.path.rstrip("/") or "/"
                payload = _read_json_body(self)
                if normalized == "/embed":
                    self._write_json(status=HTTPStatus.OK, payload=service.embed(payload))
                    return
                if normalized == "/rerank":
                    self._write_json(status=HTTPStatus.OK, payload=service.rerank(payload))
                    return
                raise TEICompatError(f"Unknown route: {self.path}", status_code=404)
            except TEICompatError as exc:
                self._write_json(status=exc.status_code, payload={"error": "service_error", "message": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive path
                self._write_json(status=HTTPStatus.INTERNAL_SERVER_ERROR, payload={"error": "internal_error", "message": str(exc)})

    return TEICompatRequestHandler


def serve(service: TEICompatService, *, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), _handler_factory(service))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_service_from_args(args: argparse.Namespace) -> TEICompatService:
    device = _resolve_device(args.device)
    if args.mode == "embed":
        backend = TransformerEmbeddingBackend(
            model_id=args.model_id,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            dtype=args.dtype,
        )
        return TEICompatService(mode="embed", model_id=args.model_id, embedding_backend=backend)
    backend = TransformerRerankerBackend(
        model_id=args.model_id,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    return TEICompatService(mode="rerank", model_id=args.model_id, reranker_backend=backend)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal TEI-compatible embedding or reranking service.")
    parser.add_argument("--mode", required=True, choices=["embed", "rerank"], help="Service mode to expose.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id to load.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the HTTP server to.")
    parser.add_argument("--port", required=True, type=int, help="Port to bind the HTTP server to.")
    parser.add_argument("--device", default="auto", help="Torch device, for example 'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Torch dtype for model weights.")
    parser.add_argument("--max-length", default=8192, type=int, help="Maximum token length for tokenizer truncation.")
    parser.add_argument("--batch-size", default=4, type=int, help="Batch size for model inference.")
    args = parser.parse_args()
    service = build_service_from_args(args)
    serve(service, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
