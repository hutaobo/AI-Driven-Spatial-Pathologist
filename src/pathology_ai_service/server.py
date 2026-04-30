from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any

from pydantic import ValidationError

from .config import ServiceSettings
from .core import PathologyAIService, ServiceError, build_service
from .models import (
    ClusterAnnotationRequest,
    DocumentInput,
    DocumentUpsertRequest,
    HEContourClassifyRequest,
    ReviewRequest,
    StructureMultimodalNamingRequest,
)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_length).decode("utf-8") if content_length else "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ServiceError(f"Request body is not valid JSON: {exc}", status_code=400) from exc
    if not isinstance(payload, dict):
        raise ServiceError("Request body must be a JSON object.", status_code=400)
    return payload


def _handler_factory(service: PathologyAIService) -> type[BaseHTTPRequestHandler]:
    class PathologyAIRequestHandler(BaseHTTPRequestHandler):
        server_version = "pathology-ai/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def _write_json(self, *, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _handle_review(self, *, review_type: str) -> None:
            payload = ReviewRequest.model_validate(_read_json_body(self))
            result = service.review(review_type=review_type, payload=payload)
            self._write_json(status=HTTPStatus.OK, payload=result.model_dump())

        def do_GET(self) -> None:  # noqa: N802
            try:
                if self.path.rstrip("/") == "/health":
                    self._write_json(status=HTTPStatus.OK, payload=service.health())
                    return
                self._write_json(
                    status=HTTPStatus.NOT_FOUND,
                    payload={"error": "not_found", "message": f"Unknown route: {self.path}"},
                )
            except ServiceError as exc:
                self._write_json(
                    status=exc.status_code,
                    payload={"error": "service_error", "message": str(exc)},
                )
            except Exception as exc:  # pragma: no cover - defensive path
                self._write_json(
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    payload={"error": "internal_error", "message": str(exc)},
                )

        def do_POST(self) -> None:  # noqa: N802
            try:
                normalized = self.path.rstrip("/") or "/"
                if normalized in {"/documents/upsert", "/v1/documents/upsert"}:
                    body = _read_json_body(self)
                    if "documents" not in body and "document_id" in body:
                        body = {"documents": [DocumentInput.model_validate(body).model_dump()]}
                    payload = DocumentUpsertRequest.model_validate(body)
                    result = service.upsert_documents(payload)
                    self._write_json(status=HTTPStatus.OK, payload=result.model_dump())
                    return
                if normalized in {"/annotations/cluster", "/v1/annotations/cluster"}:
                    payload = ClusterAnnotationRequest.model_validate(_read_json_body(self))
                    result = service.annotate_cluster(payload)
                    self._write_json(status=HTTPStatus.OK, payload=result.model_dump())
                    return
                if normalized in {"/he/contours/classify", "/v1/he/contours/classify"}:
                    payload = HEContourClassifyRequest.model_validate(_read_json_body(self))
                    result = service.classify_he_contours(payload)
                    self._write_json(status=HTTPStatus.OK, payload=result.model_dump())
                    return
                if normalized in {"/annotations/structure-multimodal", "/v1/annotations/structure-multimodal"}:
                    payload = StructureMultimodalNamingRequest.model_validate(_read_json_body(self))
                    result = service.name_structure_multimodal(payload)
                    self._write_json(status=HTTPStatus.OK, payload=result.model_dump())
                    return
                if normalized in {"/review", "/v1/review"}:
                    body = _read_json_body(self)
                    review_type = body.pop("review_type", None)
                    if review_type not in {"structure", "case"}:
                        raise ServiceError("review_type must be either 'structure' or 'case'.", status_code=400)
                    payload = ReviewRequest.model_validate(body)
                    result = service.review(review_type=review_type, payload=payload)
                    self._write_json(status=HTTPStatus.OK, payload=result.model_dump())
                    return
                if normalized in {"/reviews/structure", "/v1/reviews/structure"}:
                    self._handle_review(review_type="structure")
                    return
                if normalized in {"/reviews/case", "/v1/reviews/case"}:
                    self._handle_review(review_type="case")
                    return
                raise ServiceError(f"Unknown route: {self.path}", status_code=404)
            except ValidationError as exc:
                self._write_json(
                    status=HTTPStatus.BAD_REQUEST,
                    payload={"error": "validation_error", "message": str(exc)},
                )
            except ServiceError as exc:
                self._write_json(
                    status=exc.status_code,
                    payload={"error": "service_error", "message": str(exc)},
                )
            except Exception as exc:  # pragma: no cover - defensive path
                self._write_json(
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    payload={"error": "internal_error", "message": str(exc)},
                )

    return PathologyAIRequestHandler


def serve(service: PathologyAIService, *, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), _handler_factory(service))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    settings = ServiceSettings.from_env()
    parser = argparse.ArgumentParser(description="Run the local pathology AI review service.")
    parser.add_argument("--host", default=settings.host, help="Host to bind the HTTP server to.")
    parser.add_argument("--port", default=settings.port, type=int, help="Port to bind the HTTP server to.")
    args = parser.parse_args()
    service = build_service(settings)
    serve(service, host=args.host, port=args.port)
