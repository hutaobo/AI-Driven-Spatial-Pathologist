from __future__ import annotations

from dataclasses import dataclass
import os


def _parse_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value, got: {raw!r}")


def _parse_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _parse_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


@dataclass(slots=True)
class ServiceSettings:
    host: str = "0.0.0.0"
    port: int = 8000

    llm_base_url: str = "http://vllm:8000/v1"
    llm_model: str = "openai/gpt-oss-120b"

    embed_base_url: str = "http://embedder:80"
    embed_model: str = "BAAI/bge-m3"

    rerank_base_url: str = "http://reranker:80"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    vector_db: str = "qdrant"
    qdrant_base_url: str = "http://qdrant:6333"
    qdrant_collection: str = "pathology_reference"
    qdrant_api_key: str | None = None

    default_top_k: int = 6
    strict_json: bool = True
    request_timeout_seconds: float = 30.0
    max_chunk_chars: int = 1200
    chunk_overlap_chars: int = 120
    log_raw_prompts: bool = False

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        return cls(
            host=os.environ.get("PATHOLOGY_AI_HOST", "0.0.0.0"),
            port=_parse_int("PATHOLOGY_AI_PORT", 8000),
            llm_base_url=os.environ.get("LLM_BASE_URL", "http://vllm:8000/v1"),
            llm_model=os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
            embed_base_url=os.environ.get("EMBED_BASE_URL", "http://embedder:80"),
            embed_model=os.environ.get("EMBED_MODEL", "BAAI/bge-m3"),
            rerank_base_url=os.environ.get("RERANK_BASE_URL", "http://reranker:80"),
            rerank_model=os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            vector_db=os.environ.get("VECTOR_DB", "qdrant"),
            qdrant_base_url=os.environ.get("QDRANT_BASE_URL", "http://qdrant:6333"),
            qdrant_collection=os.environ.get("QDRANT_COLLECTION", "pathology_reference"),
            qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
            default_top_k=_parse_int("DEFAULT_TOP_K", 6),
            strict_json=_parse_bool("STRICT_JSON", True),
            request_timeout_seconds=_parse_float("REQUEST_TIMEOUT_SECONDS", 30.0),
            max_chunk_chars=_parse_int("MAX_CHUNK_CHARS", 1200),
            chunk_overlap_chars=_parse_int("CHUNK_OVERLAP_CHARS", 120),
            log_raw_prompts=_parse_bool("LOG_RAW_PROMPTS", False),
        )

    def public_dict(self) -> dict[str, object]:
        return {
            "host": self.host,
            "port": self.port,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "embed_base_url": self.embed_base_url,
            "embed_model": self.embed_model,
            "rerank_base_url": self.rerank_base_url,
            "rerank_model": self.rerank_model,
            "vector_db": self.vector_db,
            "qdrant_base_url": self.qdrant_base_url,
            "qdrant_collection": self.qdrant_collection,
            "default_top_k": self.default_top_k,
            "strict_json": self.strict_json,
            "max_chunk_chars": self.max_chunk_chars,
            "chunk_overlap_chars": self.chunk_overlap_chars,
        }
