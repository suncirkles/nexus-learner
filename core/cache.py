"""
core/cache.py
-------------
Semantic cache for LLM agent calls.

Stores structured outputs keyed by a semantic embedding of the input prompt.
Uses an in-memory vector index for cosine-similarity matching.

Contract
--------
  cache = get_cache()
  result = cache.lookup(prompt_str, SchemaClass)        # BaseModel | None
  cache.store(prompt_str, SchemaClass, result_obj, model_name)
  stats  = cache.stats()   # {"hits": N, "misses": N, "stores": N}
  cache.clear()            # test helper — resets entries and stats

Embedding backends (tried in order)
------------------------------------
  1. sentence-transformers all-MiniLM-L6-v2  (no API key, 384-dim)
  2. OpenAI text-embedding-3-small           (requires OPENAI_API_KEY)

When neither backend is available the cache is a transparent no-op:
  lookup() always returns None, store() is silent.
"""

import logging
import math
import threading
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Cosine similarity floor for a cache hit (tune via SemanticCache constructor)
_DEFAULT_THRESHOLD: float = 0.92

# Module-level singleton + its lock
_cache_instance: Optional["SemanticCache"] = None
_singleton_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal entry
# ---------------------------------------------------------------------------

class _Entry:
    __slots__ = ("prompt", "embedding", "result_json", "schema_name", "model")

    def __init__(
        self,
        prompt: str,
        embedding: list[float],
        result_json: str,
        schema_name: str,
        model: str,
    ) -> None:
        self.prompt = prompt
        self.embedding = embedding
        self.result_json = result_json
        self.schema_name = schema_name
        self.model = model


# ---------------------------------------------------------------------------
# Cache class
# ---------------------------------------------------------------------------

class SemanticCache:
    """Thread-safe in-memory semantic cache backed by cosine similarity."""

    def __init__(self, similarity_threshold: float = _DEFAULT_THRESHOLD) -> None:
        self._threshold = similarity_threshold
        self._entries: list[_Entry] = []
        self._stats: dict[str, int] = {"hits": 0, "misses": 0, "stores": 0}
        self._lock = threading.Lock()
        self._embed_fn = _make_embed_fn()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, prompt: str, schema: type[BaseModel]) -> Optional[BaseModel]:
        """Return a cached result if a semantically similar entry exists, else None."""
        if self._embed_fn is None:
            with self._lock:
                self._stats["misses"] += 1
            return None

        try:
            emb = self._embed_fn(prompt)
        except Exception as exc:
            logger.debug("Cache embed failed during lookup: %s", exc)
            with self._lock:
                self._stats["misses"] += 1
            return None

        schema_name = schema.__name__

        with self._lock:
            best_sim: float = 0.0
            best_entry: Optional[_Entry] = None

            for entry in self._entries:
                if entry.schema_name != schema_name:
                    continue
                sim = _cosine(emb, entry.embedding)
                if sim > best_sim:
                    best_sim, best_entry = sim, entry

            if best_entry is not None and best_sim >= self._threshold:
                try:
                    result = schema.model_validate_json(best_entry.result_json)
                    self._stats["hits"] += 1
                    logger.debug(
                        "Cache HIT schema=%s sim=%.3f prompt=%.60r",
                        schema_name, best_sim, prompt,
                    )
                    return result
                except Exception as exc:
                    logger.warning("Cache deserialisation failed for %s: %s", schema_name, exc)

            self._stats["misses"] += 1
            return None

    def store(
        self,
        prompt: str,
        schema: type[BaseModel],
        result: BaseModel,
        model: str = "unknown",
    ) -> None:
        """Store a result keyed by the prompt's embedding. Silent on any error."""
        if self._embed_fn is None:
            return
        try:
            emb = self._embed_fn(prompt)
            result_json = result.model_dump_json()
            entry = _Entry(
                prompt=prompt,
                embedding=emb,
                result_json=result_json,
                schema_name=schema.__name__,
                model=model,
            )
            with self._lock:
                self._entries.append(entry)
                self._stats["stores"] += 1
            logger.debug("Cache STORE schema=%s model=%s", schema.__name__, model)
        except Exception as exc:
            logger.warning("Cache store failed for %s: %s", schema.__name__, exc)

    def stats(self) -> dict[str, int]:
        """Return a snapshot of hit/miss/store counts."""
        with self._lock:
            return dict(self._stats)

    def clear(self) -> None:
        """Remove all entries and reset stats. Primarily for tests."""
        with self._lock:
            self._entries.clear()
            self._stats = {"hits": 0, "misses": 0, "stores": 0}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# ---------------------------------------------------------------------------
# Embedding backend
# ---------------------------------------------------------------------------

def _make_embed_fn():
    """
    Return a callable(text: str) -> list[float], or None if no backend found.
    Tries sentence-transformers first (no key), then OpenAI (key required).
    """
    # 1. sentence-transformers (preferred — no API key, fully offline)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _model = SentenceTransformer("all-MiniLM-L6-v2")

        def _st_embed(text: str) -> list[float]:
            return _model.encode(text, normalize_embeddings=True).tolist()

        logger.debug("Semantic cache: using SentenceTransformer all-MiniLM-L6-v2")
        return _st_embed
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("SentenceTransformer init failed: %s", exc)

    # 2. OpenAI text-embedding-3-small (requires valid key)
    try:
        from langchain_openai import OpenAIEmbeddings  # type: ignore
        from core.config import settings  # local import to avoid circular at module load

        if settings.OPENAI_API_KEY:
            _oai = OpenAIEmbeddings(
                api_key=settings.OPENAI_API_KEY,
                model="text-embedding-3-small",
            )

            def _oai_embed(text: str) -> list[float]:
                return _oai.embed_query(text)

            logger.debug("Semantic cache: using OpenAI text-embedding-3-small")
            return _oai_embed
    except Exception as exc:
        logger.debug("OpenAI embedding backend unavailable: %s", exc)

    logger.warning(
        "Semantic cache: no embedding backend available "
        "(install sentence-transformers or set OPENAI_API_KEY). "
        "Cache will be a no-op."
    )
    return None


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (handles unnormalised inputs)."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

def get_cache() -> SemanticCache:
    """Return (or lazily create) the module-level SemanticCache singleton."""
    global _cache_instance
    with _singleton_lock:
        if _cache_instance is None:
            _cache_instance = SemanticCache()
        return _cache_instance
