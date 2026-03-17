"""
core/cache.py
-------------
Semantic cache for LLM structured output, backed by Qdrant and
all-MiniLM-L6-v2 (local, sentence-transformers, 384-dim).

Usage
-----
    from core.cache import get_cache, init_semantic_cache

    cache = get_cache()
    cached = cache.lookup(prompt, MySchema)
    if cached is None:
        result = llm_call(...)
        cache.store(prompt, MySchema, result, model_name)

Graceful degradation
--------------------
If Qdrant is unavailable or sentence-transformers is not installed,
``get_cache()`` returns a ``_NullCache`` — all methods are no-ops.
``generate_structured()`` never fails due to cache errors.
"""

from __future__ import annotations

import importlib
import json
import logging
import threading
import time
import uuid
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared embedder loader
# ---------------------------------------------------------------------------

def _load_embedder(label: str):
    """Load all-MiniLM-L6-v2 via langchain_huggingface or langchain_community fallback.

    Returns the embedder instance, or None if sentence-transformers is not installed.
    Suppresses noisy import warnings from both packages.
    """
    import warnings

    for _import_path in [
        "langchain_huggingface.HuggingFaceEmbeddings",
        "langchain_community.embeddings.HuggingFaceEmbeddings",
    ]:
        try:
            module_path, cls_name = _import_path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            HFEmbeddings = getattr(mod, cls_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                embedder = HFEmbeddings(model_name="all-MiniLM-L6-v2")
            logger.debug("%s: loaded embedder via %s", label, _import_path)
            return embedder
        except Exception as exc:
            logger.debug("%s: embedder import failed (%s): %s", label, _import_path, exc)

    logger.warning(
        "%s: sentence-transformers not available — "
        "install with: pip install sentence-transformers",
        label,
    )
    return None


# ---------------------------------------------------------------------------
# Null cache — returned when Qdrant / embeddings are unavailable
# ---------------------------------------------------------------------------

class _NullCache:
    """No-op cache returned when the real cache cannot be initialised."""

    def __init__(self) -> None:
        self._hits = 0
        self._misses = 0
        self._stores = 0

    def lookup(self, prompt: str, schema: type) -> None:  # type: ignore[return]
        return None

    def store(self, prompt: str, schema: type, result: "BaseModel", model_name: str = "") -> None:
        pass

    def clear(self) -> None:
        pass

    def stats(self) -> dict:
        return {
            "enabled": False,
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "hit_rate": 0.0,
            "collection": None,
            "threshold": None,
        }


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------

class SemanticCache:
    """Qdrant-backed semantic cache for Pydantic structured-output results.

    Cache key: ``f"{schema.__name__}::{prompt}"`` — schema name is prepended
    before embedding so that different schemas never collide, even if their
    prompts are identical.  A secondary Qdrant payload filter on ``schema_name``
    provides belt-and-suspenders correctness at query time.
    """

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection: str,
        threshold: float,
        ttl_seconds: int,
        max_entries: int,
    ) -> None:
        self._collection = collection
        self._threshold = threshold
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._ready = False

        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._counter_lock = threading.Lock()

        # --- Load embedder ---
        self._embedder = _load_embedder("SemanticCache")
        if self._embedder is None:
            return

        # --- Connect to Qdrant ---
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key or None,
                timeout=5,
            )
            # Health check
            self._client.get_collections()

            # Create collection if absent
            existing = {c.name for c in self._client.get_collections().collections}
            if self._collection not in existing:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
                logger.info("SemanticCache: created collection '%s'", self._collection)

            self._ready = True
            logger.info(
                "SemanticCache: ready (collection=%s, threshold=%.2f)",
                self._collection,
                self._threshold,
            )
        except Exception as exc:
            logger.warning("SemanticCache: Qdrant unavailable — %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, prompt: str, schema: type) -> Optional["BaseModel"]:
        """Return a cached result or None (cache miss)."""
        if not self._ready:
            return None

        key = f"{schema.__name__}::{prompt}"
        try:
            vector = self._embed(key)
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            hits = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=1,
                score_threshold=self._threshold,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="schema_name",
                            match=MatchValue(value=schema.__name__),
                        )
                    ]
                ),
                with_payload=True,
            )

            if not hits:
                self._misses += 1
                return None

            hit = hits[0]
            payload = hit.payload or {}

            # Client-side TTL eviction
            if self._ttl_seconds > 0:
                created_at = payload.get("created_at", 0)
                if (time.time() - created_at) > self._ttl_seconds:
                    self._misses += 1
                    # Lazy delete stale entry
                    try:
                        from qdrant_client.models import PointIdsList
                        self._client.delete(
                            collection_name=self._collection,
                            points_selector=PointIdsList(points=[hit.id]),
                        )
                    except Exception:
                        pass
                    return None

            result_json = payload.get("result_json", "")
            if not result_json:
                self._misses += 1
                return None

            result = schema.model_validate(json.loads(result_json))  # type: ignore[attr-defined]
            self._hits += 1

            # Fire-and-forget hit count update
            try:
                current_hits = payload.get("hit_count", 0) + 1
                self._client.set_payload(
                    collection_name=self._collection,
                    payload={"hit_count": current_hits},
                    points=[hit.id],
                )
            except Exception:
                pass

            return result

        except Exception as exc:
            logger.debug("SemanticCache.lookup error: %s", exc)
            self._misses += 1
            return None

    def store(
        self,
        prompt: str,
        schema: type,
        result: "BaseModel",
        model_name: str = "",
    ) -> None:
        """Embed and upsert a result into the cache."""
        if not self._ready:
            return

        key = f"{schema.__name__}::{prompt}"
        try:
            vector = self._embed(key)
            from qdrant_client.models import PointStruct

            point_id = str(uuid.uuid4())
            payload = {
                "schema_name": schema.__name__,
                "prompt_text": prompt[:500],
                "result_json": result.model_dump_json(),  # type: ignore[attr-defined]
                "model_name": model_name,
                "created_at": time.time(),
                "hit_count": 0,
            }
            self._client.upsert(
                collection_name=self._collection,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
            )
            self._stores += 1
            self._evict_if_needed()
        except Exception as exc:
            logger.debug("SemanticCache.store error: %s", exc)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "enabled": self._ready,
            "hits": self._hits,
            "misses": self._misses,
            "stores": self._stores,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            "collection": self._collection,
            "threshold": self._threshold,
        }

    def clear(self) -> None:
        """Drop and recreate the cache collection (used in tests and demo)."""
        if not self._ready:
            return
        try:
            from qdrant_client.models import Distance, VectorParams

            self._client.delete_collection(self._collection)
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            self._hits = 0
            self._misses = 0
            self._stores = 0
            logger.info("SemanticCache: cleared collection '%s'", self._collection)
        except Exception as exc:
            logger.warning("SemanticCache.clear error: %s", exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    def _evict_if_needed(self) -> None:
        """Evict oldest entries when the collection exceeds max_entries."""
        try:
            count_result = self._client.count(collection_name=self._collection, exact=True)
            if count_result.count <= self._max_entries:
                return

            excess = count_result.count - self._max_entries
            # Scroll oldest entries by created_at (ascending)
            scroll_result, _ = self._client.scroll(
                collection_name=self._collection,
                limit=excess,
                with_payload=["created_at"],
                order_by="created_at",  # ascending; oldest first
            )
            ids_to_delete = [p.id for p in scroll_result]
            if ids_to_delete:
                from qdrant_client.models import PointIdsList
                self._client.delete(
                    collection_name=self._collection,
                    points_selector=PointIdsList(points=ids_to_delete),
                )
                logger.debug("SemanticCache: evicted %d old entries", len(ids_to_delete))
        except Exception as exc:
            logger.debug("SemanticCache._evict_if_needed error: %s", exc)


# ---------------------------------------------------------------------------
# Redis-backed semantic cache
# ---------------------------------------------------------------------------

class RedisSemanticCache:
    """Redis-backed semantic cache for Pydantic structured-output results.

    Cache key format: ``nexus_cache:{schema_name}:{uuid}`` stored as a Redis
    Hash.  Embeddings use the same local all-MiniLM-L6-v2 model as
    ``SemanticCache``; similarity is computed client-side with numpy cosine.

    Graceful degradation: if the ``redis`` package is missing or the server
    is unreachable, ``_ready`` is set to ``False`` and every method is a no-op.
    """

    def __init__(
        self,
        redis_url: str,
        redis_db: int,
        threshold: float,
        ttl_seconds: int,
        max_entries: int,
    ) -> None:
        self._redis_url = redis_url
        self._redis_db = redis_db
        self._threshold = threshold
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._ready = False

        self._hits = 0
        self._misses = 0
        self._stores = 0

        # --- Load embedder ---
        self._embedder = _load_embedder("RedisSemanticCache")
        if self._embedder is None:
            return

        # --- Connect to Redis ---
        try:
            import redis as redis_pkg
            import numpy  # noqa: F401 — validate numpy is available for vector ops

            self._redis = redis_pkg.Redis.from_url(
                self._redis_url,
                db=self._redis_db,
                socket_connect_timeout=5,
                decode_responses=False,  # raw bytes needed for numpy serialisation
            )
            # Health check
            self._redis.ping()

            self._ready = True
            logger.info(
                "RedisSemanticCache: ready (url=%s, db=%d, threshold=%.2f)",
                self._redis_url,
                self._redis_db,
                self._threshold,
            )
        except ImportError:
            logger.warning(
                "RedisSemanticCache: 'redis' package not installed — "
                "install with: pip install redis"
            )
        except Exception as exc:
            logger.warning("RedisSemanticCache: Redis unavailable — %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, prompt: str, schema: type) -> Optional["BaseModel"]:
        """Return a cached result or None (cache miss)."""
        if not self._ready:
            return None

        import numpy as np

        key_prefix = f"nexus_cache:{schema.__name__}:*"
        embed_key = f"{schema.__name__}::{prompt}"
        try:
            query_vector = np.array(self._embed(embed_key), dtype=np.float32)

            best_score = -1.0
            best_hash_key: Optional[bytes] = None
            best_payload: Optional[dict] = None

            # Scan all keys for this schema
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match=key_prefix, count=100)
                for raw_key in keys:
                    entry = self._redis.hgetall(raw_key)
                    if not entry:
                        continue
                    emb_bytes = entry.get(b"embedding")
                    if not emb_bytes:
                        continue
                    stored_vector = np.frombuffer(emb_bytes, dtype=np.float32)
                    score = float(
                        np.dot(query_vector, stored_vector)
                        / (np.linalg.norm(query_vector) * np.linalg.norm(stored_vector) + 1e-10)
                    )
                    if score > best_score:
                        best_score = score
                        best_hash_key = raw_key
                        best_payload = entry
                if cursor == 0:
                    break

            if best_score < self._threshold or best_hash_key is None or best_payload is None:
                self._misses += 1
                return None

            # Client-side TTL check (belt-and-suspenders)
            if self._ttl_seconds > 0:
                created_at = float(best_payload.get(b"created_at", 0))
                if (time.time() - created_at) > self._ttl_seconds:
                    self._misses += 1
                    try:
                        self._redis.delete(best_hash_key)
                    except Exception:
                        pass
                    return None

            result_json = best_payload.get(b"result_json", b"").decode("utf-8")
            if not result_json:
                self._misses += 1
                return None

            result = schema.model_validate(json.loads(result_json))  # type: ignore[attr-defined]
            self._hits += 1

            # Fire-and-forget hit count increment
            try:
                current_hits = int(best_payload.get(b"hit_count", b"0")) + 1
                self._redis.hset(best_hash_key, "hit_count", current_hits)
            except Exception:
                pass

            return result

        except Exception as exc:
            logger.debug("RedisSemanticCache.lookup error: %s", exc)
            self._misses += 1
            return None

    def store(
        self,
        prompt: str,
        schema: type,
        result: "BaseModel",
        model_name: str = "",
    ) -> None:
        """Embed and store a result in Redis."""
        if not self._ready:
            return

        import numpy as np

        embed_key = f"{schema.__name__}::{prompt}"
        try:
            vector = np.array(self._embed(embed_key), dtype=np.float32)
            point_id = str(uuid.uuid4())
            redis_key = f"nexus_cache:{schema.__name__}:{point_id}"

            mapping = {
                "schema_name": schema.__name__,
                "prompt_text": prompt[:500],
                "result_json": result.model_dump_json(),  # type: ignore[attr-defined]
                "model_name": model_name,
                "created_at": str(time.time()),
                "hit_count": "0",
                "embedding": vector.tobytes(),
            }
            self._redis.hset(redis_key, mapping=mapping)
            if self._ttl_seconds > 0:
                self._redis.expire(redis_key, self._ttl_seconds)

            self._stores += 1
            self._evict_if_needed()
        except Exception as exc:
            logger.debug("RedisSemanticCache.store error: %s", exc)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "enabled": self._ready,
            "hits": self._hits,
            "misses": self._misses,
            "stores": self._stores,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            "collection": f"redis:db{self._redis_db}",
            "threshold": self._threshold,
        }

    def clear(self) -> None:
        """Delete all ``nexus_cache:*`` keys from Redis (used in tests and demo)."""
        if not self._ready:
            return
        try:
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match="nexus_cache:*", count=200)
                if keys:
                    self._redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            self._hits = 0
            self._misses = 0
            self._stores = 0
            logger.info("RedisSemanticCache: cleared %d keys", deleted)
        except Exception as exc:
            logger.warning("RedisSemanticCache.clear error: %s", exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    def _evict_if_needed(self) -> None:
        """Delete oldest entries (by created_at) when the key count exceeds max_entries."""
        try:
            # Count all nexus_cache keys — SCAN is the only option without Redis Search.
            all_keys: list[bytes] = []
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match="nexus_cache:*", count=200)
                all_keys.extend(keys)
                if cursor == 0:
                    break

            excess = len(all_keys) - self._max_entries
            if excess <= 0:
                return

            # Fetch created_at for each key and delete the oldest `excess` entries.
            key_ages: list[tuple[float, bytes]] = []
            for k in all_keys:
                ts_bytes = self._redis.hget(k, "created_at")
                ts = float(ts_bytes) if ts_bytes else 0.0
                key_ages.append((ts, k))
            key_ages.sort()  # ascending: oldest first

            to_delete = [k for _, k in key_ages[:excess]]
            if to_delete:
                self._redis.delete(*to_delete)
                logger.debug("RedisSemanticCache: evicted %d old entries", len(to_delete))
        except Exception as exc:
            logger.debug("RedisSemanticCache._evict_if_needed error: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton (thread-safe)
# ---------------------------------------------------------------------------

_MODULE_CACHE: SemanticCache | RedisSemanticCache | _NullCache | None = None
_CACHE_LOCK = threading.Lock()


def init_semantic_cache() -> SemanticCache | RedisSemanticCache | _NullCache:
    """Initialise (or return) the module-level cache singleton.

    Reads settings from ``core.config.settings``.  If settings cannot be
    imported (e.g. standalone use outside the project), falls back to
    environment variables directly.  Returns a ``_NullCache`` if the cache
    is disabled or fails to initialise.

    Backend selection is controlled by ``SEMANTIC_CACHE_BACKEND``:
    - ``"qdrant"`` (default) — uses ``SemanticCache``
    - ``"redis"``            — uses ``RedisSemanticCache``
    """
    global _MODULE_CACHE
    with _CACHE_LOCK:
        if _MODULE_CACHE is not None:
            return _MODULE_CACHE

        # Read settings
        enabled = True
        backend = "qdrant"
        qdrant_url = "http://localhost:6333"
        qdrant_api_key = ""
        collection = "nexus_semantic_cache"
        threshold = 0.92
        ttl_seconds = 86400
        max_entries = 10000
        redis_url = "redis://localhost:6379"
        redis_db = 1

        try:
            from core.config import settings  # type: ignore[import]
            enabled = settings.SEMANTIC_CACHE_ENABLED
            backend = settings.SEMANTIC_CACHE_BACKEND
            qdrant_url = settings.QDRANT_URL
            qdrant_api_key = settings.QDRANT_API_KEY
            collection = settings.SEMANTIC_CACHE_COLLECTION
            threshold = settings.SEMANTIC_CACHE_THRESHOLD
            ttl_seconds = settings.SEMANTIC_CACHE_TTL_SECONDS
            max_entries = settings.SEMANTIC_CACHE_MAX_ENTRIES
            redis_url = settings.REDIS_URL
            redis_db = settings.REDIS_CACHE_DB
        except Exception:
            import os
            enabled = os.environ.get("SEMANTIC_CACHE_ENABLED", "true").lower() != "false"
            backend = os.environ.get("SEMANTIC_CACHE_BACKEND", backend)
            qdrant_url = os.environ.get("QDRANT_URL", qdrant_url)
            qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")
            redis_url = os.environ.get("REDIS_URL", redis_url)
            redis_db = int(os.environ.get("REDIS_CACHE_DB", redis_db))

        if not enabled:
            logger.info("SemanticCache: disabled via SEMANTIC_CACHE_ENABLED=false")
            _MODULE_CACHE = _NullCache()
            return _MODULE_CACHE

        if backend == "redis":
            cache: SemanticCache | RedisSemanticCache = RedisSemanticCache(
                redis_url=redis_url,
                redis_db=redis_db,
                threshold=threshold,
                ttl_seconds=ttl_seconds,
                max_entries=max_entries,
            )
        else:
            cache = SemanticCache(
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                collection=collection,
                threshold=threshold,
                ttl_seconds=ttl_seconds,
                max_entries=max_entries,
            )

        _MODULE_CACHE = cache if cache._ready else _NullCache()
        return _MODULE_CACHE


def get_cache() -> SemanticCache | RedisSemanticCache | _NullCache:
    """Return the module-level cache singleton, initialising it if needed."""
    if _MODULE_CACHE is None:
        return init_semantic_cache()
    return _MODULE_CACHE


def _reset_cache_singleton() -> None:
    """Reset the module singleton — for tests only."""
    global _MODULE_CACHE
    with _CACHE_LOCK:
        _MODULE_CACHE = None
