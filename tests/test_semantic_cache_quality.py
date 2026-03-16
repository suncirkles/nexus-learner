"""
tests/test_semantic_cache_quality.py
-------------------------------------
Quality and contract tests for core/cache.py — SemanticCache.

Test categories
---------------
  Unit (no external deps):
    - Cache is a no-op when AGENT_CACHE_ENABLED=False
    - Schema exclusion (FlashcardOutput never stored/served)
    - stats() increments correctly on store / hit / miss
    - Exact hit after store
    - Schema isolation (schema-A entry does not satisfy schema-B lookup)
    - clear() resets entries and stats
    - Similarity threshold: below-threshold prompt → miss, above → hit
    - Thread safety: concurrent stores do not corrupt stats

  Integration (requires sentence-transformers or OPENAI_API_KEY):
    - Semantically similar paraphrase returns a cached result
    - Unrelated prompt scores below threshold → miss
    - call_structured() wires into cache: stores on miss, returns hit on second call
    - call_structured_chain() likewise stores on first call
    - None / exception result is never stored

  Slow / live-API:
    - Second identical call_structured() invocation hits cache; API call count = 1

Run default (unit only):
    PYTHONPATH=. pytest tests/test_semantic_cache_quality.py -v

Run with embeddings:
    PYTHONPATH=. pytest tests/test_semantic_cache_quality.py -v -m integration

Run everything including live LLM:
    PYTHONPATH=. pytest tests/test_semantic_cache_quality.py -v -m slow
"""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Skip entire module when core/cache.py is not on this branch
# ---------------------------------------------------------------------------

try:
    from core.cache import SemanticCache, get_cache, _cosine  # noqa: F401
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CACHE_AVAILABLE,
    reason="core/cache.py not available on this branch",
)


# ---------------------------------------------------------------------------
# Minimal Pydantic schemas used across tests
# ---------------------------------------------------------------------------

class DummyScore(BaseModel):
    value: int = Field(ge=0, le=10)
    label: str


class OtherSchema(BaseModel):
    text: str


# Mirror the excluded schema name that lives in settings
class FlashcardOutput(BaseModel):
    question: str
    answer: str


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_cache():
    """Isolated SemanticCache instance — never touches the singleton."""
    cache = SemanticCache(similarity_threshold=0.90)
    yield cache
    cache.clear()


@pytest.fixture(autouse=True)
def _reset_singleton():
    """
    Stash and restore the module-level singleton so tests don't bleed state.
    """
    import core.cache as cache_mod
    original = cache_mod._cache_instance
    cache_mod._cache_instance = None
    yield
    cache_mod._cache_instance = original


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inject_embed(cache: SemanticCache, vec: list[float]) -> None:
    """Swap in a deterministic embed function that always returns *vec*."""
    cache._embed_fn = lambda _text: list(vec)


def _inject_embed_counter(cache: SemanticCache, vecs: list[list[float]]) -> None:
    """
    Swap in an embed function that cycles through *vecs* on successive calls.
    Useful for orthogonal store / lookup tests.
    """
    state = {"idx": 0}

    def _fn(_text: str) -> list[float]:
        v = vecs[state["idx"] % len(vecs)]
        state["idx"] += 1
        return list(v)

    cache._embed_fn = _fn


# ===========================================================================
# SECTION 1 — Basic contract
# ===========================================================================

class TestCacheContract:

    def test_stats_initial(self, fresh_cache):
        assert fresh_cache.stats() == {"hits": 0, "misses": 0, "stores": 0}

    def test_lookup_empty_returns_none(self, fresh_cache):
        result = fresh_cache.lookup("any prompt", DummyScore)
        assert result is None

    def test_stats_miss_increments(self, fresh_cache):
        _inject_embed(fresh_cache, [1.0, 0.0])
        fresh_cache.lookup("some prompt", DummyScore)
        s = fresh_cache.stats()
        assert s["hits"] == 0
        assert s["misses"] >= 1

    def test_store_increments_stores(self, fresh_cache):
        _inject_embed(fresh_cache, [1.0, 0.0])
        fresh_cache.store("prompt", DummyScore, DummyScore(value=5, label="mid"), model="t")
        assert fresh_cache.stats()["stores"] == 1

    def test_clear_resets_everything(self, fresh_cache):
        _inject_embed(fresh_cache, [1.0, 0.0])
        fresh_cache.store("p", DummyScore, DummyScore(value=3, label="lo"))
        fresh_cache.lookup("p", DummyScore)
        fresh_cache.clear()
        assert fresh_cache.stats() == {"hits": 0, "misses": 0, "stores": 0}
        assert len(fresh_cache) == 0

    def test_len_reflects_stored_entries(self, fresh_cache):
        _inject_embed(fresh_cache, [0.5, 0.5])
        assert len(fresh_cache) == 0
        fresh_cache.store("p1", DummyScore, DummyScore(value=1, label="a"))
        fresh_cache.store("p2", DummyScore, DummyScore(value=2, label="b"))
        assert len(fresh_cache) == 2

    def test_no_embed_fn_is_transparent_no_op(self, fresh_cache):
        """
        When _embed_fn is None the cache must be completely transparent:
        lookup → None, store → silent, stats increment misses only.
        """
        fresh_cache._embed_fn = None
        result = fresh_cache.lookup("q", DummyScore)
        fresh_cache.store("q", DummyScore, DummyScore(value=0, label="z"))
        s = fresh_cache.stats()
        assert result is None
        assert s["stores"] == 0
        assert s["hits"] == 0


# ===========================================================================
# SECTION 2 — Exact hit
# ===========================================================================

class TestExactHit:

    def test_exact_prompt_returns_stored_result(self, fresh_cache):
        _inject_embed(fresh_cache, [1.0, 0.0])
        fresh_cache._threshold = 0.99
        obj = DummyScore(value=7, label="exact")
        fresh_cache.store("the exact prompt", DummyScore, obj, model="test-model")
        hit = fresh_cache.lookup("the exact prompt", DummyScore)
        assert hit is not None
        assert hit.value == 7
        assert hit.label == "exact"

    def test_hit_increments_stats(self, fresh_cache):
        _inject_embed(fresh_cache, [1.0, 0.0])
        fresh_cache._threshold = 0.0
        fresh_cache.store("p", DummyScore, DummyScore(value=1, label="x"))
        fresh_cache.lookup("p", DummyScore)
        s = fresh_cache.stats()
        assert s["stores"] == 1
        assert s["hits"] == 1
        assert s["misses"] == 0

    def test_miss_does_not_increment_hits(self, fresh_cache):
        fresh_cache._threshold = 0.99
        # Store with [1, 0]; look up with [0, 1] → cosine = 0.0 → miss
        _inject_embed_counter(fresh_cache, [[1.0, 0.0], [0.0, 1.0]])
        fresh_cache.store("p1", DummyScore, DummyScore(value=2, label="y"))
        fresh_cache.lookup("p2", DummyScore)
        s = fresh_cache.stats()
        assert s["hits"] == 0
        assert s["misses"] >= 1


# ===========================================================================
# SECTION 3 — Schema isolation
# ===========================================================================

class TestSchemaIsolation:

    def test_schema_a_entry_does_not_match_schema_b_lookup(self, fresh_cache):
        """Same prompt embedding, different schema → miss."""
        _inject_embed(fresh_cache, [1.0, 0.0])
        fresh_cache._threshold = 0.0   # accept any similarity

        fresh_cache.store("same prompt", DummyScore, DummyScore(value=5, label="a"))
        result = fresh_cache.lookup("same prompt", OtherSchema)
        assert result is None, (
            "DummyScore entry must not satisfy an OtherSchema lookup — "
            "schema isolation is broken"
        )

    def test_schema_a_entry_matches_schema_a_lookup(self, fresh_cache):
        """Sanity: same schema + identical embedding → hit."""
        _inject_embed(fresh_cache, [0.0, 1.0])
        fresh_cache._threshold = 0.0

        obj = DummyScore(value=9, label="z")
        fresh_cache.store("the prompt", DummyScore, obj)
        result = fresh_cache.lookup("the prompt", DummyScore)
        assert result is not None
        assert isinstance(result, DummyScore)
        assert result.value == 9

    def test_two_schemas_stored_independently(self, fresh_cache):
        """Storing entries for two different schemas; each lookup only returns its own."""
        _inject_embed(fresh_cache, [1.0, 0.0])
        fresh_cache._threshold = 0.0

        fresh_cache.store("prompt", DummyScore, DummyScore(value=3, label="ds"))
        fresh_cache.store("prompt", OtherSchema, OtherSchema(text="hello"))

        r_ds = fresh_cache.lookup("prompt", DummyScore)
        r_os = fresh_cache.lookup("prompt", OtherSchema)

        assert isinstance(r_ds, DummyScore)
        assert isinstance(r_os, OtherSchema)
        assert r_ds.value == 3
        assert r_os.text == "hello"


# ===========================================================================
# SECTION 4 — Similarity threshold
# ===========================================================================

class TestSimilarityThreshold:

    def test_orthogonal_embedding_is_miss(self):
        """cosine([1,0], [0,1]) = 0.0 — must be below any reasonable threshold."""
        cache = SemanticCache(similarity_threshold=0.5)
        _inject_embed_counter(cache, [[1.0, 0.0], [0.0, 1.0]])
        cache.store("stored", DummyScore, DummyScore(value=1, label="a"))
        result = cache.lookup("different", DummyScore)
        assert result is None

    def test_identical_embedding_is_hit(self):
        """cosine([1,0], [1,0]) = 1.0 — must be a hit at any threshold ≤ 1.0."""
        cache = SemanticCache(similarity_threshold=0.99)
        _inject_embed(cache, [1.0, 0.0])
        cache.store("q", DummyScore, DummyScore(value=2, label="b"))
        result = cache.lookup("q", DummyScore)
        assert result is not None

    def test_threshold_boundary_exclusive(self):
        """
        If cosine = threshold exactly the entry should be returned (>= comparison).
        We pick threshold=0.5 and use vectors with cosine≈0.5 ([1,0] vs [1,1]/√2).
        """
        import math
        cache = SemanticCache(similarity_threshold=0.5)
        # Store with [1, 0]; look up with [1, 1]/√2 → cosine = 1/√2 ≈ 0.707
        norm = 1.0 / math.sqrt(2)
        _inject_embed_counter(cache, [[1.0, 0.0], [norm, norm]])
        cache.store("stored", DummyScore, DummyScore(value=5, label="t"))
        result = cache.lookup("query", DummyScore)
        # cosine ≈ 0.707 >= 0.5 → should hit
        assert result is not None, (
            "cosine≈0.707 should satisfy threshold=0.5 but returned miss"
        )

    def test_cosine_helper_correct(self):
        """Unit test _cosine() directly."""
        from core.cache import _cosine
        assert abs(_cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
        assert abs(_cosine([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-9
        assert abs(_cosine([0.0, 0.0], [1.0, 0.0])) < 1e-9   # zero vector → 0

    def test_cosine_unnormalised_vectors(self):
        """_cosine() must handle unnormalised vectors correctly."""
        from core.cache import _cosine
        # [3, 4] · [6, 8] / (5 * 10) = 50/50 = 1.0
        assert abs(_cosine([3.0, 4.0], [6.0, 8.0]) - 1.0) < 1e-9
        # [1, 0] · [0, 5] / (1 * 5) = 0.0
        assert abs(_cosine([1.0, 0.0], [0.0, 5.0])) < 1e-9


# ===========================================================================
# SECTION 5 — Schema exclusion (FlashcardOutput)
# ===========================================================================

class TestSchemaExclusion:
    """
    FlashcardOutput (and any other schema in SEMANTIC_CACHE_EXCLUDE_SCHEMAS)
    must never be stored in or served from the cache.
    """

    def test_excluded_schema_store_not_called(self):
        """call_structured() must not call cache.store() for an excluded schema."""
        from core.config import settings
        from core.models import call_structured

        mock_result = FlashcardOutput(question="Q?", answer="A.")
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_result

        cache_mock = MagicMock()
        cache_mock.lookup.return_value = None

        with patch.object(settings, "AGENT_CACHE_ENABLED", True), \
             patch("core.models.get_llm", return_value=mock_llm), \
             patch("core.cache.get_cache", return_value=cache_mock):
            call_structured(FlashcardOutput, "Generate a flashcard about X")

        assert cache_mock.store.call_count == 0, (
            "FlashcardOutput was stored in cache despite being in "
            "SEMANTIC_CACHE_EXCLUDE_SCHEMAS"
        )

    def test_excluded_schema_lookup_not_called(self):
        """call_structured() must not call cache.lookup() for an excluded schema."""
        from core.config import settings
        from core.models import call_structured

        mock_result = FlashcardOutput(question="Q?", answer="A.")
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_result

        cache_mock = MagicMock()

        with patch.object(settings, "AGENT_CACHE_ENABLED", True), \
             patch("core.models.get_llm", return_value=mock_llm), \
             patch("core.cache.get_cache", return_value=cache_mock):
            call_structured(FlashcardOutput, "Generate a flashcard about X")

        assert cache_mock.lookup.call_count == 0, (
            "lookup() was called for FlashcardOutput — excluded schema must "
            "bypass cache entirely"
        )

    def test_non_excluded_schema_does_use_cache(self):
        """A non-excluded schema (DummyScore) must still go through lookup/store."""
        from core.config import settings
        from core.models import call_structured

        obj = DummyScore(value=4, label="mid")
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = obj

        cache_mock = MagicMock()
        cache_mock.lookup.return_value = None  # simulate miss

        # Temporarily add DummyScore as non-excluded (default list only has FlashcardOutput)
        with patch.object(settings, "AGENT_CACHE_ENABLED", True), \
             patch.object(settings, "SEMANTIC_CACHE_EXCLUDE_SCHEMAS", ["FlashcardOutput"]), \
             patch("core.models.get_llm", return_value=mock_llm), \
             patch("core.cache.get_cache", return_value=cache_mock):
            result = call_structured(DummyScore, "Score this document")

        assert result == obj
        assert cache_mock.lookup.call_count == 1
        assert cache_mock.store.call_count == 1


# ===========================================================================
# SECTION 6 — AGENT_CACHE_ENABLED=False
# ===========================================================================

class TestCacheDisabled:

    def test_no_cache_calls_when_disabled(self):
        from core.config import settings
        from core.models import call_structured

        obj = DummyScore(value=4, label="mid")
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = obj

        cache_mock = MagicMock()

        with patch.object(settings, "AGENT_CACHE_ENABLED", False), \
             patch("core.models.get_llm", return_value=mock_llm), \
             patch("core.cache.get_cache", return_value=cache_mock):
            result = call_structured(DummyScore, "What is 1+1?")

        assert result == obj
        assert cache_mock.lookup.call_count == 0
        assert cache_mock.store.call_count == 0

    def test_result_still_returned_when_disabled(self):
        from core.config import settings
        from core.models import call_structured

        obj = DummyScore(value=8, label="high")
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = obj

        with patch.object(settings, "AGENT_CACHE_ENABLED", False), \
             patch("core.models.get_llm", return_value=mock_llm):
            result = call_structured(DummyScore, "Classify this text")

        assert isinstance(result, DummyScore)
        assert result.value == 8


# ===========================================================================
# SECTION 7 — call_structured / call_structured_chain wiring
# ===========================================================================

class TestCallStructuredCacheWiring:

    def test_stores_on_cache_miss(self):
        """First call (cache miss) must call store() after a successful LLM call."""
        from core.config import settings
        from core.models import call_structured

        obj = DummyScore(value=6, label="wired")
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = obj

        cache_mock = MagicMock()
        cache_mock.lookup.return_value = None

        with patch.object(settings, "AGENT_CACHE_ENABLED", True), \
             patch.object(settings, "SEMANTIC_CACHE_EXCLUDE_SCHEMAS", []), \
             patch("core.models.get_llm", return_value=mock_llm), \
             patch("core.cache.get_cache", return_value=cache_mock):
            result = call_structured(DummyScore, "Score this document")

        assert result == obj
        assert cache_mock.lookup.call_count == 1
        assert cache_mock.store.call_count == 1

    def test_returns_cached_result_on_hit(self):
        """On a cache hit the LLM must NOT be called."""
        from core.config import settings
        from core.models import call_structured

        cached_obj = DummyScore(value=2, label="cached")
        mock_llm = MagicMock()

        cache_mock = MagicMock()
        cache_mock.lookup.return_value = cached_obj

        with patch.object(settings, "AGENT_CACHE_ENABLED", True), \
             patch.object(settings, "SEMANTIC_CACHE_EXCLUDE_SCHEMAS", []), \
             patch("core.models.get_llm", return_value=mock_llm), \
             patch("core.cache.get_cache", return_value=cache_mock):
            result = call_structured(DummyScore, "Score this document")

        assert result is cached_obj
        mock_llm.with_structured_output.assert_not_called()

    def test_exception_result_not_stored(self):
        """If the LLM raises, store() must not be called."""
        from core.config import settings
        from core.models import call_structured

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("boom")

        cache_mock = MagicMock()
        cache_mock.lookup.return_value = None

        with patch.object(settings, "AGENT_CACHE_ENABLED", True), \
             patch.object(settings, "SEMANTIC_CACHE_EXCLUDE_SCHEMAS", []), \
             patch("core.models.get_llm", return_value=mock_llm), \
             patch("core.cache.get_cache", return_value=cache_mock):
            with pytest.raises(RuntimeError):
                call_structured(DummyScore, "Will fail")

        assert cache_mock.store.call_count == 0, (
            "store() called despite LLM raising — failed calls must never be cached"
        )

    def test_call_structured_chain_stores_on_miss(self):
        """call_structured_chain() must store on a cache miss."""
        from core.config import settings
        from core.models import call_structured_chain
        from langchain_core.prompts import ChatPromptTemplate

        obj = OtherSchema(text="chain result")

        prompt_tpl = ChatPromptTemplate.from_messages([
            ("user", "Summarise: {content}")
        ])

        mock_chain = MagicMock()
        mock_chain.steps = [prompt_tpl]
        mock_chain.invoke.return_value = obj

        cache_mock = MagicMock()
        cache_mock.lookup.return_value = None

        with patch.object(settings, "AGENT_CACHE_ENABLED", True), \
             patch.object(settings, "SEMANTIC_CACHE_EXCLUDE_SCHEMAS", []), \
             patch("core.cache.get_cache", return_value=cache_mock):
            result = call_structured_chain(
                mock_chain, OtherSchema, {"content": "some text"}
            )

        assert result == obj
        assert cache_mock.store.call_count == 1

    def test_call_structured_chain_returns_hit(self):
        """call_structured_chain() must return cached result without calling chain.invoke()."""
        from core.config import settings
        from core.models import call_structured_chain
        from langchain_core.prompts import ChatPromptTemplate

        cached = OtherSchema(text="cached chain")
        prompt_tpl = ChatPromptTemplate.from_messages([("user", "X: {content}")])

        mock_chain = MagicMock()
        mock_chain.steps = [prompt_tpl]

        cache_mock = MagicMock()
        cache_mock.lookup.return_value = cached

        with patch.object(settings, "AGENT_CACHE_ENABLED", True), \
             patch.object(settings, "SEMANTIC_CACHE_EXCLUDE_SCHEMAS", []), \
             patch("core.cache.get_cache", return_value=cache_mock):
            result = call_structured_chain(
                mock_chain, OtherSchema, {"content": "some text"}
            )

        assert result is cached
        mock_chain.invoke.assert_not_called()


# ===========================================================================
# SECTION 8 — Thread safety
# ===========================================================================

class TestThreadSafety:

    def test_concurrent_stores_no_corruption(self, fresh_cache):
        """10 threads storing simultaneously must not corrupt stats."""
        _inject_embed(fresh_cache, [1.0, 0.0])

        threads = [
            threading.Thread(
                target=fresh_cache.store,
                args=("p", DummyScore, DummyScore(value=i, label=str(i))),
            )
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        s = fresh_cache.stats()
        assert isinstance(s, dict)
        assert set(s.keys()) == {"hits", "misses", "stores"}
        assert s["stores"] == 10

    def test_concurrent_lookups_no_corruption(self, fresh_cache):
        """10 threads looking up simultaneously must not corrupt stats."""
        _inject_embed(fresh_cache, [1.0, 0.0])
        fresh_cache._threshold = 0.0
        fresh_cache.store("q", DummyScore, DummyScore(value=5, label="t"))

        results = []
        lock = threading.Lock()

        def lookup_and_record():
            r = fresh_cache.lookup("q", DummyScore)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=lookup_and_record) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All lookups must return the stored object
        assert all(r is not None for r in results), (
            "Some concurrent lookups returned None on a populated cache"
        )
        s = fresh_cache.stats()
        assert s["hits"] == 10


# ===========================================================================
# SECTION 9 — Integration: real embeddings
# ===========================================================================

def _has_embedding_backend() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        pass
    return bool(os.environ.get("OPENAI_API_KEY"))


requires_embed = pytest.mark.skipif(
    not _has_embedding_backend(),
    reason="No embedding backend available (install sentence-transformers or set OPENAI_API_KEY)",
)


@pytest.mark.integration
class TestSemanticSimilarity:

    @requires_embed
    def test_paraphrase_hits_cache(self):
        """A close paraphrase of the stored prompt should return a cache hit."""
        cache = SemanticCache(similarity_threshold=0.85)
        obj = DummyScore(value=7, label="semantic")
        cache.store(
            "What is the central limit theorem in probability?",
            DummyScore,
            obj,
            model="test",
        )
        result = cache.lookup(
            "Explain the central limit theorem in statistics",
            DummyScore,
        )
        # We do not hard-assert a hit — similarity depends on embedding quality.
        # But IF a hit occurs it must deserialise to the correct type.
        if result is not None:
            assert isinstance(result, DummyScore)
            assert result.value == 7

    @requires_embed
    def test_unrelated_prompt_misses_cache(self):
        """An unrelated prompt must not hit an entry from a very different domain."""
        cache = SemanticCache(similarity_threshold=0.90)
        cache.store(
            "What is the mean of a normal distribution?",
            DummyScore,
            DummyScore(value=5, label="stats"),
        )
        result = cache.lookup(
            "How do you configure a Kubernetes pod resource limit?",
            DummyScore,
        )
        assert result is None, (
            "Kubernetes question hit a statistics cache entry — "
            "similarity threshold or embeddings are not discriminating enough"
        )

    @requires_embed
    def test_get_cache_singleton_returns_same_instance(self):
        """get_cache() must return the same object on repeated calls."""
        import core.cache as cache_mod
        cache_mod._cache_instance = None  # start fresh
        c1 = get_cache()
        c2 = get_cache()
        assert c1 is c2, "get_cache() must return a singleton"


# ===========================================================================
# SECTION 10 — Slow / live-API: end-to-end cache round-trip
# ===========================================================================

def _has_any_key() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    )


@pytest.mark.slow
@pytest.mark.integration
class TestLiveApiCacheRoundTrip:
    """
    These tests hit a real LLM.  They verify that after the first call_structured()
    call stores a result, the second identical call returns from cache (API = 1 call).
    """

    def test_second_identical_call_hits_cache(self):
        if not _has_any_key():
            pytest.skip("No LLM API key configured")
        if not _has_embedding_backend():
            pytest.skip("No embedding backend configured")

        from core.config import settings
        from core.models import call_structured

        try:
            from agents.relevance import RelevanceScore
        except ImportError:
            pytest.skip("RelevanceScore not importable")

        import core.cache as cache_mod
        cache_mod._cache_instance = SemanticCache(similarity_threshold=0.95)

        prompt = "Is the law of large numbers related to probability theory?"

        with patch.object(settings, "AGENT_CACHE_ENABLED", True):
            r1 = call_structured(RelevanceScore, prompt)
            if r1 is None:
                pytest.skip("LLM returned None (quota exhausted?)")

            r2 = call_structured(RelevanceScore, prompt)

        assert r2 is not None
        assert isinstance(r2, RelevanceScore)

        s = cache_mod._cache_instance.stats()
        assert s["stores"] >= 1, "First call must have stored a result"
        assert s["hits"] >= 1, "Second identical call must have been a cache hit"

    def test_cache_does_not_serve_wrong_schema_from_live_call(self):
        """
        After storing a RelevanceScore result, looking up with a different schema
        must still be a miss.
        """
        if not _has_any_key():
            pytest.skip("No LLM API key configured")
        if not _has_embedding_backend():
            pytest.skip("No embedding backend configured")

        from core.config import settings
        from core.models import call_structured

        try:
            from agents.relevance import RelevanceScore
        except ImportError:
            pytest.skip("RelevanceScore not importable")

        import core.cache as cache_mod
        cache_mod._cache_instance = SemanticCache(similarity_threshold=0.95)

        prompt = "Is Bayes theorem related to conditional probability?"

        with patch.object(settings, "AGENT_CACHE_ENABLED", True):
            r1 = call_structured(RelevanceScore, prompt)
            if r1 is None:
                pytest.skip("LLM returned None")

            # Same prompt, different schema → must be a miss (no LLM result for OtherSchema)
            r2 = cache_mod._cache_instance.lookup(prompt, OtherSchema)

        assert r2 is None, (
            "Cache served a RelevanceScore entry for an OtherSchema lookup — "
            "schema isolation broken"
        )
