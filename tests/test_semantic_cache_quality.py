"""
tests/test_semantic_cache_quality.py
-------------------------------------
RAGAS quality-gate tests for the semantic cache.

Marked @pytest.mark.slow — not run in the standard fast suite.

Run:
    PYTHONPATH=. pytest tests/test_semantic_cache_quality.py -v -m slow

Skips gracefully when:
- No LLM provider keys are set
- Qdrant is unavailable
- sentence-transformers is not installed
"""

from __future__ import annotations

import textwrap
import pytest

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Schemas (duplicated from sample_free_tier.py — tests are self-contained)
# ---------------------------------------------------------------------------

class Flashcard(BaseModel):
    question: str = Field(description="Active recall question")
    answer: str = Field(description="Concise, accurate answer")
    difficulty: str = Field(description="easy | medium | hard")

class FlashcardOutput(BaseModel):
    cards: list[Flashcard] = Field(description="1–3 flashcards derived from the passage")
    chunk_summary: str = Field(description="One-sentence summary of the passage")

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(description="True if the chunk is relevant to the target topics")
    reason: str = Field(description="One-sentence justification")


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT = textwrap.dedent("""
    Conditional probability is the probability of an event A occurring given that
    another event B has already occurred. It is written as P(A|B) and defined as:

        P(A|B) = P(A ∩ B) / P(B),  provided P(B) > 0.

    Bayes' Theorem follows directly: P(A|B) = [P(B|A) · P(A)] / P(B).
    P(A) is the prior, P(A|B) is the posterior, P(B|A) is the likelihood.
""").strip()

TARGET_TOPICS = ["Conditional Probability", "Bayes' Theorem"]

CARD_PROMPT = textwrap.dedent(f"""
    You are an Active Recall flashcard generator.
    Generate 1–3 high-quality flashcards from the passage below.
    Focus on conceptual understanding, not rote memorisation.
    Assign difficulty: easy | medium | hard.

    Passage:
    {SAMPLE_TEXT}
""").strip()

RELEVANCE_PROMPT = textwrap.dedent(f"""
    Decide whether the following text chunk is relevant to these topics:
    {TARGET_TOPICS}

    Chunk:
    {SAMPLE_TEXT}
""").strip()


def _skip_if_no_providers():
    """Raise pytest.skip if no LLM provider keys are available."""
    import os
    keys = ["GROQ_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    if not any(os.environ.get(k) for k in keys):
        pytest.skip("No LLM provider API keys configured")


def _get_cache_or_skip():
    """Return a ready SemanticCache or skip the test."""
    try:
        from core.cache import init_semantic_cache, _reset_cache_singleton  # type: ignore[import]
        _reset_cache_singleton()
        cache = init_semantic_cache()
    except ImportError as exc:
        pytest.skip(f"core.cache not importable: {exc}")

    if not cache.stats()["enabled"]:
        pytest.skip("SemanticCache not ready (Qdrant unavailable or sentence-transformers missing)")
    return cache


def _first_available_model() -> str:
    """Return the first available model string, or skip."""
    from scripts.model_hop import available_providers  # type: ignore[import]
    providers = available_providers()
    if not providers:
        pytest.skip("No LLM provider API keys configured")
    return next(iter(providers.values()))


def _ragas_score(qa_pairs: list[tuple[str, str]], context: str) -> tuple[float | None, float | None]:
    """Run RAGAS and return (faithfulness, response_relevancy).

    Returns (None, None) if RAGAS is unavailable or evaluation fails.
    """
    try:
        from scripts.model_hop import build_ragas_evaluator, run_ragas_benchmark  # type: ignore[import]
        eval_llm, eval_embeddings = build_ragas_evaluator()
        results = run_ragas_benchmark({"test": qa_pairs}, eval_llm, eval_embeddings, context)
        r = results[0]
        return r["faithfulness"], r["response_relevancy"]
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cache_hit_produces_identical_ragas_scores():
    """Cache hits must return objects identical to the first-run result.

    Because cached hits return the exact same Pydantic object (deserialised
    from stored JSON), RAGAS scores must be numerically identical to run 1
    and within tolerance of an uncached baseline.
    """
    _skip_if_no_providers()
    cache = _get_cache_or_skip()
    model = _first_available_model()

    from scripts.model_hop import generate_structured  # type: ignore[import]

    # Use RelevanceDecision (not excluded from cache)
    prompts = [
        RELEVANCE_PROMPT,
        RELEVANCE_PROMPT + "\n\nPlease be concise.",
        "Is Bayes' theorem discussed?\n\n" + SAMPLE_TEXT[:200],
    ]

    # Step 1: Uncached baseline
    cache.clear()
    baseline_qa: list[tuple[str, str]] = []
    for p in prompts:
        r = generate_structured(model, RelevanceDecision, p, use_cache=False)
        if r:
            baseline_qa.append((p[:80], r.reason))

    # Step 2: Populate cache (run 1 with cache enabled)
    cache.clear()
    run1_qa: list[tuple[str, str]] = []
    for p in prompts:
        r = generate_structured(model, RelevanceDecision, p, use_cache=True)
        if r:
            run1_qa.append((p[:80], r.reason))

    if cache.stats()["stores"] == 0:
        pytest.skip("All LLM calls returned None (quota exhausted) — cannot populate cache")

    # Step 3: Serve from cache (run 2)
    run2_qa: list[tuple[str, str]] = []
    for p in prompts:
        r = generate_structured(model, RelevanceDecision, p, use_cache=True)
        if r:
            run2_qa.append((p[:80], r.reason))

    assert cache.stats()["hits"] > 0, "Run 2 should have cache hits"

    # Step 4: RAGAS comparison (best-effort; skip if RAGAS unavailable)
    if run1_qa and run2_qa:
        faith_b, _ = _ragas_score(baseline_qa, SAMPLE_TEXT)
        faith_1, _ = _ragas_score(run1_qa, SAMPLE_TEXT)
        faith_2, _ = _ragas_score(run2_qa, SAMPLE_TEXT)

        if faith_1 is not None and faith_2 is not None:
            assert abs(faith_2 - faith_1) < 0.001, (
                f"Cache hit faithfulness ({faith_2}) differs from run 1 ({faith_1}): "
                "cached objects should be identical"
            )
        if faith_b is not None and faith_2 is not None:
            assert abs(faith_2 - faith_b) < 0.05, (
                f"Cache hit faithfulness ({faith_2}) too far from baseline ({faith_b})"
            )


@pytest.mark.slow
def test_cache_boundary_scores_within_tolerance():
    """Semantically similar (rephrased) prompts served from cache must not
    degrade RAGAS faithfulness by more than 0.05 vs the original.

    A cache MISS at the threshold boundary is informational — not a failure.
    """
    _skip_if_no_providers()
    cache = _get_cache_or_skip()
    model = _first_available_model()

    from scripts.model_hop import generate_structured  # type: ignore[import]

    cache.clear()

    # Populate cache with original
    r_original = generate_structured(model, RelevanceDecision, RELEVANCE_PROMPT, use_cache=True)
    if r_original is None:
        pytest.skip("LLM returned None (quota?)")

    original_qa = [(RELEVANCE_PROMPT[:80], r_original.reason)]
    faith_orig, _ = _ragas_score(original_qa, SAMPLE_TEXT)

    # Slightly rephrased variant
    rephrased = RELEVANCE_PROMPT.replace("Decide whether", "Determine if").replace(
        "text chunk", "passage"
    )
    hits_before = cache.stats()["hits"]
    r_rephrased = generate_structured(model, RelevanceDecision, rephrased, use_cache=True)

    if cache.stats()["hits"] > hits_before:
        # Cache HIT — verify faithfulness within tolerance
        if r_rephrased and faith_orig is not None:
            rephrased_qa = [(rephrased[:80], r_rephrased.reason)]
            faith_rep, _ = _ragas_score(rephrased_qa, SAMPLE_TEXT)
            if faith_rep is not None:
                assert abs(faith_rep - faith_orig) < 0.05, (
                    f"Rephrased cache hit faithfulness ({faith_rep}) too far from original ({faith_orig})"
                )
    else:
        # Cache MISS — informational, not a failure
        print(f"\n  [INFO] Rephrased prompt was a MISS (threshold not crossed) — informational only")


@pytest.mark.slow
def test_excluded_schema_never_served_from_cache():
    """FlashcardOutput (default exclusion) must never be stored or served."""
    _skip_if_no_providers()
    cache = _get_cache_or_skip()
    model = _first_available_model()

    from scripts.model_hop import generate_structured  # type: ignore[import]

    cache.clear()

    r1 = generate_structured(model, FlashcardOutput, CARD_PROMPT, use_cache=True)
    r2 = generate_structured(model, FlashcardOutput, CARD_PROMPT, use_cache=True)

    stats = cache.stats()
    assert stats["hits"] == 0, (
        f"FlashcardOutput should be excluded from cache but got hits={stats['hits']}"
    )
    assert stats["stores"] == 0, (
        f"FlashcardOutput should never be stored but got stores={stats['stores']}"
    )

    # Both calls must have returned valid FlashcardOutput (LLM was called both times)
    if r1 is not None:
        assert isinstance(r1, FlashcardOutput)
    if r2 is not None:
        assert isinstance(r2, FlashcardOutput)


@pytest.mark.slow
def test_cache_disabled_globally():
    """SEMANTIC_CACHE_ENABLED=False must make the cache a no-op."""
    import sys

    try:
        from core.cache import _reset_cache_singleton  # type: ignore[import]
        import core.cache as cache_module  # type: ignore[import]
        from core.config import settings  # type: ignore[import]
    except ImportError as exc:
        pytest.skip(f"core modules not importable: {exc}")

    original_enabled = settings.SEMANTIC_CACHE_ENABLED
    try:
        settings.SEMANTIC_CACHE_ENABLED = False
        _reset_cache_singleton()
        cache = cache_module.init_semantic_cache()

        s = cache.stats()
        assert s["enabled"] is False, "Cache should be disabled"
        assert s["hits"] == 0

        # Even if providers are available, verify no-op behaviour without actual LLM calls
        result = cache.lookup("any prompt", RelevanceDecision)
        assert result is None

    finally:
        settings.SEMANTIC_CACHE_ENABLED = original_enabled
        _reset_cache_singleton()
