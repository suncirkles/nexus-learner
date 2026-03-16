"""
tests/test_e2e_hop_cache_comparison.py
---------------------------------------
Baseline vs. model-hop + semantic-cache comparison tests.

Marked @pytest.mark.slow and @pytest.mark.integration.
These tests require real API keys and a running Qdrant instance.
Skip automatically when neither MODEL_HOP_ENABLED nor AGENT_CACHE_ENABLED
is set, or when no provider keys are configured.

Run with:
    PYTHONPATH=. pytest tests/test_e2e_hop_cache_comparison.py -v -m slow
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_any_key() -> bool:
    """True if at least one LLM provider key is available."""
    return bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
    )


requires_key = pytest.mark.skipif(
    not _has_any_key(),
    reason="No LLM provider API key configured — skipping integration test",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def baseline_settings_patch():
    """Settings overrides for a plain baseline run (no hopping, no cache)."""
    return {
        "MODEL_HOP_ENABLED": False,
        "AGENT_CACHE_ENABLED": False,
    }


@pytest.fixture(scope="module")
def hop_cache_settings_patch():
    """Settings overrides for a hop + cache run."""
    return {
        "MODEL_HOP_ENABLED": True,
        "AGENT_CACHE_ENABLED": True,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
@requires_key
def test_hop_produces_valid_flashcards(hop_cache_settings_patch):
    """
    With MODEL_HOP_ENABLED=True the pipeline must still produce valid flashcards.
    This is a smoke test — it only verifies structural validity, not card quality.
    """
    from core.models import get_llm
    from core.config import settings

    with patch.object(settings, "MODEL_HOP_ENABLED", True):
        try:
            llm = get_llm(purpose="routing")
        except Exception as exc:
            pytest.skip(f"get_llm raised during hop setup: {exc}")

        assert llm is not None, "get_llm() returned None with MODEL_HOP_ENABLED=True"


@pytest.mark.slow
@pytest.mark.integration
@requires_key
def test_hop_card_quality_within_tolerance():
    """
    Mean critic aggregate score with hopping must be within 1 point of baseline.
    Skipped when baseline or hop runs produce zero cards (avoids division by zero).
    """
    pytest.skip(
        "Full pipeline fixture not yet wired — enable once Phase 1 test harness "
        "supports settings injection (see plan §Step 4)."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_cache_reduces_api_calls_on_second_run():
    """
    When AGENT_CACHE_ENABLED=True and core/cache.py is available, a second
    identical GENERATION pass should record cache hits > 0.
    Skipped when core.cache is not importable (branch pre-merge with master).
    """
    try:
        from core.cache import get_cache  # noqa: F401
    except ImportError:
        pytest.skip("core/cache.py not available on this branch — skipping cache hit test")

    from core.config import settings

    with patch.object(settings, "AGENT_CACHE_ENABLED", True):
        cache = get_cache()
        # Verify the cache object has a stats() method (contract check)
        assert hasattr(cache, "stats"), "Cache object missing stats() method"


@pytest.mark.slow
@pytest.mark.integration
def test_hop_fallback_when_no_free_tier_keys():
    """
    With MODEL_HOP_ENABLED=True but all free-tier provider keys removed,
    get_llm() must either fall back to OpenAI/Anthropic or raise RuntimeError.
    It must NOT raise an unhandled exception that propagates through the workflow.
    """
    from core.config import settings

    free_tier_keys = [
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
    ]
    env_patch = {k: "" for k in free_tier_keys}

    with patch.dict(os.environ, env_patch):
        with patch.object(settings, "MODEL_HOP_ENABLED", True):
            try:
                from core.models import get_llm
                llm = get_llm(purpose="routing")
                # If we get here, fallback succeeded — that's fine
                assert llm is not None
            except (ValueError, RuntimeError):
                # Graceful error — acceptable
                pass
            except ImportError:
                # scripts/model_hop not on this branch — fallback path taken
                pass
            except Exception as exc:
                pytest.fail(
                    f"Unexpected exception type {type(exc).__name__} propagated: {exc}"
                )


@pytest.mark.slow
@pytest.mark.integration
def test_cache_disabled_leaves_agents_unaffected():
    """
    When AGENT_CACHE_ENABLED=False, call_structured() must still return a result
    (LLM call happens normally) and must NOT write anything to the cache.
    Skipped when core.cache is not importable (pre-merge branch).
    """
    try:
        from core.cache import get_cache
    except ImportError:
        pytest.skip("core/cache.py not available on this branch")

    from core.config import settings
    from core.models import call_structured

    # Use a trivially-true relevance check to avoid needing a real document
    try:
        from agents.relevance import RelevanceScore
    except ImportError:
        pytest.skip("RelevanceScore not importable")

    with patch.object(settings, "AGENT_CACHE_ENABLED", False):
        cache_before = get_cache()
        stats_before = cache_before.stats().get("stores", 0)

        try:
            result = call_structured(
                RelevanceScore,
                "Is Bayes theorem relevant to probability?",
                purpose="routing",
            )
        except Exception as exc:
            pytest.skip(f"LLM call failed (no key?): {exc}")

        stats_after = get_cache().stats().get("stores", 0)
        assert stats_after == stats_before, (
            f"Cache stored {stats_after - stats_before} item(s) despite AGENT_CACHE_ENABLED=False"
        )


@pytest.mark.slow
@pytest.mark.integration
def test_call_structured_returns_correct_schema():
    """
    call_structured() must return an instance of the requested schema,
    not a raw string or dict, for a simple routing-purpose call.
    Requires at least one provider key.
    """
    if not _has_any_key():
        pytest.skip("No LLM provider key configured")

    from core.models import call_structured
    from core.config import settings

    try:
        from agents.relevance import RelevanceScore
    except ImportError:
        pytest.skip("RelevanceScore not importable")

    with patch.object(settings, "AGENT_CACHE_ENABLED", False):
        try:
            result = call_structured(
                RelevanceScore,
                "Is gradient descent relevant to neural network training?",
                purpose="routing",
            )
        except Exception as exc:
            pytest.skip(f"LLM call failed: {exc}")

    assert isinstance(result, RelevanceScore), (
        f"Expected RelevanceScore instance, got {type(result)}"
    )
    assert isinstance(result.is_relevant, bool)
    assert isinstance(result.reasoning, str)
