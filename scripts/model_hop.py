"""
scripts/model_hop.py
---------------------
Portable multi-provider model hopping utilities backed by LiteLLM.

Zero project-specific dependencies — works with any LangChain project.
LiteLLM reads standard env vars automatically:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY,
    GOOGLE_API_KEY / GEMINI_API_KEY, DEEPSEEK_API_KEY

Provides:
  - get_llm()              — instantiate any LiteLLM-backed chat model
  - bind_structured()      — structured output with json_mode fallback
  - is_quota_error()       — detect quota/rate-limit errors across providers
  - generate_structured()  — structured output with quota handling + semantic cache
  - available_providers()  — discover providers with API keys configured
  - build_ragas_evaluator()— build RAGAS evaluator LLM + embeddings
  - run_ragas_benchmark()  — run RAGAS faithfulness + response relevancy
  - print_ragas_table()    — print formatted comparison table

Model selection
---------------
Three ways to select a model:

1. Explicit LiteLLM model string:
       llm = get_llm("groq/llama-3.3-70b-versatile")

2. Tier-based (walks priority list, returns first model with a key set):
       llm = get_llm(tier="fast")       # routing, classification
       llm = get_llm(tier="balanced")   # generation, extraction
       llm = get_llm(tier="reasoning")  # critic, analysis, multi-step
       llm = get_llm(tier="quality")    # paid baselines

3. Task-name sugar (maps task → tier internally):
       llm = get_llm(task="routing")    # same as tier="fast"
       llm = get_llm(task="generation") # same as tier="balanced"
"""

from __future__ import annotations

import importlib
import os
import warnings
from typing import Any, Optional, TYPE_CHECKING

import litellm
litellm.drop_params = True  # silently drop unsupported params (e.g. tool_choice=any on Gemini)

if TYPE_CHECKING:
    from pydantic import BaseModel
    from langchain_core.language_models.chat_models import BaseChatModel

# ---------------------------------------------------------------------------
# Tier / task configuration
# ---------------------------------------------------------------------------

TIER_MODELS: dict[str, list[str]] = {
    "fast": [
        "groq/llama-3.3-70b-versatile",
        "gemini/gemini-2.0-flash",
        "anthropic/claude-haiku-4-5-20251001",
    ],
    "balanced": [
        "gemini/gemini-2.0-flash",
        "anthropic/claude-sonnet-4-6",
        "deepseek/deepseek-chat",
        "openai/gpt-4o",
    ],
    "reasoning": [
        "groq/deepseek-r1-distill-llama-70b",   # DeepSeek-R1 on Groq free tier
        "anthropic/claude-sonnet-4-6",
        "openai/o3-mini",
    ],
    "quality": [
        "anthropic/claude-opus-4-6",
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4-6",
    ],
}

TASK_TIERS: dict[str, str] = {
    "routing":        "fast",
    "classification": "fast",
    "generation":     "balanced",
    "primary":        "balanced",
    "extraction":     "balanced",
    "structured":     "balanced",
    "reasoning":      "reasoning",
    "analysis":       "reasoning",
    "critic":         "reasoning",
    "quality":        "quality",
    "evaluation":     "quality",
}

# Maps LiteLLM provider prefix → env var names that signal key presence
KEY_MAP: dict[str, list[str]] = {
    "groq/":       ["GROQ_API_KEY"],
    "gemini/":     ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "anthropic/":  ["ANTHROPIC_API_KEY"],
    "deepseek/":   ["DEEPSEEK_API_KEY"],
    "openai/":     ["OPENAI_API_KEY"],
}

# ---------------------------------------------------------------------------
# Provider utilities
# ---------------------------------------------------------------------------

def _pick_model_for_tier(tier: str) -> str:
    """Return the first model in the tier whose provider key is set.

    Raises RuntimeError if no key is found for any model in the tier.
    """
    candidates = TIER_MODELS.get(tier)
    if not candidates:
        raise ValueError(f"Unknown tier: {tier!r}. Available: {list(TIER_MODELS)}")
    for model in candidates:
        for prefix, env_vars in KEY_MAP.items():
            if model.startswith(prefix):
                if any(os.environ.get(v) for v in env_vars):
                    print(f"  [model_hop] tier={tier!r} → {model}")
                    return model
                break
    tried = ", ".join(candidates)
    raise RuntimeError(
        f"No API key found for any model in tier={tier!r}. "
        f"Tried: {tried}. Set the appropriate key env var."
    )


def get_llm(
    model: Optional[str] = None,
    *,
    tier: Optional[str] = None,
    task: Optional[str] = None,
    temperature: float = 0.0,
) -> "BaseChatModel":
    """Return a LiteLLM-backed chat model.

    Exactly one of model, tier, or task should be provided.

    Args:
        model:       Explicit LiteLLM model string, e.g. "groq/llama-3.3-70b-versatile"
        tier:        Tier name — "fast" | "balanced" | "reasoning" | "quality"
        task:        Task name — walks TASK_TIERS to resolve tier automatically
        temperature: Sampling temperature (default 0.0)
    """
    from langchain_litellm import ChatLiteLLM

    if task is not None:
        resolved_tier = TASK_TIERS.get(task)
        if resolved_tier is None:
            raise ValueError(f"Unknown task: {task!r}. Available: {list(TASK_TIERS)}")
        model = _pick_model_for_tier(resolved_tier)
    elif tier is not None:
        model = _pick_model_for_tier(tier)
    elif model is None:
        raise ValueError("Provide model=, tier=, or task=")

    return ChatLiteLLM(model=model, temperature=temperature)


def bind_structured(llm: "BaseChatModel", schema: type["BaseModel"]) -> Any:
    """Bind structured output to an LLM, falling back to json_mode if tool-calling fails.

    Some providers (older Gemini versions) don't support tool-calling schema binding
    but do support json_mode. This wrapper hides that difference.
    """
    try:
        return llm.with_structured_output(schema)
    except Exception:
        return llm.with_structured_output(schema, method="json_mode")


def is_quota_error(exc: Exception) -> bool:
    """Return True if the exception is an API quota or rate-limit error.

    Distinguishes infrastructure-level throttling (not a code bug) from actual
    programming errors, so callers can skip gracefully instead of hard-failing.
    Checks LiteLLM exception types first, then falls back to string matching
    across OpenAI, Anthropic, Groq, Google, and DeepSeek error signatures.
    """
    # Check LiteLLM typed exceptions first (most reliable)
    try:
        from litellm import exceptions as _le
        if isinstance(exc, (_le.RateLimitError, _le.BudgetExceededError,
                            _le.AuthenticationError)):
            return True
    except (ImportError, AttributeError):
        pass

    msg = str(exc).upper()
    return any(kw in msg for kw in (
        "RESOURCE_EXHAUSTED",       # Google
        "RATE_LIMIT_EXCEEDED",      # OpenAI / Groq
        "RATELIMITERROR",           # LiteLLM class name in message
        "INSUFFICIENT_QUOTA",       # OpenAI billing
        "EXCEEDED YOUR CURRENT QUOTA",  # OpenAI billing (alt phrasing)
        "INSUFFICIENT BALANCE",     # DeepSeek — account needs top-up
        "INSUFFICIENT_BALANCE",     # DeepSeek alternate encoding
        "402",                      # HTTP payment required (DeepSeek balance)
        "429",                      # HTTP rate limit (all providers)
        "OVERLOADED",               # Anthropic
        "TOO MANY REQUESTS",        # generic
    ))


def generate_structured(
    model: str,
    schema: type["BaseModel"],
    prompt: str,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> Optional["BaseModel"]:
    """Generate structured output from the given model or tier.

    Args:
        model:       LiteLLM model string (e.g. "gemini/gemini-2.0-flash") OR
                     tier name (e.g. "balanced") OR task name (e.g. "generation")
        schema:      Pydantic model class for structured output
        prompt:      Input prompt
        temperature: Sampling temperature
        use_cache:   Whether to use semantic cache (default True). Schemas listed
                     in SEMANTIC_CACHE_EXCLUDE_SCHEMAS bypass cache regardless.

    Returns the parsed schema instance on success, or None if quota is
    exhausted (prints a warning). Raises on genuine errors.
    """
    # Schema-level exclusion check (reads config, graceful on ImportError)
    _cache = None
    if use_cache:
        try:
            from core.config import settings  # type: ignore[import]
            if schema.__name__ in settings.SEMANTIC_CACHE_EXCLUDE_SCHEMAS:
                use_cache = False  # honour per-schema exclusion
        except Exception:
            pass

    # Cache lookup (lazy init, never raises)
    if use_cache:
        try:
            from core.cache import get_cache  # type: ignore[import]
            _cache = get_cache()
            cached = _cache.lookup(prompt, schema)
            if cached is not None:
                print(f"  [cache HIT] {schema.__name__} — served from semantic cache")
                return cached
        except Exception:
            _cache = None

    if model in TIER_MODELS:
        llm = get_llm(tier=model, temperature=temperature)
    elif model in TASK_TIERS:
        llm = get_llm(task=model, temperature=temperature)
    else:
        llm = get_llm(model=model, temperature=temperature)

    result = None
    try:
        structured = bind_structured(llm, schema)
        result = structured.invoke(prompt)
    except Exception as e:
        if is_quota_error(e):
            model_name = getattr(llm, "model", model)
            print(f"  [WARN] quota/rate-limit ({model_name}): {_short_error(e)}")
            return None
        raise

    # Cache store (non-blocking, failure-safe)
    if use_cache and _cache is not None and result is not None:
        try:
            _cache.store(prompt, schema, result, getattr(llm, "model", model))
        except Exception:
            pass

    return result


def available_providers(include_openai: bool = True) -> dict[str, str]:
    """Return providers that have API keys configured.

    Returns a dict mapping display label → LiteLLM model string.

    Args:
        include_openai: If False, skip OpenAI even if key is present (useful
                        when you know the account is over quota).
    """
    providers: dict[str, str] = {}
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        providers["Gemini 2.0 Flash"] = "gemini/gemini-2.0-flash"
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers["Claude Sonnet"] = "anthropic/claude-sonnet-4-6"
    if os.environ.get("GROQ_API_KEY"):
        providers["Groq Llama-3.3-70B"] = "groq/llama-3.3-70b-versatile"
    if os.environ.get("DEEPSEEK_API_KEY"):
        providers["DeepSeek-V3"] = "deepseek/deepseek-chat"
    if include_openai and os.environ.get("OPENAI_API_KEY"):
        providers["GPT-4o"] = "openai/gpt-4o"
    return providers


# ---------------------------------------------------------------------------
# RAGAS evaluator
# ---------------------------------------------------------------------------

def build_ragas_evaluator(
    prefer_provider: str = "anthropic",
) -> tuple[Any, Optional[Any]]:
    """Build a RAGAS evaluator LLM and (optionally) embeddings.

    Evaluator LLM priority  : Anthropic → Google → raises
    Embeddings priority     : HuggingFace local → community → None (skip ResponseRelevancy)

    Google embeddings are intentionally skipped: RAGAS's LangchainEmbeddingsWrapper
    calls the v1beta embedContent endpoint, which returns 404 for all current Google
    embedding model IDs.

    Returns:
        (eval_llm, eval_embeddings) — eval_embeddings may be None; in that case
        the caller should run Faithfulness only.
    """
    # Suppress RAGAS deprecation warnings for LangchainLLMWrapper (still functional)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

    from langchain_litellm import ChatLiteLLM

    # --- Evaluator LLM ---
    lc_llm = None
    if prefer_provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        # Use Haiku for evaluation — fast and cheap; quality sufficient for RAGAS scoring
        lc_llm = ChatLiteLLM(model="anthropic/claude-haiku-4-5-20251001", temperature=0.0)
        print("  Evaluator LLM  : Claude Haiku 4.5 (Anthropic)")
    elif os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        lc_llm = ChatLiteLLM(model="gemini/gemini-2.0-flash", temperature=0.0)
        print("  Evaluator LLM  : Gemini 2.0 Flash (Google)")
    else:
        raise ValueError(
            "No evaluator LLM available. Set ANTHROPIC_API_KEY or GOOGLE_API_KEY."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eval_llm = LangchainLLMWrapper(lc_llm)

    # --- Evaluator embeddings ---
    eval_embeddings = None
    for _import_path, _label in [
        ("langchain_huggingface.HuggingFaceEmbeddings",          "HuggingFace (langchain-huggingface)"),
        ("langchain_community.embeddings.HuggingFaceEmbeddings",  "HuggingFace (community)"),
    ]:
        try:
            module_path, cls_name = _import_path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            HFEmbeddings = getattr(mod, cls_name)
            lc_emb = HFEmbeddings(model_name="all-MiniLM-L6-v2")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eval_embeddings = LangchainEmbeddingsWrapper(lc_emb)
            print(f"  Evaluator embeds: {_label} / all-MiniLM-L6-v2 (local)")
            break
        except Exception:
            continue

    if eval_embeddings is None:
        print("  Evaluator embeds: none — ResponseRelevancy will be skipped")
        print("    Install sentence-transformers to enable: pip install sentence-transformers")

    return eval_llm, eval_embeddings


# ---------------------------------------------------------------------------
# RAGAS benchmark runner
# ---------------------------------------------------------------------------

def run_ragas_benchmark(
    provider_outputs: dict[str, list[tuple[str, str]]],
    eval_llm: Any,
    eval_embeddings: Optional[Any],
    context: str,
) -> list[dict]:
    """Run RAGAS faithfulness (+ response relevancy if embeddings available).

    Args:
        provider_outputs: {label: [(question, answer), ...]} — one entry per provider
        eval_llm:         RAGAS-wrapped evaluator LLM (from build_ragas_evaluator)
        eval_embeddings:  RAGAS-wrapped embeddings or None
        context:          Source text used as retrieved_context for all samples

    Returns:
        List of dicts with keys: provider, n_cards, faithfulness, response_relevancy
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from ragas.metrics.collections import Faithfulness, ResponseRelevancy
        except ImportError:
            from ragas.metrics import Faithfulness, ResponseRelevancy
        from ragas import evaluate, EvaluationDataset, SingleTurnSample

    faithfulness_metric = Faithfulness(llm=eval_llm)
    active_metrics = [faithfulness_metric]
    if eval_embeddings is not None:
        active_metrics.append(ResponseRelevancy(llm=eval_llm, embeddings=eval_embeddings))

    results: list[dict] = []
    for label, qa_pairs in provider_outputs.items():
        samples = [
            SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=[context],
            )
            for q, a in qa_pairs
        ]
        dataset = EvaluationDataset(samples=samples)
        try:
            scores = evaluate(dataset, metrics=active_metrics)
            df = scores.to_pandas()
            faith_col = next((c for c in df.columns if "faithfulness" in c.lower()), None)
            relev_col = next((c for c in df.columns if "relevancy" in c.lower() or "relevance" in c.lower()), None)
            results.append({
                "provider":           label,
                "n_cards":            len(qa_pairs),
                "faithfulness":       round(float(df[faith_col].mean()), 3) if faith_col else None,
                "response_relevancy": round(float(df[relev_col].mean()), 3) if relev_col else None,
            })
        except Exception as e:
            print(f"  [WARN] RAGAS evaluation failed for {label}: {e}")
            results.append({
                "provider": label, "n_cards": len(qa_pairs),
                "faithfulness": None, "response_relevancy": None,
            })

    return results


def print_ragas_table(results: list[dict]) -> None:
    """Print a formatted comparison table from run_ragas_benchmark output."""
    col_w = 24
    print(f"\n  {'Provider':<{col_w}}  {'Cards':>5}  {'Faithfulness':>14}  {'Resp. Relevancy':>16}")
    print(f"  {'-'*col_w}  {'-----':>5}  {'-'*14}  {'-'*16}")
    for r in results:
        faith = f"{r['faithfulness']:.3f}" if r["faithfulness"] is not None else "N/A"
        relev = f"{r['response_relevancy']:.3f}" if r["response_relevancy"] is not None else "N/A"
        print(f"  {r['provider']:<{col_w}}  {r['n_cards']:>5}  {faith:>14}  {relev:>16}")
    print()
    print("  faithfulness      : 1.0 = answer fully grounded in source; 0.0 = hallucinated")
    print("  response_relevancy: 1.0 = answer directly addresses the question; 0.0 = off-topic")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _short_error(e: Exception, max_len: int = 120) -> str:
    """Truncate an error message for readable inline warnings."""
    msg = str(e)
    return msg[:max_len] + "…" if len(msg) > max_len else msg
