"""
scripts/model_hop.py
---------------------
Reusable multi-provider model hopping utilities for Nexus Learner experiments.

Provides a thin, composable layer on top of core.models.get_llm() that makes
it easy to:
  - instantiate any configured provider (OpenAI, Anthropic, Groq, Google)
  - bind structured output with automatic json_mode fallback
  - detect quota / rate-limit errors across all providers
  - generate structured output with graceful quota handling
  - discover which providers have API keys configured
  - run RAGAS faithfulness + response-relevancy benchmarks across providers

Intended to be imported by experiment scripts, not the main application.

Usage
-----
    from scripts.model_hop import (
        get_llm, bind_structured, is_quota_error,
        generate_structured, available_providers,
        build_ragas_evaluator, run_ragas_benchmark, print_ragas_table,
    )

    llm = get_llm("groq", purpose="routing")
    structured = bind_structured(llm, MySchema)
    result = generate_structured("google", MySchema, prompt)

    providers = available_providers()          # {"Gemini 2.0 Flash": ("google", "primary"), ...}
    eval_llm, eval_emb = build_ragas_evaluator()
    scores = run_ragas_benchmark(
        {"Claude": claude_cards, "Groq": groq_cards},
        eval_llm, eval_emb,
        context=source_text,
    )
    print_ragas_table(scores)
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel
    from langchain_core.language_models.chat_models import BaseChatModel

# ---------------------------------------------------------------------------
# Provider utilities
# ---------------------------------------------------------------------------

def get_llm(
    provider: str,
    purpose: str = "primary",
    temperature: float = 0.0,
) -> "BaseChatModel":
    """Return an LLM for the given provider and purpose.

    Thin wrapper around core.models.get_llm() — all provider/key logic lives there.
    """
    from core.models import get_llm as _get_llm
    return _get_llm(purpose=purpose, provider=provider, temperature=temperature)


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
    Checks for error signatures across OpenAI, Anthropic, Groq, and Google APIs.
    """
    msg = str(exc).upper()
    return any(kw in msg for kw in (
        "RESOURCE_EXHAUSTED",   # Google
        "RATE_LIMIT_EXCEEDED",  # OpenAI / Groq
        "INSUFFICIENT_QUOTA",   # OpenAI billing
        "429",                  # HTTP status code (all providers)
        "OVERLOADED",           # Anthropic
        "TOO MANY REQUESTS",    # generic
    ))


def generate_structured(
    provider: str,
    schema: type["BaseModel"],
    prompt: str,
    purpose: str = "primary",
    temperature: float = 0.0,
) -> Optional["BaseModel"]:
    """Generate structured output from the given provider.

    Returns the parsed schema instance on success, or None if the provider's
    quota is exhausted (prints a warning). Raises on genuine errors.
    """
    try:
        llm = get_llm(provider, purpose=purpose, temperature=temperature)
        structured = bind_structured(llm, schema)
        return structured.invoke(prompt)
    except Exception as e:
        if is_quota_error(e):
            print(f"  [WARN] {provider} quota/rate-limit: {_short_error(e)}")
            return None
        raise


def available_providers(include_openai: bool = True) -> dict[str, tuple[str, str]]:
    """Return providers that have API keys configured in settings.

    Returns a dict mapping display label → (provider_id, purpose).
    The "purpose" is the recommended use for each provider in the Nexus
    free-tier strategy:
        google   → primary  (generation)
        anthropic → primary (generation / paid baseline)
        groq     → routing  (classification)
        openai   → primary  (paid baseline, if available)

    Args:
        include_openai: If False, skip OpenAI even if key is present (useful
                        when you know the account is over quota).
    """
    from core.config import settings
    providers: dict[str, tuple[str, str]] = {}
    if settings.GOOGLE_API_KEY:
        providers["Gemini 2.0 Flash"] = ("google", "primary")
    if settings.ANTHROPIC_API_KEY:
        providers["Claude Sonnet"] = ("anthropic", "primary")
    if settings.GROQ_API_KEY:
        providers["Groq Llama-3.3-70B"] = ("groq", "routing")
    if include_openai and settings.OPENAI_API_KEY:
        providers["GPT-4o"] = ("openai", "primary")
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
    from core.config import settings

    # Suppress RAGAS deprecation warnings for LangchainLLMWrapper (still functional)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

    # --- Evaluator LLM ---
    lc_llm = None
    if prefer_provider == "anthropic" and settings.ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        # Use Haiku for evaluation — fast and cheap; quality sufficient for RAGAS scoring
        lc_llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            api_key=settings.ANTHROPIC_API_KEY,
        )
        print("  Evaluator LLM  : Claude Haiku 4.5 (Anthropic)")
    elif settings.GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        lc_llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_PRIMARY_MODEL,
            temperature=0.0,
            google_api_key=settings.GOOGLE_API_KEY,
        )
        print(f"  Evaluator LLM  : Gemini {settings.GEMINI_PRIMARY_MODEL}")
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
        ("langchain_huggingface.HuggingFaceEmbeddings",         "HuggingFace (langchain-huggingface)"),
        ("langchain_community.embeddings.HuggingFaceEmbeddings", "HuggingFace (community)"),
    ]:
        try:
            module_path, cls_name = _import_path.rsplit(".", 1)
            import importlib
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
            faith_col  = next((c for c in df.columns if "faithfulness" in c.lower()), None)
            relev_col  = next((c for c in df.columns if "relevancy" in c.lower() or "relevance" in c.lower()), None)
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
