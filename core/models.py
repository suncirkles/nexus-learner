"""
core/models.py
--------------
LLM factory module. Provides a pluggable `get_llm()` function that returns
the correct LangChain chat model based on application configuration.
Supports OpenAI, Anthropic, Groq, Google, and DeepSeek providers with
separate models for primary (generation) and routing (classification) purposes.

When MODEL_HOP_ENABLED=True, delegates to scripts/model_hop.get_llm() (LiteLLM).
Falls back to the original provider path on any error or missing import.

Also provides `call_structured()` and `call_structured_chain()` helpers
that wrap LLM calls with optional semantic caching.

Rate-limit resilience is handled by passing `max_retries` directly to the
underlying SDK constructors, which implement their own exponential-backoff retry.
"""

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from .config import settings

logger = logging.getLogger(__name__)


def get_llm(
    purpose: str = "primary",
    provider: str = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Factory function to get a pluggable LLM instance based on configuration.

    When MODEL_HOP_ENABLED=True, delegates to scripts/model_hop.get_llm() (LiteLLM).
    Falls back to the original OpenAI/Anthropic path on any error or missing import.

    Args:
        purpose: "primary" (e.g., Socratic/Persona) or "routing" (e.g., Classifier/Supervisor)
        provider: Override default provider — ignored when hopping
        temperature: LLM temperature
    """
    if settings.MODEL_HOP_ENABLED:
        try:
            from scripts.model_hop import get_llm as _hop_get_llm
            tier = (
                settings.MODEL_HOP_PRIMARY_TIER
                if purpose == "primary"
                else settings.MODEL_HOP_ROUTING_TIER
            )
            return _hop_get_llm(tier=tier, temperature=temperature)
        except Exception as e:
            logger.warning("model_hop get_llm failed (%s), falling back to original", e)

    # --- original provider path ---
    provider = provider or settings.DEFAULT_LLM_PROVIDER

    if purpose == "routing":
        model_name = settings.ROUTING_MODEL
    else:
        model_name = settings.PRIMARY_MODEL

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY,
            max_retries=settings.LLM_MAX_RETRIES,
        )
    elif provider == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        # Haiku 4.5 for routing, Sonnet 4.6 for primary
        anthropic_model = "claude-haiku-4-5-20251001" if purpose == "routing" else "claude-sonnet-4-6"
        return ChatAnthropic(
            model=anthropic_model,
            temperature=temperature,
            api_key=settings.ANTHROPIC_API_KEY,
            max_retries=settings.LLM_MAX_RETRIES,
        )
    elif provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set.")
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=settings.GROQ_API_KEY,
            max_retries=settings.LLM_MAX_RETRIES,
        )
    elif provider == "google":
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        google_model = (
            settings.GEMINI_ROUTING_MODEL if purpose == "routing"
            else settings.GEMINI_PRIMARY_MODEL
        )
        return ChatGoogleGenerativeAI(
            model=google_model,
            temperature=temperature,
            google_api_key=settings.GOOGLE_API_KEY,
            max_retries=settings.LLM_MAX_RETRIES,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def call_structured(
    schema: type[BaseModel],
    prompt: str,
    purpose: str = "primary",
    temperature: float = 0.0,
    use_cache: bool = True,
) -> BaseModel | None:
    """
    LLM call with optional semantic cache. Falls back gracefully on any error.
    Returns None on quota exhaustion.

    Cache is a no-op when:
    - AGENT_CACHE_ENABLED=False
    - core.cache not importable
    - schema is in SEMANTIC_CACHE_EXCLUDE_SCHEMAS
    - Qdrant is unavailable
    """
    _cache = None

    # 1. Gate: cache enabled in config?
    if use_cache and settings.AGENT_CACHE_ENABLED:
        try:
            from core.cache import get_cache  # noqa: PLC0415
            if schema.__name__ not in settings.SEMANTIC_CACHE_EXCLUDE_SCHEMAS:
                _cache = get_cache()
                cached = _cache.lookup(prompt, schema)
                if cached is not None:
                    logger.debug("Cache HIT for schema=%s", schema.__name__)
                    return cached
        except Exception:
            _cache = None

    # 2. LLM call
    llm = get_llm(purpose=purpose, temperature=temperature)
    try:
        result = llm.with_structured_output(schema).invoke(prompt)
    except Exception as e:
        try:
            from scripts.model_hop import is_quota_error  # noqa: PLC0415
            if is_quota_error(e):
                logger.warning("Quota exhausted for schema=%s: %s", schema.__name__, e)
                return None
        except Exception:
            pass
        raise

    # 3. Store
    if _cache is not None and result is not None:
        try:
            _cache.store(prompt, schema, result, getattr(llm, "model", "unknown"))
        except Exception:
            pass

    return result


def call_structured_chain(
    chain: Any,
    schema: type[BaseModel],
    input_dict: dict,
    use_cache: bool = True,
) -> BaseModel | None:
    """
    Pipe-chain variant of call_structured(). Renders the first template step
    to produce a cache key string, then falls back to chain.invoke() on any
    cache error.

    Cache is a no-op under the same conditions as call_structured().
    """
    _cache = None
    cache_key = None

    if use_cache and settings.AGENT_CACHE_ENABLED:
        try:
            from core.cache import get_cache  # noqa: PLC0415
            if schema.__name__ not in settings.SEMANTIC_CACHE_EXCLUDE_SCHEMAS:
                # Render the prompt template to a plain string for cache keying
                messages = chain.steps[0].format_messages(**input_dict)
                cache_key = "\n".join(m.content for m in messages)
                _cache = get_cache()
                cached = _cache.lookup(cache_key, schema)
                if cached is not None:
                    logger.debug("Cache HIT (chain) for schema=%s", schema.__name__)
                    return cached
        except Exception:
            _cache = None
            cache_key = None

    result = chain.invoke(input_dict)

    if _cache is not None and cache_key is not None and result is not None:
        try:
            _cache.store(cache_key, schema, result, "chain")
        except Exception:
            pass

    return result
