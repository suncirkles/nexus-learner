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
import random
import re
import time
from typing import Any, Callable

_RETRY_MAX = 3
_RETRY_BACKOFF = 5.0

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from .config import settings

logger = logging.getLogger(__name__)


_LONG_WAIT_THRESHOLD = 60.0  # seconds — daily quota limits, not worth retrying


def _parse_retry_after(err_str: str) -> float | None:
    """Parse provider retry-after hints to seconds.

    Handles both formats:
      "try again in 7.625s"           → 7.625
      "try again in 7m24.095999999s"  → 444.096
    """
    m = re.search(r"try again in (?:(\d+)m)?([0-9.]+)s", err_str, re.IGNORECASE)
    if m:
        return float(m.group(1) or 0) * 60 + float(m.group(2) or 0)
    return None


def invoke_with_retry(fn: Callable, *args, **kwargs) -> Any:
    """Call fn(*args, **kwargs) with automatic retry on transient rate-limit errors.

    - Parses the provider's "try again in X[mY]s" hint (handles minutes+seconds).
    - Adds jitter to avoid thundering herd across concurrent background threads.
    - If the suggested wait > 60 s (daily quota), re-raises immediately — no
      point burning retries when the limit resets in minutes, not seconds.
    - Re-raises on genuine errors or after _RETRY_MAX exhausted attempts.
    """
    from scripts.model_hop import is_quota_error  # noqa: PLC0415
    for attempt in range(_RETRY_MAX):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not is_quota_error(e):
                raise
            err_str = str(e)
            # Permanent balance error — blacklist and re-raise immediately (no retry)
            if any(kw in err_str.upper() for kw in ("INSUFFICIENT BALANCE", "INSUFFICIENT_BALANCE")):
                _provider_m = re.search(r"(\w+)Exception", err_str)
                _model_m = re.search(r"model\s*`([^`]+)`", err_str)
                if _provider_m:
                    try:
                        from scripts.model_hop import mark_model_exhausted  # noqa: PLC0415
                        _label = (f"{_provider_m.group(1).lower()}/{_model_m.group(1)}"
                                  if _model_m else _provider_m.group(1).lower())
                        mark_model_exhausted(_label)
                    except Exception:
                        pass
                logger.warning("Insufficient balance, blacklisting model: %s", e)
                raise
            wait_secs = _parse_retry_after(err_str)
            # Daily quota — blacklist the model and let caller retry with next one
            if wait_secs is not None and wait_secs > _LONG_WAIT_THRESHOLD:
                err_str = str(e)
                provider_m = re.search(r"(\w+)Exception", err_str)
                model_m = re.search(r"model\s*`([^`]+)`", err_str)
                if provider_m and model_m:
                    try:
                        from scripts.model_hop import mark_model_exhausted  # noqa: PLC0415
                        mark_model_exhausted(f"{provider_m.group(1).lower()}/{model_m.group(1)}")
                    except Exception:
                        pass
                logger.warning("Daily quota exhausted (wait=%.0fs), skipping retries: %s", wait_secs, e)
                raise
            if attempt < _RETRY_MAX - 1:
                wait = (wait_secs if wait_secs is not None else _RETRY_BACKOFF * (attempt + 1)) + random.uniform(0.5, 3.0)
                logger.warning(
                    "Rate limit (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, _RETRY_MAX, wait, e,
                )
                time.sleep(wait)
            else:
                raise


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

    # 2. LLM call (with retry for transient rate limits)
    from core.context import get_request_id, get_session_id

    llm = get_llm(purpose=purpose, temperature=temperature)
    config = {
        "metadata": {
            "request_id": get_request_id(),
            "session_id": get_session_id(),
        },
        "tags": [get_session_id()]
    }

    result = None
    for _model_try in range(2):  # retry once with next model on daily quota exhaustion
        try:
            result = invoke_with_retry(llm.with_structured_output(schema).invoke, prompt, config=config)
            break
        except Exception as e:
            try:
                from scripts.model_hop import is_quota_error  # noqa: PLC0415
                if is_quota_error(e) and _model_try == 0:
                    # model was just blacklisted; rebuild with the next available one
                    llm = get_llm(purpose=purpose, temperature=temperature)
                    continue
                if is_quota_error(e):
                    logger.warning("Quota exhausted for schema=%s, all models tried: %s",
                                   schema.__name__, e)
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

    from core.context import get_request_id, get_session_id
    config = {
        "metadata": {
            "request_id": get_request_id(),
            "session_id": get_session_id(),
        },
        "tags": [get_session_id()]
    }

    result = None
    for _model_try in range(2):  # retry once with next model on daily quota exhaustion
        try:
            result = invoke_with_retry(chain.invoke, input_dict, config=config)
            break
        except Exception as e:
            try:
                from scripts.model_hop import is_quota_error  # noqa: PLC0415
                if is_quota_error(e) and _model_try == 0:
                    # Rebuild chain with next available model (exhausted one is now blacklisted)
                    llm = get_llm(purpose="primary", temperature=0.0)
                    chain = chain.steps[0] | llm.with_structured_output(schema)
                    continue
                if is_quota_error(e):
                    logger.warning("Quota exhausted (chain), all models tried: %s", e)
                    return None
            except Exception:
                pass
            raise

    if _cache is not None and cache_key is not None and result is not None:
        try:
            _cache.store(cache_key, schema, result, "chain")
        except Exception:
            pass

    return result
