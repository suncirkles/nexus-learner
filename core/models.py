"""
core/models.py
--------------
LLM factory module. Provides a pluggable `get_llm()` function that returns
the correct LangChain chat model based on application configuration.
Supports OpenAI and Anthropic providers with separate models for
primary (generation) and routing (classification) purposes.

Rate-limit resilience is handled by passing `max_retries` directly to the
underlying SDK constructors (openai / anthropic), which implement their own
exponential-backoff retry.  This keeps the returned object a proper
BaseChatModel that callers can chain with .with_structured_output() etc.
"""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from .config import settings

logger = logging.getLogger(__name__)


def get_llm(
    purpose: str = "primary",
    provider: str = None,
    temperature: float = 0.0
) -> BaseChatModel:
    """
    Factory function to get a pluggable LLM instance based on configuration.

    Args:
        purpose: "primary" (e.g., Socratic/Persona) or "routing" (e.g., Classifier/Supervisor)
        provider: Override default provider ("openai" or "anthropic")
        temperature: LLM temperature

    The returned model has SDK-level retry (exponential backoff) configured via
    settings.LLM_MAX_RETRIES, so transient rate-limit errors are handled
    automatically without requiring changes in calling code.
    """
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
