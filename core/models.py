"""
core/models.py
--------------
LLM factory module. Provides a pluggable `get_llm()` function that returns
the correct LangChain chat model based on application configuration.
Supports OpenAI and Anthropic providers with separate models for
primary (generation) and routing (classification) purposes.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from .config import settings

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
            api_key=settings.OPENAI_API_KEY
        )
    elif provider == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        # Haiku for routing, Sonnet for primary if Anthropic is chosen
        anthropic_model = "claude-3-haiku-20240307" if purpose == "routing" else "claude-3-5-sonnet-20240620"
        return ChatAnthropic(
            model=anthropic_model, 
            temperature=temperature,
            api_key=settings.ANTHROPIC_API_KEY
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
