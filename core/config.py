"""
core/config.py
--------------
Application configuration using Pydantic Settings.
All values can be overridden via environment variables or a `.env` file.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    LANGCHAIN_API_KEY: str = ""
    
    # LangSmith Tracing
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_PROJECT: str = "nexus_learner_mvp"
    
    # Models Config
    DEFAULT_LLM_PROVIDER: str = "openai" # "openai" or "anthropic"
    PRIMARY_MODEL: str = "gpt-4o"
    ROUTING_MODEL: str = "gpt-4o-mini"
    
    # Database
    DB_URL: str = "sqlite:///./nexus.db"
    
    # Vector DB (Qdrant)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "nexus_chunks"
    
    # Application State
    AUTO_ACCEPT_CONTENT: bool = False # If True, bypasses Mentor Review flag

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Global settings instance
settings = Settings()
