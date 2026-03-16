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
    GROQ_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""            # Google AI Studio (Gemini)
    LANGCHAIN_API_KEY: str = ""
    
    # LangSmith Tracing
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_PROJECT: str = "nexus_learner_mvp"
    
    # Models Config
    DEFAULT_LLM_PROVIDER: str = "openai"  # "openai" | "anthropic" | "groq" | "google"
    PRIMARY_MODEL: str = "gpt-4o"
    ROUTING_MODEL: str = "gpt-4o-mini"
    GEMINI_PRIMARY_MODEL: str = "gemini-2.0-flash"     # 1500 RPD free tier; swap to gemini-2.5-flash for quality (25 RPD only)
    GEMINI_ROUTING_MODEL: str = "gemini-2.0-flash"     # same; 2.5-flash has 25 RPD — too low for routing volume
    
    # Database
    DB_URL: str = "sqlite:///./nexus.db"
    
    # Vector DB (Qdrant)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "nexus_chunks"
    
    # Application State
    AUTO_ACCEPT_CONTENT: bool = False  # If True, bypasses Mentor Review flag

    # Logging
    LOG_LEVEL: str = "INFO"           # DEBUG | INFO | WARNING | ERROR | CRITICAL
    LOG_DIR: str = "logs"             # Directory for log files (relative to project root)
    LOG_FILE: str = "nexus_learner.log"
    LOG_MAX_BYTES: int = 10_485_760   # 10 MB per file
    LOG_BACKUP_COUNT: int = 5         # Keep 5 rotated files

    # Rate limiting
    GENERATION_CHUNK_DELAY: float = 1.0   # seconds to sleep between chunks in background generation
    LLM_MAX_RETRIES: int = 4              # max retry attempts on rate limit errors
    LLM_RETRY_BASE_DELAY: float = 5.0    # initial backoff delay in seconds (doubles each retry)

    # Page image cache (rendered PNG per PDF page; used by Source Snippet panel)
    PAGE_CACHE_DIR: str = "page_cache"   # relative to project root; created on first use

    # Chunking Settings
    CHUNK_SIZE: int = 3000            # math-aware separators split at ~paragraph level
    CHUNK_OVERLAP: int = 400          # less critical now; generation uses full subtopic context
    MAX_SUBTOPIC_CHARS: int = 12000   # max chars fed to Socratic per subtopic (≈3k tokens)

    # Web Scraping Settings
    WEB_SCRAPE_TIMEOUT: int = 10            # HTTP request timeout seconds
    WEB_MAX_PAGES_PER_TOPIC: int = 3        # Max pages to scrape per topic
    WEB_MAX_CONTENT_CHARS: int = 8000       # Max chars per scraped page
    WEB_SEARCH_MAX_RESULTS: int = 5         # DuckDuckGo results per query
    CONTENT_SAFETY_ENABLED: bool = True     # Enable safety guardrails

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Global settings instance
settings = Settings()
