"""
core/config.py
--------------
Application configuration using Pydantic Settings.
All values can be overridden via environment variables or a `.env` file.
"""

import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""            # Google AI Studio (Gemini)
    DEEPSEEK_API_KEY: str = ""          # DeepSeek (OpenAI-compatible API)
    LANGCHAIN_API_KEY: str = ""

    # LangSmith Tracing
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_PROJECT: str = "nexus_learner_mvp"

    # Models Config
    DEFAULT_LLM_PROVIDER: str = "openai"  # "openai" | "anthropic" | "groq" | "google" | "deepseek"
    PRIMARY_MODEL: str = "gpt-4o"
    ROUTING_MODEL: str = "gpt-4o-mini"
    GEMINI_PRIMARY_MODEL: str = "gemini-2.0-flash"     # 1500 RPD free tier
    GEMINI_ROUTING_MODEL: str = "gemini-2.0-flash"
    DEEPSEEK_PRIMARY_MODEL: str = "deepseek-chat"
    DEEPSEEK_ROUTING_MODEL: str = "deepseek-chat"

    # Database
    DB_URL: str = ""

    # Modal (Serverless Backend)
    MODAL_TOKEN_ID: str = ""
    MODAL_TOKEN_SECRET: str = ""
    MODAL_ENVIRONMENT: str = "main"

    # Vector DB — PGVector (PostgreSQL/Supabase, uses DB_URL)
    VECTOR_STORE_TYPE: str = "pgvector"
    PGVECTOR_COLLECTION_NAME: str = "nexus_vectors"

    # Embeddings
    # "openai"      = OpenAIEmbeddings (requires valid key, 1536 dims)
    # "huggingface" = local all-MiniLM-L6-v2 via fastembed (no key needed, 384 dims)
    EMBEDDING_PROVIDER: str = "openai"

    # Model hopping (free-tier multi-provider routing via LiteLLM)
    MODEL_HOP_ENABLED: bool = False              # False = original OpenAI/Anthropic only
    MODEL_HOP_PRIMARY_TIER: str = "balanced"     # tier for purpose="primary"
    MODEL_HOP_ROUTING_TIER: str = "fast"         # tier for purpose="routing"

    # Semantic Cache (Qdrant-backed, local sentence-transformers embeddings)
    SEMANTIC_CACHE_ENABLED: bool = True
    AGENT_CACHE_ENABLED: bool = False            # alias used by call_structured helpers
    SEMANTIC_CACHE_COLLECTION: str = "nexus_semantic_cache"
    SEMANTIC_CACHE_THRESHOLD: float = 0.92      # cosine similarity floor for a cache hit
    SEMANTIC_CACHE_TTL_SECONDS: int = 86400     # 0 = no TTL; default 24h
    SEMANTIC_CACHE_MAX_ENTRIES: int = 10000     # soft cap; oldest entries evicted beyond this
    # Schemas excluded from caching (non-deterministic agents — temperature > 0)
    SEMANTIC_CACHE_EXCLUDE_SCHEMAS: list[str] = Field(
        default=["FlashcardOutput"],
        description="Schema __name__ values never stored in or served from cache",
    )
    SEMANTIC_CACHE_BACKEND: str = "pgvector" # "pgvector" (default) | "redis" | "qdrant" (deprecated)
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_CACHE_DB: int = 1                  # separate DB index for cache

    # Storage (Shared Volumes)
    DATA_DIR: str = os.getenv("DATA_DIR", ".")
    UPLOAD_DIR: str = "temp_uploads"    # relative to DATA_DIR
    
    # Logging
    LOG_LEVEL: str = "INFO"           # DEBUG | INFO | WARNING | ERROR | CRITICAL
    LOG_DIR: str = "logs"             # relative to DATA_DIR
    LOG_FILE: str = "nexus_learner.log"
    LOG_MAX_BYTES: int = 10_485_760   # 10 MB per file
    LOG_BACKUP_COUNT: int = 5         # Keep 5 rotated files

    # Page image cache (rendered PNG per PDF page; used by Source Snippet panel)
    PAGE_CACHE_DIR: str = "page_cache"   # relative to DATA_DIR

    # Application State
    AUTO_ACCEPT_CONTENT: bool = False  # If True, bypasses Mentor Review flag

    # Critic auto-reject threshold (per dimension, 1–4 scale).
    # A flashcard is rejected if ANY dimension scores strictly below this value.
    # Default 2 = reject only when a dimension hits the absolute minimum (1).
    # Set to 1 to disable auto-rejection entirely (cards always reach Pending Review).
    CRITIC_REJECT_MIN_SCORE: int = 2

    # Rate limiting
    GENERATION_CHUNK_DELAY: float = 1.0   # seconds to sleep between chunks in background generation
    LLM_MAX_RETRIES: int = 4              # max retry attempts on rate limit errors
    LLM_RETRY_BASE_DELAY: float = 5.0    # initial backoff delay in seconds (doubles each retry)

    # Chunking Settings
    CHUNK_SIZE: int = 3000
    CHUNK_OVERLAP: int = 400
    MAX_SUBTOPIC_CHARS: int = 12000   # max chars fed to Socratic per subtopic (≈3k tokens)

    # Card generation limits
    # Maximum flashcards generated per PDF per question type per generation run.
    # Hard-capped at 50 regardless of what this is set to.
    # Set to 0 to use the hard cap (50) as the effective limit.
    MAX_CARDS_PER_PDF: int = 3

    # Maximum flashcards generated per topic per question type per generation run.
    # Limits how many chunks are fed to the LLM per topic — saves tokens during testing.
    # Hard-capped at 50. Existing approved/pending cards count toward this limit,
    # so re-running generation only fills the remaining capacity.
    MAX_CARDS_PER_TOPIC_TYPE: int = 5

    # Batch API settings
    BATCH_MODEL: str = "claude-sonnet-4-6"   # Anthropic model for batch jobs (must be a Claude model)
    BATCH_MIN_CHUNKS: int = 5                # minimum chunks to justify a batch submission

    # Web Scraping Settings
    WEB_SCRAPE_TIMEOUT: int = 10            # HTTP request timeout seconds
    WEB_MAX_PAGES_PER_TOPIC: int = 3        # Max pages to scrape per topic
    WEB_MAX_CONTENT_CHARS: int = 8000       # Max chars per scraped page
    WEB_SEARCH_MAX_RESULTS: int = 5         # DuckDuckGo results per query
    CONTENT_SAFETY_ENABLED: bool = True     # Enable safety guardrails

    # Security
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]

    @property
    def abs_upload_dir(self) -> str:
        path = os.path.join(self.DATA_DIR, self.UPLOAD_DIR)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def abs_log_dir(self) -> str:
        path = os.path.join(self.DATA_DIR, self.LOG_DIR)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def abs_page_cache_dir(self) -> str:
        path = os.path.join(self.DATA_DIR, self.PAGE_CACHE_DIR)
        os.makedirs(path, exist_ok=True)
        return path

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Global settings instance
settings = Settings()
