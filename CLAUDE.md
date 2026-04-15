# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Fill in API keys

# Start Qdrant vector DB (required)
docker-compose up -d

# Run the FastAPI service layer (required — start before Streamlit)
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

# Run the Streamlit frontend (in a second terminal)
streamlit run app.py

# Run all tests (set PYTHONPATH)
PYTHONPATH=. pytest tests/ -v

# Run only unit tests
PYTHONPATH=. pytest tests/unit/ -v

# Run a single test file
PYTHONPATH=. pytest tests/test_filtering_accuracy.py -v
```

## Architecture

Nexus Learner transforms educational documents (PDFs, images, web pages) into Active Recall flashcards using a multi-agent, multi-phase LangGraph pipeline with human-in-the-loop (HITL) review.

### Request Flow

```
Streamlit pages
    → ui/api_client.py   (httpx, persistent connection pool)
    → FastAPI (port 8000)
    → Services           (business logic)
    → Repositories       (SQL + Vector factory)
    → SQLite / PGVector / Qdrant
```

All Streamlit pages communicate exclusively through `ui/api_client.py`. No page imports `core.database` or calls services directly.

### Two Ingestion Phases

**Phase 1 (`workflows/phase1_ingestion.py`)** — Local document ingestion with two modes:
- **INDEXING mode**: PDF → extract text → chunk → build Topic/Subtopic hierarchy (CuratorAgent) → assign chunks to subtopics (TopicAssignerAgent) → embed into Qdrant
- **GENERATION mode**: Retrieve pre-indexed chunks filtered by topic relevance (TopicMatcherAgent + RelevanceAgent) → generate flashcards (SocraticAgent) → grade them (CriticAgent)

**Phase 2 (`workflows/phase2_web_ingestion.py`)** — Web research pipeline: DuckDuckGo search → scrape pages → same indexing/generation flow as Phase 1.

### Agent Responsibilities

| Agent | File | Role |
|---|---|---|
| IngestionAgent | `agents/ingestion.py` | PDF extraction (PyMuPDF + Tesseract OCR fallback), chunking, Qdrant embedding |
| CuratorAgent | `agents/curator.py` | LLM-driven Topic→Subtopic hierarchy extraction and DB persistence |
| TopicAssignerAgent | `agents/topic_assigner.py` | Maps chunks to subtopics during indexing; enforces "No General Content" policy |
| TopicMatcherAgent | `agents/topic_matcher.py` | Semantic matching of user-provided topics to indexed subtopics |
| RelevanceAgent | `agents/relevance.py` | Filters chunks by relevance to target topics (conservative: marks uncertain chunks as relevant) |
| SocraticAgent | `agents/socratic.py` | Generates 1–3 Active Recall Q&A pairs per chunk |
| CriticAgent | `agents/critic.py` | Scores flashcard grounding 1–5; auto-rejects if score < 3 |
| WebResearcherAgent | `agents/web_researcher.py` | Scrapes and deduplicates web content for Phase 2 |

### Service & Repository Layer

| Layer | Path | Role |
|---|---|---|
| Services | `services/` | Business logic — `SubjectService`, `TopicService`, `FlashcardService` |
| SQL Repos | `repositories/sql/` | SQLAlchemy queries — one file per table |
| Vector Repo | `repositories/vector/factory.py` | Dynamic provider switching (Qdrant / PGVector) |
| FastAPI routers | `api/routers/` | REST endpoints; `subjects`, `topics`, `flashcards`, `library`, `system` |
| Schemas | `api/schemas.py` | Pydantic request/response models |
| DI | `api/dependencies.py` | FastAPI `Depends()` factories for services and repos |

### Core Infrastructure

- **`core/config.py`** — Pydantic `Settings` loaded from `.env`. Key settings: `DEFAULT_LLM_PROVIDER`, `PRIMARY_MODEL`, `ROUTING_MODEL`, `AUTO_ACCEPT_CONTENT`, `QDRANT_COLLECTION_NAME`, `SEMANTIC_CACHE_BACKEND`, `REDIS_URL`.
- **`core/models.py`** — `get_llm(purpose, provider, temperature)` factory. Supports `openai`, `anthropic`, `groq` providers via LiteLLM; switch via `DEFAULT_LLM_PROVIDER` env var.
- **`core/database.py`** — SQLAlchemy ORM. Tables: `subjects`, `documents`, `subject_document_association`, `topics`, `subtopics`, `content_chunks`, `flashcards`. No migration framework — schema is created fresh on first run.
- **`core/background.py`** — Daemon thread manager for processing remaining chunks after the initial burst of cards is shown to the user. Background task state is in-memory and lost on restart.
- **`core/cache.py`** — Semantic LLM output cache. `SemanticCache` (Qdrant-backed, default) or `RedisSemanticCache` (Redis-backed, set `SEMANTIC_CACHE_BACKEND=redis`). Falls back to `_NullCache` if backend is unreachable.

### Data Storage

- **PostgreSQL** (Supabase) — Relational data (subjects, topics, flashcards, etc.) AND vector storage if `pgvector` enabled.
- **Qdrant** (Docker, port 6333) — Alternative vector store for semantic search/cache.
- **Redis** (optional, port 6379, db 1) — Persistent semantic cache backend

### Flashcard Lifecycle

`pending` (generated) → mentor approves/rejects → `approved` cards appear in the Learner study room. Setting `AUTO_ACCEPT_CONTENT=true` skips HITL review.

### Streamlit Frontend (`app.py`)

Five views: **Dashboard**, **Study Materials** (upload + progress), **Mentor Review** (HITL bulk approve/reject), **Learner** (Active Recall), **System Tools** (admin).

All pages import from `ui/api_client.py`. The client uses a module-level `httpx.Client` singleton with keep-alive connections. `127.0.0.1` is forced (not `localhost`) to avoid the Windows IPv6→IPv4 fallback delay (~2.7 s per fresh connection).

### LangSmith Tracing

Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env` to enable tracing (project: `nexus_learner_mvp`).

### Key `.env` Settings

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_LLM_PROVIDER` | `openai` | `openai` / `anthropic` / `groq` |
| `PRIMARY_MODEL` | `gpt-4o` | Main generation model |
| `ROUTING_MODEL` | `gpt-4o-mini` | Fast routing/classification model |
| `AUTO_ACCEPT_CONTENT` | `false` | Skip HITL review |
| `SEMANTIC_CACHE_ENABLED` | `true` | Enable/disable semantic cache |
| `SEMANTIC_CACHE_BACKEND` | `qdrant` | `qdrant` or `redis` |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection (used when backend=redis) |
| `REDIS_CACHE_DB` | `1` | Redis DB index for cache |
| `VECTOR_STORE_TYPE` | `qdrant` | `qdrant` or `pgvector` |
| `PGVECTOR_COLLECTION_NAME` | `nexus_vectors` | Table name for PGVector storage |
| `API_BASE_URL` | `http://127.0.0.1:8000` | FastAPI base URL used by Streamlit |
