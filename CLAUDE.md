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

# Run application
streamlit run app.py

# Run all tests (set PYTHONPATH)
PYTHONPATH=. pytest tests/ -v

# Run a single test file
PYTHONPATH=. pytest tests/test_filtering_accuracy.py -v
```

## Architecture

Nexus Learner transforms educational documents (PDFs, images, web pages) into Active Recall flashcards using a multi-agent, multi-phase LangGraph pipeline with human-in-the-loop (HITL) review.

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

### Core Infrastructure

- **`core/config.py`** — Pydantic `Settings` loaded from `.env`. Key settings: `DEFAULT_LLM_PROVIDER`, `PRIMARY_MODEL` (gpt-4o), `ROUTING_MODEL` (gpt-4o-mini), `AUTO_ACCEPT_CONTENT`, `QDRANT_COLLECTION_NAME`.
- **`core/models.py`** — `get_llm(purpose, provider, temperature)` factory. Supports `openai` and `anthropic` providers; switch via `DEFAULT_LLM_PROVIDER` env var.
- **`core/database.py`** — SQLAlchemy ORM. Tables: `subjects`, `documents`, `subject_document_association`, `topics`, `subtopics`, `content_chunks`, `flashcards`. No migration framework—schema is created fresh on first run.
- **`core/background.py`** — Daemon thread manager for processing remaining chunks after the initial burst of cards is shown to the user. Background task state is in-memory and lost on restart.

### Data Storage

- **SQLite** (`nexus_v3.db`) — Relational data (subjects, topics, flashcards, etc.)
- **Qdrant** (Docker, port 6333) — Vector embeddings of content chunks for semantic search

### Flashcard Lifecycle

`pending` (generated) → mentor approves/rejects → `approved` cards appear in the Learner study room. Setting `AUTO_ACCEPT_CONTENT=true` skips HITL review.

### Streamlit Frontend (`app.py`)

Five views: **Dashboard**, **Study Materials** (upload + progress), **Mentor Review** (HITL bulk approve/reject), **Learner** (Active Recall), **System Tools** (admin).

### LangSmith Tracing

Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env` to enable tracing (project: `nexus_learner_mvp`).
