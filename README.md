# Nexus Learner

Transform educational documents into Active Recall flashcards using a multi-agent AI pipeline with human-in-the-loop review.

## What it does

Upload a PDF, image, or point it at a web topic — Nexus Learner:

1. Extracts and chunks the content
2. Builds a Topic → Subtopic knowledge hierarchy (LLM-driven)
3. Generates Active Recall Q&A flashcard pairs (SocraticAgent)
4. Grades each card for grounding quality (CriticAgent, 1–5 score)
5. Presents cards to a **Mentor** for approve/reject review
6. Approved cards are available in the **Learner** study room

## Architecture

```
Streamlit UI  →  ui/api_client.py  →  FastAPI (port 8000)
                                          ↓
                                     Services
                                          ↓
                               SQL Repos   Vector Repo
                                  ↓              ↓
                               SQLite         Qdrant
```

The LangGraph agent pipeline runs inside the FastAPI service layer. A semantic cache (Qdrant or Redis) deduplicates identical LLM calls across runs.

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- At least one LLM API key: OpenAI, Groq, or Anthropic

### Setup

```bash
git clone <repo>
cd nexus-learner

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # add your API keys
docker-compose up -d            # start Qdrant
```

### Run

Open two terminals:

```bash
# Terminal 1 — FastAPI service layer
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Streamlit frontend
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Key `.env` Settings

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | Required if using OpenAI |
| `GROQ_API_KEY` | — | Required if using Groq (free tier available) |
| `DEFAULT_LLM_PROVIDER` | `openai` | `openai` / `groq` / `anthropic` |
| `AUTO_ACCEPT_CONTENT` | `false` | Skip mentor review (useful for testing) |
| `SEMANTIC_CACHE_BACKEND` | `qdrant` | `qdrant` (default) or `redis` |
| `REDIS_URL` | `redis://localhost:6379` | Only needed for `redis` cache backend |

## Agents

| Agent | Role |
|---|---|
| IngestionAgent | PDF/image extraction, chunking, Qdrant embedding |
| CuratorAgent | LLM-driven Topic→Subtopic hierarchy |
| TopicAssignerAgent | Maps chunks to subtopics |
| TopicMatcherAgent | Semantic matching of topics to subtopics |
| RelevanceAgent | Filters chunks by topic relevance |
| SocraticAgent | Generates Active Recall Q&A pairs |
| CriticAgent | Grades flashcard grounding (1–5) |
| WebResearcherAgent | Scrapes and deduplicates web content |

## Running Tests

```bash
PYTHONPATH=. pytest tests/unit/ -v     # fast unit tests (no LLM calls)
PYTHONPATH=. pytest tests/ -v          # all tests (requires API keys)
```

## Stack

- **LangGraph** — agent orchestration
- **LangChain** — LLM abstraction, embeddings
- **FastAPI + uvicorn** — service layer REST API
- **Streamlit** — frontend UI
- **SQLAlchemy + SQLite** — relational data
- **Qdrant** — vector store (embeddings + semantic cache)
- **Redis** — optional persistent semantic cache backend
- **PyMuPDF + Tesseract** — PDF and image extraction
- **LangSmith** — optional tracing (set `LANGCHAIN_TRACING_V2=true`)
