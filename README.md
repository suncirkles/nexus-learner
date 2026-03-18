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

[View System Architecture Diagram (Excalidraw)](documents/architecture.excalidraw)

*The system uses LangGraph to orchestrate a pipeline of specialized AI agents within a FastAPI service layer.*


The LangGraph agent pipeline runs inside the FastAPI service layer. A semantic cache (Qdrant or Redis) deduplicates identical LLM calls across runs.

## Quick Start

## Setup and Run

### 1. Requirements

- Python 3.11+ (if running locally)
- Docker Desktop (if running via Docker Compose)
- Accounts/API keys for LLM providers:
  - OpenAI API key (required for core logic)
  - Groq, Anthropic, Google Gemini, and/or DeepSeek keys (optional)

### 2. Environment Configuration

Copy `.env.example` to `.env` and fill out your keys:

```bash
cp .env.example .env
```

You **must** supply `OPENAI_API_KEY` to run the baseline configuration. Add others if you'd like to test model availability checking or agent routing via LiteLLM.

### 3. Run via Docker Compose (Recommended)

The easiest way to spin up the entire application—which includes the vector database (Qdrant), semantic cache (Redis), FastAPI backend, and Streamlit frontend—is via Docker Compose:

```bash
docker-compose up --build -d
```

Once started:

- Streamlit UI: <http://localhost:8501>
- FastAPI Swagger UI: <http://localhost:8000/docs>
- Qdrant DB logs are stored gracefully within the volume mount.
- Note: Any changes to requirements or structural code will require another `--build`.

**Useful Docker Commands:**
- View logs for all services: `docker-compose logs -f`
- View logs for a specific service: `docker-compose logs -f api` or `docker-compose logs -f ui`
- Shut down the stack: `docker-compose down`

### 4. Run Locally (Alternative)

If you prefer to run the components independently or for localized debugging:

First install the underlying tools (like Tesseract OCR) and the python dependencies:

```bash
pip install -r requirements.txt
```

Start up the local Qdrant container:

```bash
docker-compose up -d qdrant redis
```

Start the FastAPI backend:

```bash
uvicorn api.main:app --reload --port 8000
```

Start the Streamlit frontend UI:

```bash
streamlit run app.py
```

## Acknowledgements

Special thanks to the open-source community around AI agent development. In particular, the foundation of our multi-agent capabilities was significantly inspired by and utilizes components from **[msitarzewski/agency-agents](https://github.com/msitarzewski/agency-agents)**.

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
