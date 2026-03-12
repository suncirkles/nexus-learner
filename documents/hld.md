# Nexus Learner: High Level Design (HLD)

## 1. System Overview

Nexus Learner is a multi-agent AI platform that converts static educational documents and web content into interactive Active Recall flashcards. The system uses a pipeline of specialized AI agents orchestrated by LangGraph, with a Streamlit-based UI for human-in-the-loop review.

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                     │
│  Dashboard │ Study Materials │ Mentor Review │ Learner   │
└──────┬───────────────┬──────────────┬────────────────────┘
       │               │              │
       ▼               ▼              ▼
┌──────────────────────────────────────────────────────────┐
│                  Application Layer (app.py)               │
│  render_dashboard │ render_study_materials │ ...          │
└──────────────────────────┬───────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
│  LangGraph  │   │   Agents     │   │ Background Tasks │
│  Workflow   │──▶│  (7 agents)  │   │  (threading)     │
└─────────────┘   └──────┬───────┘   └──────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌───────────┐  ┌───────────┐  ┌───────────┐
   │  SQLite   │  │  Qdrant   │  │  LLM APIs │
   │  (ORM)    │  │ (Vectors) │  │ (OpenAI)  │
   └───────────┘  └───────────┘  └───────────┘
```

---

## 2. Architecture Layers

### 2.1 Presentation Layer (Streamlit)
Single-page application with sidebar navigation and five views:
- **Dashboard** – Subject tiles with aggregated stats.
- **Study Materials** – File upload and Web Research trigger.
- **Mentor Review** – HITL flashcard approval workflow with bulk actions.
- **Learner** – Active Recall study interface with source attribution.
- **System Tools** – Admin controls (reset, rename, archive, delete).

### 2.2 Application Layer (`app.py`)
Handles routing, session state, and database queries for the UI. Uses `st.fragment` for localized re-renders. Provides specialized tabs for Document Upload vs. Web Research.

### 2.3 Agent Layer (`agents/`)
Seven specialized agents, each with a single responsibility:

| Agent | File | Purpose | LLM |
| :--- | :--- | :--- | :--- |
| **Ingestion Agent** | `agents/ingestion.py` | PDF/image extraction, chunking, dedup, vector embedding | Embeddings only |
| **Safety Agent** | `agents/safety.py` | Subject and content safety screening | Primary |
| **Web Research Agent** | `agents/web_researcher.py` | Search trusted sources, scrape and clean web content | N/A |
| **Topic Parser Agent** | `agents/topic_parser.py` | Extract specific topics from text or files | Primary |
| **Curator Agent** | `agents/curator.py` | Topic/Subtopic hierarchy extraction and merge | Primary |
| **Socratic Agent** | `agents/socratic.py` | Active Recall Q&A generation, recreation, suggestion | Primary |
| **Critic Agent** | `agents/critic.py` | Grounding evaluation (1–5 scoring) | Primary |

### 2.4 Workflow Layer (`workflows/`)
LangGraph-based stateful workflows defining the processing pipelines:

```
[Phase 1: Documents] START → Ingest → Curate → Generate → Critic → [Continue?]
                                                                    │
                                                            Yes ────► Increment ──┐
                                                            No  ────► END      ◄──┘

[Phase 2: Web]       START → Safety → Research → Ingest Web → Curate → Generate → Critic 
                                                                                  │
                                                            Yes ────◄─────────────┘ (loop)
                                                            No  ────► next_document? → END
```

### 2.5 Infrastructure Layer (`core/`)

| Module | Responsibility |
| :--- | :--- |
| `core/config.py` | Pydantic Settings, .env loading, Scraper settings |
| `core/models.py` | LLM factory (`get_llm()`) |
| `core/database.py` | SQLAlchemy ORM models, migrations, and session management |
| `core/background.py` | Thread-based background task manager for PDF and Web tasks |

---

## 3. Data Model (ERD)

```
┌──────────┐     ┌───────────┐     ┌──────────┐     ┌───────────┐     ┌───────────┐
│ Subject  │────▶│  Document │     │  Topic   │────▶│ Subtopic  │────▶│ Flashcard │
│──────────│     │───────────│     │──────────│     │───────────│     │───────────│
│ id (PK)  │     │ id (PK)   │     │ id (PK)  │     │ id (PK)   │     │ id (PK)   │
│ name     │     │ subject_id│     │ subject_id│    │ topic_id  │     │ subtopic_id│
│ is_archived    │ filename  │     │ doc_id   │     │ name      │     │ question  │
└──────────┘     │ source_url│     │ name     │     │ summary   │     │ answer    │
                 │ hash      │     │ summary  │     └───────────┘     │ critic_*  │
                 └───────────┘     └──────────┘                       │ status    │
                                                                      └───────────┘
                 ┌───────────────┐
                 │ ContentChunk  │
                 │───────────────│
                 │ id (PK)       │     ┌──────────────────────┐
                 │ document_id   │────▶│ Qdrant Vector Store  │
                 │ text          │     │ (nexus_chunks)       │
                 │ source_url    │     └──────────────────────┘
                 └───────────────┘
```

---

## 4. Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| Frontend | Streamlit | Rapid UI prototyping |
| Orchestration | LangGraph | Stateful multi-agent workflow |
| Primary LLM | OpenAI GPT-4o | Content generation, curation, evaluation |
| Routing LLM | OpenAI GPT-4o-mini | Chunk-to-subtopic classification |
| Vector DB | Qdrant (Docker) | Semantic search and RAG |
| Relational DB | SQLite (via SQLAlchemy) | Structured data persistence |
| Embeddings | OpenAI text-embedding-ada-002 | Document vectorization |
| OCR | Tesseract (via pytesseract) | Scanned document support |
| PDF Parsing | PyMuPDF (fitz) | Text extraction from PDFs |
| Search | DuckDuckGo (ddgs) | Web Research search engine |

---

## 5. Project Structure

```
nexus-learner/
├── app.py                          # Streamlit frontend (all views)
├── agents/
│   ├── safety.py                   # Safety & Relevance screening
│   ├── web_researcher.py           # Scraping & Search
│   ├── curator.py                  # Topic/Subtopic hierarchy extraction
│   ├── socratic.py                 # Flashcard generation
│   └── critic.py                   # Grounding evaluation (1-5 scoring)
├── core/
│   ├── config.py                   # Pydantic Settings
│   ├── database.py                 # SQLAlchemy ORM models & migrations
│   └── background.py              # Background task manager
├── workflows/
│   ├── phase1_ingestion.py         # LangGraph pipeline for PDFs/Images
│   └── phase2_web_ingestion.py     # LangGraph pipeline for Web Content
├── tests/
│   ├── test_e2e.py                 # End-to-end PDF workflow test
│   ├── test_web_ingestion.py       # End-to-end Web workflow test
│   └── test_topic_actions.py       # Bulk action unit test
├── documents/
│   ├── prd.md                      # Product Requirements Document
│   └── hld.md                      # High Level Design (this file)
└── requirements.txt                # Python dependencies
```
