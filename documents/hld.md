# Nexus Learner: High Level Design (HLD)

## 1. System Overview

Nexus Learner is a multi-agent AI platform that converts static educational documents into interactive Active Recall flashcards. The system uses a pipeline of specialized AI agents orchestrated by LangGraph, with a Streamlit-based UI for human-in-the-loop review.

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
│  Workflow   │──▶│  (4 agents)  │   │  (threading)     │
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
- **Study Materials** – File upload, ingestion pipeline trigger.
- **Mentor Review** – HITL flashcard approval workflow with bulk actions.
- **Learner** – Active Recall study interface.
- **System Tools** – Admin controls (reset, rename, delete).

### 2.2 Application Layer (`app.py`)
Handles routing, session state, and database queries for the UI. Uses `st.fragment` for localized re-renders to minimize full-page reruns.

### 2.3 Agent Layer (`agents/`)
Four specialized agents, each with a single responsibility:

| Agent | File | Purpose | LLM |
| :--- | :--- | :--- | :--- |
| **Ingestion Agent** | `agents/ingestion.py` | PDF/image extraction, chunking, dedup, vector embedding | Embeddings only |
| **Curator Agent** | `agents/curator.py` | Topic/Subtopic hierarchy extraction and merge | Primary |
| **Socratic Agent** | `agents/socratic.py` | Active Recall Q&A generation, recreation, suggestion | Primary |
| **Critic Agent** | `agents/critic.py` | Grounding evaluation (1–5 scoring) | Primary |

### 2.4 Workflow Layer (`workflows/`)
LangGraph-based stateful workflow defining the processing pipeline:

```
START → Ingest → Curate → Generate → Critic → [Continue?]
                                                  │
                                          Yes ────► Increment → Generate (loop)
                                          No  ────► END
```

### 2.5 Infrastructure Layer (`core/`)

| Module | Responsibility |
| :--- | :--- |
| `core/config.py` | Pydantic Settings, .env loading |
| `core/models.py` | LLM factory (`get_llm()`) |
| `core/database.py` | SQLAlchemy ORM models & session |
| `core/background.py` | Thread-based background task manager |

---

## 3. Data Model (ERD)

```
┌──────────┐     ┌───────────┐     ┌──────────┐     ┌───────────┐     ┌───────────┐
│ Subject  │────▶│  Document │     │  Topic   │────▶│ Subtopic  │────▶│ Flashcard │
│──────────│     │───────────│     │──────────│     │───────────│     │───────────│
│ id (PK)  │     │ id (PK)   │     │ id (PK)  │     │ id (PK)   │     │ id (PK)   │
│ name     │     │ subject_id│     │ subject_id│    │ topic_id  │     │ subtopic_id│
│ created  │     │ filename  │     │ doc_id   │     │ name      │     │ question  │
└──────────┘     │ title     │     │ name     │     │ summary   │     │ answer    │
                 │ hash      │     │ summary  │     └───────────┘     │ critic_*  │
                 └───────────┘     └──────────┘                       │ status    │
                                                                      └───────────┘
                 ┌───────────────┐
                 │ ContentChunk  │
                 │───────────────│
                 │ id (PK)       │     ┌──────────────────────┐
                 │ document_id   │────▶│ Qdrant Vector Store  │
                 │ text          │     │ (nexus_chunks)       │
                 └───────────────┘     └──────────────────────┘
```

### Key Relationships
- **Subject** 1→N **Document** (via `subject_id`)
- **Subject** 1→N **Topic** (via `subject_id`)
- **Topic** 1→N **Subtopic** (via `topic_id`)
- **Subtopic** 1→N **Flashcard** (via `subtopic_id`)
- **Document** 1→N **ContentChunk** (via `document_id`)
- **ContentChunk** → **Qdrant** (vector embeddings indexed by `document_id`)

---

## 4. Data Flow: Document Ingestion Pipeline

```
User uploads PDF
       │
       ▼
[1. Ingestion Agent]
  - Extract text (PyMuPDF + OCR fallback)
  - Compute SHA-256 hash → reject duplicates
  - Chunk text (1000 chars, 200 overlap)
  - Save chunks to SQLite (ContentChunk)
  - Embed chunks into Qdrant
       │
       ▼
[2. Curator Agent]
  - Analyze full text → extract Topic/Subtopic hierarchy
  - Merge with existing Subject structure
  - Persist to SQLite (Topic, Subtopic)
       │
       ▼
[3. Socratic Agent] ◄──── (loops per chunk)
  - Generate Active Recall Q&A
  - Classify into correct Subtopic via LLM routing
  - Save Flashcard to SQLite (status: "pending")
       │
       ▼
[4. Critic Agent]
  - Score flashcard against source text (1–5)
  - Store critic_score and critic_feedback
  - Auto-reject if score < 3
       │
       ▼
[Loop or End]
  - If more chunks → Increment index → back to step 3
  - If sync limit reached → hand off to Background Thread
  - If all done → END
```

---

## 5. Sync/Async Processing Model

To provide immediate feedback to the user while processing large documents:

1. **Sync Phase** (first 5 chunks): Processed in the main Streamlit thread. Progress bar is updated in real-time. Flashcards are immediately visible in Mentor Review.
2. **Async Phase** (remaining chunks): Handed off to a daemon thread via `core/background.py`. Progress is tracked in a global registry. User can stop or monitor background tasks from the Study Materials tab.

---

## 6. Security & Quality Guardrails

| Guardrail | Implementation |
| :--- | :--- |
| Content Grounding | Critic Agent scores all flashcards against source text |
| Auto-Rejection | Flashcards with critic score < 3 are flagged |
| Duplicate Detection | SHA-256 content hashing prevents re-ingestion |
| Human Review | All content defaults to "pending" status (configurable) |
| LLM Observability | LangSmith tracing for all agent calls |

---

## 7. Technology Stack

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

---

## 8. Project Structure

```
nexus-learner/
├── app.py                          # Streamlit frontend (all views)
├── agents/
│   ├── ingestion.py                # Document parsing, chunking, embedding
│   ├── curator.py                  # Topic/Subtopic hierarchy extraction
│   ├── socratic.py                 # Flashcard generation & recreation
│   └── critic.py                   # Grounding evaluation (1-5 scoring)
├── core/
│   ├── config.py                   # Pydantic Settings (.env loading)
│   ├── models.py                   # LLM factory (get_llm)
│   ├── database.py                 # SQLAlchemy ORM models
│   └── background.py              # Background thread manager
├── workflows/
│   └── phase1_ingestion.py         # LangGraph pipeline definition
├── tests/
│   ├── test_e2e.py                 # End-to-end workflow test
│   ├── test_duplicates.py          # Duplicate detection test
│   └── test_topic_actions.py       # Bulk action unit test
├── documents/
│   ├── prd.md                      # Product Requirements Document
│   └── hld.md                      # High Level Design (this file)
├── docker-compose.yml              # Qdrant service definition
├── requirements.txt                # Python dependencies
└── .env                            # Environment variables (API keys)
```
