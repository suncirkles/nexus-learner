# Nexus Learner: Product Requirements Document (PRD)

## 1. Project Overview

Nexus Learner is an agentic AI system that transforms static educational materials (PDFs, scanned images) into a dynamic, personalized learning experience. It combines multi-agent AI workflows with cognitive science principles—Active Recall, Spaced Repetition, and the Feynman Technique—to ensure genuine concept mastery.

## 2. Vision

To create an AI-driven "recursive" learning environment where the system doesn't just present information but ensures mastery through active testing, structured review, and intelligent content curation.

## 3. Core Pedagogical Pillars

| Pillar | Description |
| :--- | :--- |
| **Active Recall** | No content is consumed without immediate retrieval practice (flashcards). |
| **Spaced Repetition** | Reviews are scheduled using algorithms just before the learner is likely to forget. *(Future Phase)* |
| **Feynman Technique** | Learners explain complex topics in simple terms to identify knowledge gaps. *(Future Phase)* |

---

## 4. MVP Functional Requirements (Phase 1 — Implemented)

### Epic 1: Knowledge Ingestion & Structuring

| ID | User Story | Status |
| :--- | :--- | :---: |
| 1.1 | Upload PDFs or images, extract text with OCR fallback. | ✅ |
| 1.2 | Detect and reject duplicate document uploads via content hashing. | ✅ |
| 1.3 | Chunk text and store in both relational DB and Qdrant vector store. | ✅ |
| 1.4 | Generate a meaningful document title using LLM. | ✅ |

### Epic 2: Content Curation & Hierarchy

| ID | User Story | Status |
| :--- | :--- | :---: |
| 2.1 | Automatically extract a **Topic → Subtopic** hierarchy from document text. | ✅ |
| 2.2 | Merge new content into the existing Subject hierarchy (avoid duplicates). | ✅ |
| 2.3 | Provide a Subject-based organization for all learning content. | ✅ |

### Epic 3: AI-Powered Flashcard Generation

| ID | User Story | Status |
| :--- | :--- | :---: |
| 3.1 | Generate high-quality Active Recall Q&A flashcards per chunk. | ✅ |
| 3.2 | Classify each flashcard into the correct Subtopic using LLM routing. | ✅ |
| 3.3 | Score flashcards for factual accuracy using a Critic Agent (1–5 grounding). | ✅ |
| 3.4 | Auto-reject severely hallucinated content (score < 3). | ✅ |
| 3.5 | Process an initial burst of cards synchronously, then continue in background. | ✅ |

### Epic 4: Mentor Review (Human-in-the-Loop)

| ID | User Story | Status |
| :--- | :--- | :---: |
| 4.1 | Review pending flashcards with Approve / Reject actions. | ✅ |
| 4.2 | Bulk Approve All / Reject All at **Subtopic** level. | ✅ |
| 4.3 | Bulk Approve All / Reject All at **Topic** level. | ✅ |
| 4.4 | Collapse/expand topics for efficient review. | ✅ |
| 4.5 | Restore rejected cards to pending, or permanently delete them. | ✅ |
| 4.6 | Regenerate rejected flashcards with mentor feedback. | ✅ |
| 4.7 | Get LLM-suggested answers for rejected flashcards. | ✅ |

### Epic 5: Learner Study Room

| ID | User Story | Status |
| :--- | :--- | :---: |
| 5.1 | Study approved flashcards in an Active Recall format (show/hide answer). | ✅ |
| 5.2 | Navigate to a specific subject, topic, and subtopic for focused study. | ✅ |
| 5.3 | Deep-link from Dashboard to a specific subtopic in the Learner view. | ✅ |

### Epic 6: Dashboard & Navigation

| ID | User Story | Status |
| :--- | :--- | :---: |
| 6.1 | View Subject Tiles with topic count, approved count, and pending count. | ✅ |
| 6.2 | Navigate directly from Dashboard to Learner view for a subject. | ✅ |
| 6.3 | View global stats (approved, pending, rejected counts and progress bar). | ✅ |

### Epic 7: Administration

| ID | User Story | Status |
| :--- | :--- | :---: |
| 7.1 | Global reset: wipe SQLite database and Qdrant collections. | ✅ |
| 7.2 | Edit subject names and topic names. | ✅ |
| 7.3 | Delete subjects and topics with cascading cleanup. | ✅ |

---

### Epic 8: Web Research & Automated Ingestion (Phase 2)

| ID | User Story | Status |
| :--- | :--- | :---: |
| 8.1 | Provide a list of topics (typed or uploaded) for automated research. | ✅ |
| 8.2 | Run safety checks on subjects to ensure ethical and appropriate content. | ✅ |
| 8.3 | Search trusted educational sources (Wikipedia, MDN, Python Docs, etc.). | ✅ |
| 8.4 | Scrape and clean web content for flashcard generation. | ✅ |
| 8.5 | Prevent duplicate ingestion of web pages via content hashing. | ✅ |
| 8.6 | Display source attribution (domain/URL) in review and learner views. | ✅ |

---

## 5. Future Phase Requirements

### Phase 3: Spaced Repetition & Confidence Tracking
- SM-2 algorithm-based review scheduling.
- Confidence score dashboard per topic.

### Phase 4: Feynman Module
- "Explain this concept" open-ended prompt.
- AI identifies missing keywords or logical leaps.

### Phase 5: Mentor Persona Engine
- Style profiles (tone, analogy domain, signature keywords, depth level).
- Style transfer agent to rewrite content in a mentor's voice.
- Cross-persona discovery for struggling learners.

### Phase 6: Multi-Format Assessment
- Multiple Choice, Fill-in-the-blanks, and open-ended questions.
- Source citation on failure (link to original PDF page).

---

## 6. Non-Functional Requirements

| Category | Requirement |
| :--- | :--- |
| **Quality** | All AI-generated content must be grounded in source material. Critic Agent enforces this. |
| **Security** | Input validation before LLM processing. OWASP-aligned practices. |
| **Configurability** | LLM provider and model are configurable via `.env`. Primary and routing models are separate. |
| **Observability** | LangSmith tracing enabled for all LLM calls. Background task progress is trackable. |
| **Auditability** | All flashcards retain critic scores, feedback, and status history. |

---

## 7. Technical Stack

| Component | Technology |
| :--- | :--- |
| Language | Python 3.11+ |
| Frontend | Streamlit |
| Orchestration | LangGraph (Stateful Multi-Agent Loops) |
| LLMs | OpenAI GPT-4o (Primary), GPT-4o-mini (Routing). Anthropic Claude supported. |
| Vector Database | Qdrant (Docker) |
| Relational Database | SQLite (MVP), upgradeable to PostgreSQL |
| Embeddings | OpenAI text-embedding-ada-002 |
| OCR | Tesseract via pytesseract |
| PDF Parsing | PyMuPDF (fitz) |

---
*Note: Refer to `hld.md` and the associated `.excalidraw` diagrams for detailed architecture and entity-relationship models.*

