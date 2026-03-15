# Nexus Learner — Bug Backlog

> Generated: 2026-03-15
> Branch: feat/topic-filtering
> Total: **22 Critical · 28 High · 28 Medium · 2 Low = 80 issues**

---

## CRITICAL (22)

| # | File | Issue |
|---|------|-------|
| C1 | `app.py` | `DetachedInstanceError` in `render_flashcard_review_card` — flashcard `fc` modified after session closed |
| C2 | `app.py` | Direct model mutation on detached object: `fc.status = "approved"` will fail without re-fetch |
| C3 | `app.py` | Race condition on `background_tasks` dict — deletion while background thread may be reading |
| C4 | `core/database.py` | Schema migration runs `ALTER TABLE` without per-statement rollback — partial migrations leave DB in broken state |
| C5 | `core/database.py` | `Flashcard.chunk_id` is nullable but generation logic requires it — allows orphaned flashcards |
| C6 | `core/background.py` | Race condition: `background_tasks[doc_id]` state update happens outside lock after initial creation |
| C7 | `core/background.py` | Incomplete state key initialization — missing keys cause `KeyError` when UI reads progress |
| C8 | `agents/ingestion.py` | Hash collision: SHA256(first 10k chars + basename) — two different PDFs with same intro produce same hash; second PDF silently skipped |
| C9 | `agents/ingestion.py` | Session leak — `SessionLocal()` not always closed if exception occurs before `finally` |
| C10 | `agents/ingestion.py` | `DetachedInstanceError` after `db_chunk` commit — `db_chunk.id` accessed outside session |
| C11 | `agents/curator.py` | `DetachedInstanceError` — `existing_topics` queried, then subtopics accessed on potentially detached topic objects |
| C12 | `agents/curator.py` | Multiple inserts without atomic transaction — partial hierarchy left on error |
| C13 | `agents/socratic.py` | `Flashcard` created without `subject_id` — orphaned flashcards not visible to subject-scoped queries |
| C14 | `agents/socratic.py` | `DetachedInstanceError` in `recreate_flashcard` — `fc` from query modified after detachment |
| C15 | `agents/socratic.py` | `chunk.metadata.get()` assumes LangChain `Document`; fails on `ContentChunk` ORM object from DB |
| C16 | `workflows/phase1_ingestion.py` | `doc_id` reassigned from `existing_doc.id` on first page but not persisted to state — subsequent pages use wrong ID |
| C17 | `workflows/phase1_ingestion.py` | `pending_qdrant_docs` not initialized in state — `KeyError` on unexpected routing |
| C18 | `workflows/phase1_ingestion.py` | `_flush_qdrant_batch` called without Qdrant availability check — hangs thread on connection failure |
| C19 | `workflows/phase2_web_ingestion.py` | `DetachedInstanceError` in `node_ingest_web_document` — `existing` doc used outside session |
| C20 | `workflows/phase2_web_ingestion.py` | Chunk/Qdrant split atomicity — chunks written to SQLite but if Qdrant fails they're orphaned (no vectors, no semantic search) |
| C21 | `workflows/phase2_web_ingestion.py` | Document and `SubjectDocumentAssociation` created in two separate commits — orphaned document if second commit fails |
| C22 | `agents/critic.py` | *(re-numbered — see H17/H18)* |

---

## HIGH (28)

| # | File | Issue |
|---|------|-------|
| H1 | `app.py` | Missing cascade validation before `db.delete(doc)` — orphaned chunks/topics remain in Qdrant |
| H2 | `app.py` | `background_tasks.items()` iterated without lock — concurrent modification possible |
| H3 | `app.py` | Session closed but flashcard objects used in nested callback functions after close |
| H4 | `app.py` | `db.query(Flashcard).filter(...).update(...)` without `synchronize_session=False` — session cache desync |
| H5 | `core/database.py` | No unique constraint on `Topic.name` per document — duplicate topic names allowed |
| H6 | `core/database.py` | `Flashcard` cascades delete when `Subtopic` deleted — silent data loss on topic removal |
| H7 | `core/database.py` | Foreign key columns (`document_id`, `topic_id`, `subtopic_id`) lack database indexes — slow JOINs |
| H8 | `core/background.py` | No timeout on `phase1_graph.stream()` — daemon threads accumulate and exhaust resources if LLM/Qdrant hangs |
| H9 | `core/background.py` | Temp file cleanup catches all exceptions and returns success — disk space leak on locked files |
| H10 | `agents/ingestion.py` | `pytesseract` not guarded at import — `ImportError` on systems without Tesseract binary |
| H11 | `agents/ingestion.py` | OCR fallback returns empty string silently — empty chunks indexed without warning |
| H12 | `agents/ingestion.py` | Subject association created without verifying subject exists — FK violation on concurrent delete |
| H13 | `agents/curator.py` | LLM-generated topic/subtopic names not validated for duplicates within same response — duplicates created |
| H14 | `agents/curator.py` | No retry on LLM call failure — single failure aborts full hierarchy extraction |
| H15 | `agents/socratic.py` | LLM temperature hard-coded at 0.3 — should use `get_llm()` factory for consistency |
| H16 | `agents/socratic.py` | `AUTO_ACCEPT` status set during generation, but `CriticAgent` overrides to `rejected` silently — no notification |
| H17 | `agents/critic.py` | Auto-reject (`score < 3`) overrides `AUTO_ACCEPT_CONTENT` setting — user expectation violated |
| H18 | `agents/critic.py` | `critic_score` may be `None` or outside `[1,5]` — no bounds validation before comparison |
| H19 | `agents/topic_assigner.py` | Fallback assignment uses "General Overview" / "Introduction" — violates the agent's own "No General Content" policy |
| H20 | `agents/topic_assigner.py` | No retry on structured output parse failure — falls back silently |
| H21 | `workflows/phase1_ingestion.py` | `ContentChunk.subtopic_id.in_()` query with lazy-loaded relationship — possible `LazyLoader` error |
| H22 | `workflows/phase1_ingestion.py` | Incremental dedup skips chunks for subtopics with rejected cards — flashcard gaps after mentor rejection |
| H23 | `workflows/phase1_ingestion.py` | Hierarchy dict stores only names, not IDs — `topic_id`/`subtopic_id` not available for downstream use |
| H24 | `workflows/phase2_web_ingestion.py` | Qdrant embedding failure is non-fatal — generation proceeds without semantic context |
| H25 | `workflows/phase2_web_ingestion.py` | `node_curate` returns empty dict on missing `full_text` — no status message, silent skip |
| H26 | `workflows/phase2_web_ingestion.py` | Subtopic classification falls back to first subtopic on error — no logging, wrong assignment |
| H27 | `app.py` | N+1 query in mentor review — per-subtopic flashcard count queried in loop |
| H28 | `core/background.py` | `sum(1 for f in flashcards if f.get("status") == "success")` — SocraticAgent dict doesn't always have `status` key |

---

## MEDIUM (28)

| # | File | Issue |
|---|------|-------|
| M1 | `app.py` | `s.topic.name` in list comprehension may trigger lazy load on detached `Subtopic` |
| M2 | `app.py` | `SYNC_TOPIC_LIMIT` hard-coded — should be in `core/config.py` `Settings` |
| M3 | `app.py` | No null check before accessing doc attributes after filter query |
| M4 | `core/database.py` | `default=lambda: datetime.now(timezone.utc)` — evaluated at Python level, not DB level; use `server_default` |
| M5 | `core/database.py` | Migration not atomic — if one `ALTER TABLE` fails, earlier ones already committed |
| M6 | `core/database.py` | No constraint that `ContentChunk.subtopic_id` belongs to a subtopic under the chunk's document |
| M7 | `core/background.py` | Stop signal checked but state partially updated outside critical section |
| M8 | `core/background.py` | Broad `except Exception` in worker masks programming errors |
| M9 | `agents/ingestion.py` | `text_splitter.split_text()` can return empty list — no guard |
| M10 | `agents/curator.py` | Hard-coded 15,000-char truncation loses context for large documents |
| M11 | `agents/curator.py` | LLM invocation not logged — hard to debug failures |
| M12 | `agents/socratic.py` | Missing chunk validation — if chunk not found, `source_text` silently set to fallback string |
| M13 | `agents/critic.py` | Error response `{"error": "Flashcard not found"}` returned but callers don't check it |
| M14 | `agents/topic_assigner.py` | `.strip()` called on potentially `None` LLM output fields |
| M15 | `agents/topic_matcher.py` | Empty indexed subtopics returns empty results — no distinction between "no topics exist" vs "matcher failed" |
| M16 | `agents/topic_matcher.py` | User topic string implicitly truncated — no feedback if topics too long |
| M17 | `agents/relevance.py` | Default `is_relevant=True` on errors — junk chunks pass through, degrading flashcard quality |
| M18 | `workflows/phase1_ingestion.py` | `query.all()` returning empty list gives no explicit log message |
| M19 | `workflows/phase1_ingestion.py` | Hierarchy structure inconsistent across pages — no merge/dedup of topics already seen |
| M20 | `workflows/phase2_web_ingestion.py` | Hard-coded 1000-char chunk size — should be configurable |
| M21 | `workflows/phase2_web_ingestion.py` | URL dedup tracked but chunk content not deduped — duplicate content indexed |
| M22 | `app.py` | No audit log for document deletion |
| M23 | `app.py` | Confirm-reset state persists in `st.session_state` even after successful reset |
| M24 | `core/database.py` | No index on `flashcards.status` — full table scan on mentor review filter queries |
| M25 | `core/database.py` | No index on `flashcards.subject_id` — slow subject-scoped flashcard lookups |
| M26 | `agents/ingestion.py` | `pixmap` → PIL conversion not validated — silent failure on corrupted pages |
| M27 | `workflows/phase2_web_ingestion.py` | Phase 2 reuses Phase 1 node naming conventions but mixes mode logic — harder to maintain |
| M28 | `core/background.py` | In-memory task state lost on app restart — no persistence or recovery mechanism |

---

## LOW (2)

| # | File | Issue |
|---|------|-------|
| L1 | `app.py` | Generic error messages don't distinguish Qdrant vs SQLite failure |
| L2 | `core/database.py` | Inconsistent timestamp handling — some columns use lambda default, some omit entirely |

---

## Summary by File

| File | Critical | High | Medium | Low | Total |
|------|----------|------|--------|-----|-------|
| `app.py` | 3 | 5 | 5 | 1 | **14** |
| `core/database.py` | 2 | 3 | 5 | 1 | **11** |
| `core/background.py` | 2 | 3 | 3 | 0 | **8** |
| `agents/ingestion.py` | 3 | 3 | 2 | 0 | **8** |
| `agents/curator.py` | 2 | 2 | 2 | 0 | **6** |
| `agents/socratic.py` | 3 | 2 | 1 | 0 | **6** |
| `agents/critic.py` | 0 | 2 | 1 | 0 | **3** |
| `agents/topic_assigner.py` | 0 | 2 | 1 | 0 | **3** |
| `agents/topic_matcher.py` | 0 | 0 | 2 | 0 | **2** |
| `agents/relevance.py` | 0 | 0 | 1 | 0 | **1** |
| `workflows/phase1_ingestion.py` | 3 | 3 | 2 | 0 | **8** |
| `workflows/phase2_web_ingestion.py` | 3 | 4 | 3 | 0 | **10** |
| **TOTAL** | **21** | **29** | **28** | **2** | **80** |
