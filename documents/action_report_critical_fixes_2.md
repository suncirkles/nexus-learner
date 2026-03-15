# Action Report: Critical Bug Fixes — Round 2

**Date:** 2026-03-15
**Branch:** feat/fix-review-comments
**Fixes applied:** C1, C2, C5, C9, C10, C11, C14, C15, C16
**Deferred (already correct):** C12, C17

---

## C1 & C2 — `app.py`: DetachedInstanceError in `render_flashcard_review_card`

**Root cause:** The function accepted a passed-in `db` session and ORM `fc` object. After the caller's `finally` block closed the session, any attribute access on `fc` or mutation via `db.commit()` would raise `DetachedInstanceError`.

**Fix applied:**
1. Added `get_session` to the import from `core.database` (line 17).
2. Rewrote `render_flashcard_review_card` to:
   - Capture all display fields (`fc.id`, `fc.question`, `fc.answer`, `fc.critic_score`, `fc.critic_feedback`) as Python primitives at function entry, before any session state can change.
   - Replaced all mutation paths (`Approve`, `Reject`, `Discard`, `Restore`, `Delete`) to open a fresh `get_session()` context, re-fetch the flashcard by ID (`fc_id`), and commit within that scoped session.
   - Updated `suggest_answer` and `recreate_flashcard` calls to use `fc_question` / `fc_id` primitives instead of `fc` attributes.

**Files changed:** `app.py`

---

## C5 — `agents/socratic.py`: Warning when `chunk_id` is None

**Root cause:** `Flashcard.chunk_id` is nullable in the schema but flashcard generation always needs a source link for traceability. No warning was emitted when `chunk_id` was absent.

**Fix applied:** After resolving `chunk_id` from the chunk object, log a `WARNING` via `logger.warning(...)` when `chunk_id is None`, naming the affected `doc_id`.

**Files changed:** `agents/socratic.py`

---

## C9 — `agents/ingestion.py`: DetachedInstanceError from `create_document_record` return value

**Root cause:** `create_document_record` returned ORM objects (`existing_doc`, `new_doc`) that were bound to a session closed in the `finally` block. Callers accessing lazy-loaded attributes on the returned object would get `DetachedInstanceError`.

**Fix applied:** Added `db.expunge(existing_doc)` before returning the existing document, and `db.expunge(new_doc)` after `db.refresh(new_doc)` and before returning the new document. Expunging cleanly detaches the object while keeping all eagerly-loaded attributes accessible.

**Files changed:** `agents/ingestion.py`

---

## C10 — `agents/ingestion.py`: Explicit primitive capture of `db_chunk.id`

**Root cause:** `db_chunk.id` was read inline inside a `Document(metadata=...)` constructor call while the session was still open — safe but implicit. If session state ever changed between the `db.refresh(db_chunk)` and the metadata dict construction, the attribute read could trigger an unexpected lazy load.

**Fix applied:** Captured `chunk_id = db_chunk.id` as a local `int` variable immediately after `db.refresh(db_chunk)`, then used `chunk_id` in the `Document` metadata dict.

**Files changed:** `agents/ingestion.py`

---

## C11 — `agents/curator.py`: Defensive comment on `existing_topics` ORM access boundary

**Root cause:** The `existing_topics` ORM list is queried and then iterated to build `existing_structure_text`. After `db.flush()` calls later in the same function, SQLAlchemy expires those objects. A future regression that accidentally accessed `existing_topics` after the LLM call (during which session state may change) would cause a `DetachedInstanceError`.

**Fix applied:** Added a clearly marked comment immediately after the `existing_structure_text` assembly block, documenting that `existing_topics` ORM objects must not be accessed beyond that point. The current code is already correct (data is fully converted to strings before the LLM call). The comment prevents future regressions.

**Files changed:** `agents/curator.py`

---

## C12 — `agents/curator.py`: Transaction atomicity (already correct)

**Status:** No change needed. The curator already uses `db.flush()` within a single transaction with `db.commit()` at the end and `db.rollback()` in the `except` block, making it fully atomic.

---

## C14 — `agents/socratic.py`: Lazy reload after `db.commit()` in `recreate_flashcard`

**Root cause:** After `db.commit()`, SQLAlchemy expires all attributes on ORM objects. The original return statement `{"flashcard_id": fc.id, "question": fc.question, "answer": fc.answer}` would trigger lazy reloads (extra SQL round-trips) to re-fetch the expired attributes.

**Fix applied:** Captured `new_question = fc.question`, `new_answer = fc.answer`, and `fc_id = fc.id` as local variables before `db.commit()`. The return dict uses these primitives, eliminating post-commit lazy loads.

**Files changed:** `agents/socratic.py`

---

## C15 — `agents/socratic.py`: Wrong attribute access when chunk is an ORM object

**Root cause:** `generate_flashcard` used `chunk.page_content` (LangChain `Document` attribute) and `chunk.metadata.get("db_chunk_id")` for all chunk types. When called with a SQLAlchemy `ContentChunk` ORM object, `page_content` doesn't exist (use `chunk.text`), and `chunk.metadata` is `None` (not a dict — `.get()` would raise `AttributeError`).

**Fix applied:** Replaced the inline attribute access with an explicit type-dispatch block at the start of `generate_flashcard`:
- If `hasattr(chunk, 'page_content')`: treat as LangChain `Document`, use `chunk.page_content` and `chunk.metadata.get("db_chunk_id")` (with `isinstance` guard on metadata).
- Else: treat as ORM `ContentChunk`, use `chunk.text` and `chunk.id`.

Both paths produce plain primitive variables `source_text` and `chunk_id` used throughout the rest of the method.

**Files changed:** `agents/socratic.py`

---

## C16 — `workflows/phase1_ingestion.py`: DetachedInstanceError from `existing_doc` in `node_ingest`

**Root cause:** In `node_ingest` (INDEXING mode), `existing_doc` is queried and its `.id` attribute used to set `doc_id`. After the `finally: db.close()` block, `existing_doc` is a detached ORM object. Any caller or future code path accessing the ORM object after session close would raise `DetachedInstanceError`.

**Fix applied:**
- Converted `doc_id` assignment to `doc_id = str(existing_doc.id)` (explicit primitive string).
- Added `db.expunge(existing_doc)` immediately after, cleanly detaching the object before the session closes.

**Files changed:** `workflows/phase1_ingestion.py`

---

## C17 — `workflows/phase1_ingestion.py`: `pending_qdrant_docs` access (already correct)

**Status:** No change needed. All accesses to `pending_qdrant_docs` already use `.get("pending_qdrant_docs", [])` with a default, so missing keys are handled safely.

---

## Summary Table

| Bug | File | Status | Change Type |
|-----|------|--------|-------------|
| C1  | `app.py` | Fixed | Capture primitives + fresh session mutations |
| C2  | `app.py` | Fixed | Same as C1 |
| C5  | `agents/socratic.py` | Fixed | Warning log when chunk_id is None |
| C9  | `agents/ingestion.py` | Fixed | `db.expunge()` before return |
| C10 | `agents/ingestion.py` | Fixed | Capture `chunk_id` as primitive |
| C11 | `agents/curator.py` | Fixed (defensive) | Comment marking ORM access boundary |
| C12 | `agents/curator.py` | Already correct | No change |
| C14 | `agents/socratic.py` | Fixed | Capture primitives before `db.commit()` |
| C15 | `agents/socratic.py` | Fixed | Type-dispatch for LangChain Doc vs ORM chunk |
| C16 | `workflows/phase1_ingestion.py` | Fixed | `str()` cast + `db.expunge()` |
| C17 | `workflows/phase1_ingestion.py` | Already correct | No change |
