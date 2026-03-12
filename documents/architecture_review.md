# 🏗️ Backend Architecture Review: Nexus Learner

**Reviewer**: Backend Architect Agent
**Date**: 2026-03-12
**Scope**: Full codebase — `core/`, `agents/`, `workflows/`, `app.py`, `tests/`

---

## Executive Summary

Nexus Learner is a well-structured MVP with clean separation of concerns across agents, workflows, and core infrastructure. The LangGraph-based pipeline is a strong foundation. However, there are **critical gaps in database integrity, session management, and error handling** that should be addressed before the next iteration.

| Area | Grade | Notes |
|:---|:---:|:---|
| Architecture & Separation | **A-** | Clean agent pattern, LangGraph orchestration is solid |
| Database Design | **C+** | Missing ForeignKeys, no indexes on join columns, no migrations |
| Security | **C** | No input sanitization, API keys in memory, no rate limiting |
| Error Handling | **C+** | Inconsistent; some agents swallow exceptions silently |
| Performance | **B-** | N+1 queries in dashboard, no connection pooling |
| Concurrency | **B** | Background threads work but lack proper thread safety |
| Test Coverage | **C** | Only 3 tests; no negative or edge-case coverage |

---

## 🔴 Critical Issues (Fix Before Next Release)

### 1. Missing Foreign Key Constraints
**File**: [database.py](file:///d:/projects/Gen-AI/Nexus%20Learner/core/database.py)

All relationships use plain `Integer` columns with no `ForeignKey()` declarations. This means:
- No referential integrity at the DB level — orphaned rows are possible.
- SQLAlchemy ORM relationships (`relationship()`) cannot be used.
- Cascade deletes require manual cleanup (currently done in `app.py`).

```diff
- subject_id = Column(Integer, index=True)
+ subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
```

**Impact**: Data corruption on deletes. Manual cascade logic in `app.py` lines 689–699 is fragile.

---

### 2. Database Sessions Never Closed on Exceptions
**File**: [app.py](file:///d:/projects/Gen-AI/Nexus%20Learner/app.py) — multiple functions

Sessions are opened with `db = SessionLocal()` but closed only at the end of the function. If an exception occurs mid-function, the session leaks.

```diff
- db = SessionLocal()
- subjects = db.query(Subject).all()
- ...
- db.close()

+ db = SessionLocal()
+ try:
+     subjects = db.query(Subject).all()
+     ...
+ finally:
+     db.close()
```

**Impact**: Connection pool exhaustion under load. SQLite handles this gracefully, but PostgreSQL won't.

---

### 3. Uploaded Files Written to Project Root
**File**: [app.py](file:///d:/projects/Gen-AI/Nexus%20Learner/app.py#L334)

```python
file_path = f"tmp_{uploaded_file.name}"
```

Files are written directly to the project root with user-controlled filenames. This is a **path traversal risk** and pollutes the working directory.

```diff
- file_path = f"tmp_{uploaded_file.name}"
+ import tempfile
+ file_path = os.path.join("temp_uploads", f"{uuid.uuid4()}_{uploaded_file.name}")
```

---

## 🟡 Important Issues (Address in Next Sprint)

### 4. N+1 Query Pattern in Dashboard
**File**: [app.py](file:///d:/projects/Gen-AI/Nexus%20Learner/app.py#L220-L240)

For each subject, the dashboard runs: 1 query for topic count, 1 for topics, 1 for subtopics, 2 for flashcard counts = **~5 queries per subject**. With 10 subjects, that's 50+ queries.

**Recommendation**: Use a single aggregate query with `JOIN` and `GROUP BY`:
```sql
SELECT s.id, s.name, COUNT(DISTINCT t.id), COUNT(DISTINCT CASE WHEN f.status='approved' THEN f.id END)
FROM subjects s LEFT JOIN topics t ON ... LEFT JOIN subtopics st ON ... LEFT JOIN flashcards f ON ...
GROUP BY s.id
```

---

### 5. No Database Migration Strategy
**File**: [database.py](file:///d:/projects/Gen-AI/Nexus%20Learner/core/database.py#L87)

```python
Base.metadata.create_all(bind=engine)
```

This runs on every module import. Schema changes will silently fail (new columns won't be added to existing tables). Adopt **Alembic** for proper migrations.

---

### 6. Background Thread Safety
**File**: [background.py](file:///d:/projects/Gen-AI/Nexus%20Learner/core/background.py)

The global `background_tasks` and `stop_events` dicts are accessed from both the main Streamlit thread and daemon threads without locks. This is a race condition risk.

```diff
+ import threading
+ _lock = threading.Lock()

  def run_remaining_generation(...):
-     background_tasks[doc_id] = {"status": "processing", ...}
+     with _lock:
+         background_tasks[doc_id] = {"status": "processing", ...}
```

---

### 7. Critic Agent Uses Wrong Field
**File**: [critic.py](file:///d:/projects/Gen-AI/Nexus%20Learner/agents/critic.py#L41)

```python
fc.is_approved = False  # This field doesn't exist in the schema!
```

The schema uses `fc.status` (string: "pending"/"approved"/"rejected"), but the critic sets `is_approved` (a boolean that doesn't exist). This write silently does nothing.

```diff
- fc.is_approved = False
+ fc.status = "rejected"
```

---

### 8. Temp Files Never Cleaned Up
**File**: [app.py](file:///d:/projects/Gen-AI/Nexus%20Learner/app.py#L334-L336)

After upload, `tmp_{filename}` is created but never deleted after processing completes.

```diff
+ finally:
+     if os.path.exists(file_path):
+         os.remove(file_path)
```

---

## 🟢 Suggestions (Improve Quality & Maintainability)

### 9. Move Business Logic Out of `app.py`
`app.py` is 753 lines and mixes UI rendering with database operations, session management, and business logic. Consider extracting:
- `services/subject_service.py` — CRUD for subjects/topics
- `services/flashcard_service.py` — approval, rejection, stats
- `services/ingestion_service.py` — upload + pipeline trigger

### 10. Add SQLAlchemy ORM Relationships
With ForeignKeys in place, add proper `relationship()` declarations to enable eager loading and cleaner queries:
```python
class Topic(Base):
    subtopics = relationship("Subtopic", back_populates="topic", cascade="all, delete-orphan")
```

### 11. Use Context Managers for DB Sessions
Create a reusable context manager:
```python
from contextlib import contextmanager

@contextmanager
def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 12. Add Input Validation
No validation on subject names, topic names, or uploaded filenames. Consider:
- Max length checks
- Disallowing special characters in names
- File size limits on uploads

### 13. Test Coverage Gaps
Only 3 test files exist. Missing:
- Unit tests for each agent in isolation (mock LLM calls)
- Database schema integrity tests
- Background task lifecycle tests
- Error path tests (invalid PDFs, LLM failures)

---

## Architecture Strengths ✅

1. **Clean Agent Pattern** — Each agent has a single responsibility with clear inputs/outputs.
2. **LangGraph Orchestration** — The stateful workflow is well-designed with proper conditional edges.
3. **Pluggable LLM Factory** — `get_llm()` supports swapping between OpenAI and Anthropic easily.
4. **Content Grounding** — Critic Agent with auto-rejection is a strong quality guardrail.
5. **Duplicate Detection** — SHA-256 content hashing before ingestion prevents re-processing.
6. **Configuration Management** — Pydantic Settings with `.env` file is production-ready pattern.
7. **Sync/Async Split** — Initial burst + background processing is great UX engineering.
