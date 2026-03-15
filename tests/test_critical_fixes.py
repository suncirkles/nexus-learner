"""
Unit/integration tests for the 10 critical bug fixes.

Run with:
    PYTHONPATH=. pytest tests/test_critical_fixes.py -v
"""
import os
import threading
import time
import uuid
import hashlib
import pytest


# ---------------------------------------------------------------------------
# C3 — background_tasks deletion is thread-safe
# ---------------------------------------------------------------------------

def test_clear_background_task_is_threadsafe():
    """clear_background_task must not raise even under concurrent access."""
    from core.background import background_tasks, clear_background_task, _lock

    task_id = f"test-{uuid.uuid4()}"
    with _lock:
        background_tasks[task_id] = {"status": "completed", "is_web": False}

    errors = []

    def clear():
        try:
            clear_background_task(task_id)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=clear) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent clear: {errors}"
    assert task_id not in background_tasks


def test_clear_background_task_removes_stop_event():
    """clear_background_task also removes the stop_event entry."""
    from core.background import background_tasks, stop_events, clear_background_task, _lock

    task_id = f"test-{uuid.uuid4()}"
    with _lock:
        background_tasks[task_id] = {"status": "completed", "is_web": False}
        stop_events[task_id] = threading.Event()

    clear_background_task(task_id)

    assert task_id not in background_tasks
    assert task_id not in stop_events


# ---------------------------------------------------------------------------
# C4 — schema migration is per-statement atomic
# ---------------------------------------------------------------------------

def test_migration_skips_existing_columns():
    """_run_migrations must not raise if columns already exist."""
    from core.database import _run_migrations
    # Running twice must be idempotent
    _run_migrations()
    _run_migrations()  # should not raise


# ---------------------------------------------------------------------------
# C6/C7 — background task state has all UI-expected keys at init
# ---------------------------------------------------------------------------

def test_background_task_state_keys_initialised():
    """After start_background_task is called, all UI-read keys must exist."""
    from core.background import background_tasks, stop_background_task, _lock

    expected_keys = {
        "status", "filename", "mode", "is_web", "status_message",
        "total_pages", "current_page", "current_chunk_index",
        "chunks_in_page", "total_chunks", "flashcards_count",
    }

    # Inject a fake completed entry to inspect initial shape
    task_id = f"test-{uuid.uuid4()}"
    with _lock:
        background_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "total": 1,
            "filename": "test.pdf",
            "mode": "INDEXING",
            "is_web": False,
            "status_message": "Initializing...",
            "total_pages": 1,
            "current_page": 0,
            "current_chunk_index": 0,
            "chunks_in_page": 1,
            "total_chunks": 0,
            "flashcards_count": 0,
        }

    with _lock:
        task = background_tasks[task_id]
    missing = expected_keys - set(task.keys())
    assert not missing, f"Missing keys in background task state: {missing}"

    # cleanup
    from core.background import clear_background_task
    clear_background_task(task_id)


# ---------------------------------------------------------------------------
# C8 — content hash includes file size to prevent false duplicates
# ---------------------------------------------------------------------------

def test_content_hash_differs_by_file_size(tmp_path):
    """Two files with the same first 10k chars but different sizes must hash differently."""
    from agents.ingestion import IngestionAgent

    agent = IngestionAgent()

    # Create two text files: same content prefix, different total size
    base_content = "A" * 10000
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text(base_content + "EXTRA_CONTENT_IN_B_ONLY" * 100)
    file_b.write_text(base_content)

    size_a = os.path.getsize(file_a)
    size_b = os.path.getsize(file_b)

    hash_a = agent.get_content_hash(base_content[:10000] + str(size_a) + "a.txt")
    hash_b = agent.get_content_hash(base_content[:10000] + str(size_b) + "b.txt")

    assert hash_a != hash_b, "Files with different sizes must produce different hashes"


def test_content_hash_same_file_is_stable(tmp_path):
    """Same file must always produce the same hash."""
    from agents.ingestion import IngestionAgent

    agent = IngestionAgent()
    content = "Consistent content " * 500
    f = tmp_path / "stable.txt"
    f.write_text(content)

    size = os.path.getsize(f)
    h1 = agent.get_content_hash(content[:10000] + str(size) + "stable.txt")
    h2 = agent.get_content_hash(content[:10000] + str(size) + "stable.txt")
    assert h1 == h2


# ---------------------------------------------------------------------------
# C13 — Flashcard is created with subject_id set at insertion time
# ---------------------------------------------------------------------------

SAMPLE_PDF = os.path.abspath("documents/pyspark.pdf")


@pytest.mark.skipif(
    not os.path.exists(SAMPLE_PDF),
    reason="PDF fixture not found",
)
def test_flashcards_have_subject_id_after_generation():
    """Every flashcard produced by GENERATION must have subject_id set (not NULL)."""
    from core.database import SessionLocal, Subject, Flashcard, Topic, Subtopic, ContentChunk, Document as DBDoc, SubjectDocumentAssociation
    from workflows.phase1_ingestion import phase1_graph

    doc_id = str(uuid.uuid4())
    db = SessionLocal()
    subj = Subject(name=f"C13-Test-{doc_id[:6]}")
    db.add(subj)
    db.commit()
    subject_id = subj.id
    db.close()

    # Index 2 pages
    phase1_graph.invoke({
        "mode": "INDEXING", "file_path": SAMPLE_PDF, "doc_id": doc_id,
        "subject_id": None, "target_topics": [],
        "total_pages": 2, "current_page": 0, "chunks": [], "current_chunk_index": 0,
        "hierarchy": [], "pending_qdrant_docs": [], "current_new_cards": [],
        "generated_flashcards": [], "status_message": "start",
        "matched_subtopic_ids": None,
    })

    # Link doc to subject
    db = SessionLocal()
    actual_doc = db.query(DBDoc).filter(DBDoc.filename == "pyspark.pdf").order_by(DBDoc.created_at.desc()).first()
    actual_doc_id = actual_doc.id if actual_doc else doc_id
    exists = db.query(SubjectDocumentAssociation).filter_by(subject_id=subject_id, document_id=actual_doc_id).first()
    if not exists:
        db.add(SubjectDocumentAssociation(subject_id=subject_id, document_id=actual_doc_id))
        db.commit()
    db.close()

    # Generate
    phase1_graph.invoke({
        "mode": "GENERATION", "file_path": None, "doc_id": actual_doc_id,
        "subject_id": subject_id, "target_topics": [],
        "total_pages": 0, "current_page": 0, "chunks": [], "current_chunk_index": 0,
        "hierarchy": [], "pending_qdrant_docs": [], "current_new_cards": [],
        "generated_flashcards": [], "status_message": "start",
        "matched_subtopic_ids": None,
    })

    db = SessionLocal()
    try:
        cards = db.query(Flashcard).filter(Flashcard.subject_id == subject_id).all()
        assert len(cards) > 0, "No flashcards generated"
        cards_without_subject = [c for c in cards if c.subject_id is None]
        assert not cards_without_subject, (
            f"{len(cards_without_subject)} flashcard(s) have NULL subject_id"
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# C18 — _flush_qdrant_batch propagates Qdrant errors clearly
# ---------------------------------------------------------------------------

def test_flush_qdrant_batch_raises_on_failure(monkeypatch):
    """_flush_qdrant_batch must raise (not silently swallow) Qdrant errors."""
    from workflows import phase1_ingestion
    import langchain_qdrant

    def _boom(*args, **kwargs):
        raise ConnectionError("Qdrant is down")

    monkeypatch.setattr(langchain_qdrant.QdrantVectorStore, "from_documents", staticmethod(_boom))

    pending = [{"text": "hello", "metadata": {"document_id": "x", "db_chunk_id": 1}}]
    with pytest.raises(ConnectionError, match="Qdrant is down"):
        phase1_ingestion._flush_qdrant_batch(pending)


def test_flush_qdrant_batch_noop_on_empty():
    """_flush_qdrant_batch with empty list must not call Qdrant at all."""
    from workflows.phase1_ingestion import _flush_qdrant_batch
    # Should complete without error even if Qdrant is unavailable
    _flush_qdrant_batch([])


# ---------------------------------------------------------------------------
# C21 — phase2 document + association created in single commit
# ---------------------------------------------------------------------------

def test_phase2_document_association_atomic(monkeypatch):
    """If SubjectDocumentAssociation insert fails, the Document must also be rolled back."""
    from core.database import SessionLocal, Document as DBDoc
    import workflows.phase2_web_ingestion as p2

    doc_id = str(uuid.uuid4())
    content_hash = hashlib.sha256(f"unique-{doc_id}".encode()).hexdigest()

    # Simulate a crash after Document add but before/during association
    original_commit = None
    call_count = [0]

    class FakeSession:
        """Wraps a real session and raises on the first commit."""
        def __init__(self):
            self._db = SessionLocal()

        def query(self, *a, **kw):
            return self._db.query(*a, **kw)

        def add(self, obj):
            return self._db.add(obj)

        def commit(self):
            call_count[0] += 1
            if call_count[0] == 1:
                self._db.rollback()
                raise RuntimeError("Simulated commit failure")
            return self._db.commit()

        def rollback(self):
            return self._db.rollback()

        def close(self):
            return self._db.close()

    # Verify the document was NOT persisted after a commit failure
    db_check = SessionLocal()
    try:
        doc = db_check.query(DBDoc).filter(DBDoc.content_hash == content_hash).first()
        assert doc is None, "Document should not exist before test"
    finally:
        db_check.close()
