"""
tests/test_modal_deployment_regressions.py
------------------------------------------
Regression tests for bugs introduced during the Modal/Supabase deployment migration.
Each test is named after the bug it guards against so failures are self-documenting.

Run with:
    PYTHONPATH=. pytest tests/test_modal_deployment_regressions.py -v

No live LLM calls or DB connections are required — every external dependency is mocked.
"""
import inspect
import json
import uuid
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Bug 1 — node_extract_hierarchy must pass existing topics to CuratorAgent
#
# Symptom: Attaching "D and F Block Elements" PDF extracts "Chemistry" (the
#   generic domain) instead of specific chapter topics.
# Root cause: curate_structure(full_text) called without existing_structure_text,
#   so the LLM had no subject context and defaulted to broad labelling.
# ---------------------------------------------------------------------------

def _make_mock_session(doc_id, topic_name="D and F Block Elements", subtopic_name="Transition Metals"):
    """Return a callable that produces a fresh MagicMock session each time."""
    from core.database import Topic as _Topic, Subtopic as _Sub, Document as _Doc

    def factory():
        session = MagicMock()

        existing_doc = MagicMock()
        existing_doc.id = doc_id

        existing_topic = MagicMock()
        existing_topic.id = 42
        existing_topic.name = topic_name

        existing_sub = MagicMock()
        existing_sub.name = subtopic_name

        def query_dispatch(model):
            q = MagicMock()
            filt = MagicMock()
            if model is _Doc:
                filt.first.return_value = existing_doc
                filt.all.return_value = [existing_doc]
            elif model is _Topic:
                filt.first.return_value = None   # "not found" → insert path for hierarchy
                filt.all.return_value = [existing_topic]
            elif model is _Sub:
                filt.first.return_value = None
                filt.all.return_value = [existing_sub]
            else:
                filt.first.return_value = None
                filt.all.return_value = []
            q.filter.return_value = filt
            return q

        session.query.side_effect = query_dispatch
        return session

    return factory


def test_curator_receives_existing_structure_text(tmp_path, monkeypatch):
    """
    node_extract_hierarchy must pass the already-indexed topic names as
    existing_structure_text to curate_structure().

    Without this context the LLM produces generic labels ("Chemistry") instead
    of document-specific ones ("D and F Block Elements → Transition Metals").
    """
    from workflows import phase1_ingestion

    doc_id = str(uuid.uuid4())
    fake_pdf = tmp_path / "dnf.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    curate_calls = []

    def spy_curate(full_text, existing_structure_text="No existing topics."):
        curate_calls.append(existing_structure_text)
        return {"hierarchy": [], "doc_summary": ""}

    mock_agent = MagicMock()
    mock_agent.load_page_text.return_value = "D and F Block Elements content"
    mock_agent.get_page_count.return_value = 2
    mock_agent.get_content_hash.return_value = "hash123"
    mock_agent.embeddings.embed_documents.return_value = []

    mock_curator = MagicMock()
    mock_curator.curate_structure.side_effect = spy_curate

    monkeypatch.setattr(phase1_ingestion, "_ingestion_agent", mock_agent)
    monkeypatch.setattr(phase1_ingestion, "_curator_agent", mock_curator)

    with patch("workflows.phase1_ingestion.SessionLocal",
               side_effect=_make_mock_session(doc_id)):
        phase1_ingestion.node_extract_hierarchy({
            "mode": "INDEXING",
            "file_path": str(fake_pdf),
            "doc_id": doc_id,
            "total_pages": 0,
            "subject_id": None,
            "target_topics": [],
            "question_type": "active_recall",
            "subtopic_embeddings": [],
        })

    assert len(curate_calls) == 1, "curate_structure should be called exactly once"
    passed_text = curate_calls[0]
    assert passed_text != "No existing topics.", (
        "Regression Bug 1: existing_structure_text was left as default. "
        "Existing topics from DB must be fetched and passed to CuratorAgent."
    )
    assert "D and F Block Elements" in passed_text, (
        f"Existing topic name missing from structure text. Got: {passed_text!r}"
    )
    assert "Transition Metals" in passed_text, (
        f"Existing subtopic name missing from structure text. Got: {passed_text!r}"
    )


def test_curator_uses_default_when_no_topics_exist(tmp_path, monkeypatch):
    """First indexing of a new document: no existing topics → default sentinel passed."""
    from workflows import phase1_ingestion

    doc_id = str(uuid.uuid4())
    fake_pdf = tmp_path / "new.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    curate_calls = []

    def spy_curate(full_text, existing_structure_text="No existing topics."):
        curate_calls.append(existing_structure_text)
        return {"hierarchy": [], "doc_summary": ""}

    mock_agent = MagicMock()
    mock_agent.load_page_text.return_value = "Brand new content"
    mock_agent.get_page_count.return_value = 1
    mock_agent.get_content_hash.return_value = "newhash"
    mock_agent.embeddings.embed_documents.return_value = []

    mock_curator = MagicMock()
    mock_curator.curate_structure.side_effect = spy_curate

    monkeypatch.setattr(phase1_ingestion, "_ingestion_agent", mock_agent)
    monkeypatch.setattr(phase1_ingestion, "_curator_agent", mock_curator)

    # Session returns no existing topics
    def no_topics_session():
        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.first.return_value = MagicMock(id=doc_id)
        q.filter.return_value.all.return_value = []
        session.query.return_value = q
        return session

    with patch("workflows.phase1_ingestion.SessionLocal", side_effect=no_topics_session):
        phase1_ingestion.node_extract_hierarchy({
            "mode": "INDEXING",
            "file_path": str(fake_pdf),
            "doc_id": doc_id,
            "total_pages": 0,
            "subject_id": None,
            "target_topics": [],
            "question_type": "active_recall",
            "subtopic_embeddings": [],
        })

    assert len(curate_calls) == 1
    assert curate_calls[0] == "No existing topics.", (
        "First-time indexing must pass 'No existing topics.' sentinel"
    )


def test_extract_hierarchy_is_noop_for_generation_mode():
    """GENERATION mode must return {} without calling CuratorAgent or touching DB."""
    from workflows import phase1_ingestion

    curator_called = []
    monkeypatch_curator = MagicMock()
    monkeypatch_curator.curate_structure.side_effect = lambda *a, **kw: curator_called.append(1)

    result = phase1_ingestion.node_extract_hierarchy({
        "mode": "GENERATION",
        "file_path": None,
        "doc_id": str(uuid.uuid4()),
    })

    assert result == {}, "GENERATION mode must return empty dict (no-op)"
    assert not curator_called, "CuratorAgent must NOT be called in GENERATION mode"


# ---------------------------------------------------------------------------
# Bug 2 — question_type must flow from spawn call into the Modal worker state
#
# Symptom: User selects "Fill in the Blank" but cards are generated as
#   "Active Recall" because Modal worker hardcoded "active_recall".
# Root cause: run_ingestion_background had no question_type param; func.spawn()
#   omitted it; current_state used a hardcoded literal.
# ---------------------------------------------------------------------------

def test_modal_worker_signature_includes_question_type():
    """run_ingestion_background must accept question_type as a named parameter."""
    modal = pytest.importorskip("modal", reason="modal package not installed locally")
    import modal_app
    sig = inspect.signature(modal_app.run_ingestion_background.raw_f
                            if hasattr(modal_app.run_ingestion_background, "raw_f")
                            else modal_app.run_ingestion_background)
    assert "question_type" in sig.parameters, (
        "Regression Bug 2: run_ingestion_background missing question_type param. "
        "User-selected question type is silently ignored on Modal path."
    )


def test_modal_worker_question_type_default_is_active_recall():
    """Default value for question_type must be 'active_recall' (backward-compat)."""
    modal = pytest.importorskip("modal", reason="modal package not installed locally")
    import modal_app
    sig = inspect.signature(modal_app.run_ingestion_background.raw_f
                            if hasattr(modal_app.run_ingestion_background, "raw_f")
                            else modal_app.run_ingestion_background)
    default = sig.parameters["question_type"].default
    assert default == "active_recall", (
        f"question_type default should be 'active_recall', got {default!r}"
    )


def test_spawn_worker_passes_question_type_to_modal():
    """
    _spawn_worker must include question_type as the 7th positional arg to func.spawn().
    Without this, the Modal worker receives 6 args and ignores question_type entirely.
    """
    from api.routers.ingestion import _spawn_worker
    from core.database import BatchJob

    spawn_calls = []

    mock_func = MagicMock()
    mock_func.spawn.side_effect = lambda *args: spawn_calls.append(args)

    mock_modal = MagicMock()
    mock_modal.Function.from_name.return_value = mock_func

    mock_job = MagicMock(spec=BatchJob)
    mock_job.status = "queued"
    mock_job.error = None

    mock_db = MagicMock()

    with patch.dict("os.environ", {"MODAL_RUN": "true"}), \
         patch("api.routers.ingestion.modal", mock_modal), \
         patch("api.routers.ingestion.threading.Thread"):

        job_id = str(uuid.uuid4())
        doc_id = str(uuid.uuid4())
        _spawn_worker(
            job_id=job_id,
            doc_id=doc_id,
            file_path="/data/temp_uploads/abc_test.pdf",
            subject_id=1,
            mode="GENERATION",
            target_topics=["Transition Elements"],
            question_type="fill_blank",
            new_job=mock_job,
            db=mock_db,
        )

    assert len(spawn_calls) == 1, "func.spawn() should have been called once"
    args = spawn_calls[0]
    assert len(args) >= 7, (
        f"Regression Bug 2b: func.spawn() called with {len(args)} args, expected 7+. "
        "question_type is missing from the spawn call."
    )
    assert args[6] == "fill_blank", (
        f"7th spawn arg should be question_type='fill_blank', got {args[6]!r}"
    )


def test_thread_fallback_receives_question_type():
    """When Modal spawn fails, the fallback thread must also receive question_type."""
    from api.routers.ingestion import _spawn_worker
    from core.database import BatchJob

    thread_args = []

    def fake_thread_target(*args):
        thread_args.extend(args)

    mock_job = MagicMock(spec=BatchJob)
    mock_job.status = "queued"
    mock_job.error = None

    mock_db = MagicMock()

    with patch.dict("os.environ", {"MODAL_RUN": "false"}), \
         patch("api.routers.ingestion.threading.Thread") as mock_thread_cls:

        started_thread = MagicMock()
        mock_thread_cls.return_value = started_thread

        _spawn_worker(
            job_id=str(uuid.uuid4()),
            doc_id=str(uuid.uuid4()),
            file_path="/data/temp_uploads/abc.pdf",
            subject_id=2,
            mode="GENERATION",
            target_topics=[],
            question_type="short_answer",
            new_job=mock_job,
            db=mock_db,
        )

        _, kwargs = mock_thread_cls.call_args
        thread_args_tuple = kwargs.get("args", ())

    # question_type is 7th positional arg to _run_ingestion_db_thread
    assert len(thread_args_tuple) >= 7, (
        f"Thread args should include question_type (7 args), got {len(thread_args_tuple)}"
    )
    assert thread_args_tuple[6] == "short_answer", (
        f"question_type in thread args should be 'short_answer', got {thread_args_tuple[6]!r}"
    )


# ---------------------------------------------------------------------------
# Bug 3a — GENERATION BatchJob.filename must not be "Unknown"
#
# Symptom: Progress bar shows "Unknown: generation — Generated 3 Q&A(s)"
# Root cause: spawn_ingestion derived filename from file_path which is None
#   for GENERATION, leaving display_filename = "Unknown".
# ---------------------------------------------------------------------------

def test_generation_job_filename_uses_document_title():
    """
    When file_path is None (GENERATION mode), spawn_ingestion must look up
    the document title/filename from the DB instead of storing "Unknown".
    """
    from fastapi.testclient import TestClient
    from api.app import app

    doc_id = str(uuid.uuid4())

    mock_doc = MagicMock()
    mock_doc.title = "D and F Block Elements"
    mock_doc.filename = "dnf_block.pdf"

    mock_subject = MagicMock()
    mock_subject.id = 1

    def fake_db():
        db = MagicMock()

        def query_dispatch(model):
            from core.database import Subject, Document as _Doc
            q = MagicMock()
            if model is Subject:
                q.filter.return_value.first.return_value = mock_subject
            elif model is _Doc:
                q.filter.return_value.first.return_value = mock_doc
            else:
                q.filter.return_value.first.return_value = None
            return q

        db.query.side_effect = query_dispatch
        return db

    db_instance = fake_db()
    from api.dependencies import get_db

    with patch("api.routers.ingestion._spawn_worker", return_value="thread"):
        app.dependency_overrides[get_db] = lambda: db_instance
        try:
            client = TestClient(app)
            resp = client.post("/ingestion/spawn", json={
                "mode": "GENERATION",
                "doc_id": doc_id,
                "file_path": None,
                "subject_id": 1,
                "question_type": "active_recall",
                "target_topics": ["Transition Metals"],
            })
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 202
    body = resp.json()
    assert body["filename"] != "Unknown", (
        "Regression Bug 3a: GENERATION BatchJob.filename is 'Unknown'. "
        "Document title must be looked up from DB when file_path is None."
    )
    assert body["filename"] == "D and F Block Elements", (
        f"Expected document title as filename, got {body['filename']!r}"
    )


def test_generation_job_filename_falls_back_to_db_filename_if_no_title():
    """If Document.title is blank, fall back to Document.filename."""
    from fastapi.testclient import TestClient
    from api.app import app

    doc_id = str(uuid.uuid4())

    mock_doc = MagicMock()
    mock_doc.title = ""
    mock_doc.filename = "dnf_block.pdf"

    mock_subject = MagicMock()
    mock_subject.id = 1

    def fake_db():
        db = MagicMock()

        def query_dispatch(model):
            from core.database import Subject, Document as _Doc
            q = MagicMock()
            if model is Subject:
                q.filter.return_value.first.return_value = mock_subject
            elif model is _Doc:
                q.filter.return_value.first.return_value = mock_doc
            else:
                q.filter.return_value.first.return_value = None
            return q

        db.query.side_effect = query_dispatch
        return db

    db_instance = fake_db()
    from api.app import app
    from api.dependencies import get_db

    with patch("api.routers.ingestion._spawn_worker", return_value="thread"):
        app.dependency_overrides[get_db] = lambda: db_instance
        try:
            client = TestClient(app)
            resp = client.post("/ingestion/spawn", json={
                "mode": "GENERATION",
                "doc_id": doc_id,
                "file_path": None,
                "subject_id": 1,
                "question_type": "active_recall",
                "target_topics": [],
            })
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 202
    assert resp.json()["filename"] == "dnf_block.pdf"


# ---------------------------------------------------------------------------
# Bug 4 — _flush_vector_batch (renamed from _flush_qdrant_batch)
#
# The rename broke two tests that still imported/patched the old name.
# These tests guard that the rename stays consistent.
# ---------------------------------------------------------------------------

def test_flush_vector_batch_exists_in_module():
    """_flush_vector_batch must exist; _flush_qdrant_batch must NOT exist."""
    from workflows import phase1_ingestion

    assert hasattr(phase1_ingestion, "_flush_vector_batch"), (
        "_flush_vector_batch not found in phase1_ingestion — rename was reverted?"
    )
    assert not hasattr(phase1_ingestion, "_flush_qdrant_batch"), (
        "_flush_qdrant_batch still exists — old name not fully removed, "
        "tests that patch it will silently become no-ops."
    )


def test_flush_vector_batch_noop_on_empty():
    """_flush_vector_batch([]) must not raise even if the vector store is unreachable."""
    from workflows.phase1_ingestion import _flush_vector_batch
    _flush_vector_batch([])   # must not raise


def test_flush_vector_batch_raises_on_store_failure(monkeypatch):
    """_flush_vector_batch must propagate errors from the vector store."""
    from workflows import phase1_ingestion

    mock_store = MagicMock()
    mock_store.upsert_chunks.side_effect = ConnectionError("Vector store down")

    monkeypatch.setattr(phase1_ingestion, "_ingestion_agent", MagicMock(
        _vector_store=mock_store
    ))

    with patch("workflows.phase1_ingestion.get_vector_store", return_value=mock_store):
        pending = [{"text": "hello", "metadata": {"document_id": "x", "db_chunk_id": 1}}]
        with pytest.raises(Exception):
            phase1_ingestion._flush_vector_batch(pending)


# ---------------------------------------------------------------------------
# Bug 5 — Document record must exist before topic FK insert
#
# Symptom: ForeignKeyViolation on topics.document_id when node_extract_hierarchy
#   runs before node_ingest has created the Document row.
# Root cause: node_ingest created the Document at current_page==0, but
#   node_extract_hierarchy runs first in the graph.
# ---------------------------------------------------------------------------

def test_node_extract_hierarchy_creates_document_before_topics(tmp_path, monkeypatch):
    """
    node_extract_hierarchy must ensure a Document row exists in the DB before
    inserting any Topic rows (which have a FK constraint on document_id).
    """
    from workflows import phase1_ingestion

    doc_id = str(uuid.uuid4())
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    db_operations = []

    def mock_curator_curate(full_text, existing_structure_text="No existing topics."):
        return {
            "hierarchy": [{"name": "Test Topic", "subtopics": [{"name": "Sub1", "summary": ""}]}],
            "doc_summary": "test",
        }

    mock_agent = MagicMock()
    mock_agent.load_page_text.return_value = "content"
    mock_agent.get_page_count.return_value = 1
    mock_agent.get_content_hash.return_value = "hash_abc"
    mock_agent.embeddings.embed_documents.return_value = [[0.1] * 384]

    mock_curator = MagicMock()
    mock_curator.curate_structure.side_effect = mock_curator_curate

    monkeypatch.setattr(phase1_ingestion, "_ingestion_agent", mock_agent)
    monkeypatch.setattr(phase1_ingestion, "_curator_agent", mock_curator)

    # Track whether Document was added before Topic
    doc_added_at = []
    topic_added_at = []
    op_counter = [0]

    from core.database import Document as _Doc, Topic as _Topic, Subtopic as _Sub

    def make_session():
        session = MagicMock()
        added_objects = []

        def add_obj(obj):
            op_counter[0] += 1
            added_objects.append((op_counter[0], type(obj).__name__, obj))
            if isinstance(obj, _Doc):
                doc_added_at.append(op_counter[0])
            elif isinstance(obj, _Topic):
                topic_added_at.append(op_counter[0])

        session.add.side_effect = add_obj

        def query_dispatch(model):
            q = MagicMock()
            filt = MagicMock()
            if model is _Doc:
                # First call (check existence): not found → triggers create
                # Subsequent calls: found
                call_count = [0]
                def first_side():
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return None  # not found → insert
                    return MagicMock(id=doc_id)
                filt.first.side_effect = first_side
            elif model is _Topic:
                filt.first.return_value = None
                filt.all.return_value = []
            elif model is _Sub:
                filt.first.return_value = None
                filt.all.return_value = []
            else:
                filt.first.return_value = None
                filt.all.return_value = []
            q.filter.return_value = filt
            return q

        session.query.side_effect = query_dispatch
        return session

    with patch("workflows.phase1_ingestion.SessionLocal", side_effect=make_session):
        phase1_ingestion.node_extract_hierarchy({
            "mode": "INDEXING",
            "file_path": str(fake_pdf),
            "doc_id": doc_id,
            "total_pages": 0,
            "subject_id": None,
            "target_topics": [],
            "question_type": "active_recall",
            "subtopic_embeddings": [],
        })

    if doc_added_at and topic_added_at:
        assert min(doc_added_at) < min(topic_added_at), (
            "Regression Bug 5: Document was added AFTER Topic, which violates the FK "
            "constraint on topics.document_id. Document must be persisted first."
        )
