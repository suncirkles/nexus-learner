"""
tests/test_duplicates.py
--------------------------
Tests duplicate document detection via IngestionAgent.create_document_record().

Current behaviour (Phase 2b):
  - First call: creates and returns a new document dict.
  - Second call with identical file content: detects matching content_hash via
    DocumentRepo.get_by_content_hash(), returns the EXISTING document dict (same id).
  - Only one Document row should exist in the DB for duplicate content.

Patching notes:
  - OpenAIEmbeddings is a module-level import → patch via agents.ingestion.OpenAIEmbeddings
  - get_llm is a local import inside create_document_record → patch via core.models.get_llm.
    Raising an exception causes the title-generation block to fall back to filename
    (the block is inside try/except Exception), which is the intended behavior.
"""

import os
import uuid
import pytest
from unittest.mock import patch, MagicMock

from agents.ingestion import IngestionAgent
from core.database import SessionLocal, Document as DBDocument, Base, engine


@pytest.fixture(scope="function", autouse=True)
def setup_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    # Drop and recreate (not just drop) so subsequent test modules still have tables
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


@pytest.fixture
def mock_embeddings():
    """Suppress OpenAI embedding construction — not needed for duplicate detection."""
    with patch("agents.ingestion.OpenAIEmbeddings") as MockEmb:
        instance = MagicMock()
        MockEmb.return_value = instance
        yield instance


def _make_pdf(path: str, text: str):
    import fitz
    d = fitz.open()
    p = d.new_page()
    p.insert_text((50, 50), text)
    d.save(path)
    d.close()


def test_duplicate_detection(tmp_path, mock_embeddings):
    """Second create_document_record call with same file returns the original doc."""
    test_file = str(tmp_path / "dup_test.pdf")
    _make_pdf(test_file, "This is unique content for duplicate detection testing.")

    doc_id1 = str(uuid.uuid4())
    doc_id2 = str(uuid.uuid4())

    agent = IngestionAgent()

    # Raise inside the title-generation try/except so it falls back to filename
    with patch("core.models.get_llm", side_effect=RuntimeError("skip title gen")):
        result1 = agent.create_document_record(test_file, doc_id1)
        assert result1["id"] == doc_id1

        # Same content → returns the existing doc dict (no new row created)
        result2 = agent.create_document_record(test_file, doc_id2)

    assert result2["id"] == doc_id1, (
        "Duplicate content should return the original document id, not a new one"
    )

    db = SessionLocal()
    try:
        count = db.query(DBDocument).count()
        assert count == 1, f"Expected 1 Document in DB for duplicate content, found {count}"
    finally:
        db.close()


def test_different_content_creates_separate_documents(tmp_path, mock_embeddings):
    """Two files with different content create two separate Document rows."""
    file1 = str(tmp_path / "doc_a.pdf")
    file2 = str(tmp_path / "doc_b.pdf")
    _make_pdf(file1, "Document A — unique content alpha beta gamma delta.")
    _make_pdf(file2, "Document B — completely different content epsilon zeta.")

    doc_id_a = str(uuid.uuid4())
    doc_id_b = str(uuid.uuid4())

    agent = IngestionAgent()

    with patch("core.models.get_llm", side_effect=RuntimeError("skip title gen")):
        result_a = agent.create_document_record(file1, doc_id_a)
        result_b = agent.create_document_record(file2, doc_id_b)

    assert result_a["id"] != result_b["id"], (
        "Different content should produce different document records"
    )

    db = SessionLocal()
    try:
        count = db.query(DBDocument).count()
        assert count == 2
    finally:
        db.close()
