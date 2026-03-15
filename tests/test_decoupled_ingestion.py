"""
Integration test: verifies the two-phase (INDEXING → GENERATION) pipeline.

Uses session-scoped fixtures from conftest.py so indexing and generation each
run only once per test session. Skipped when the PDF fixture is absent.
"""
import os
import pytest  # noqa: F401 — needed for @pytest.mark.slow

SAMPLE_PDF_PATH = os.path.abspath(r"documents/pyspark.pdf")


@pytest.mark.slow
def test_indexing_creates_topics_and_chunks(indexed_doc):
    """Indexing should produce at least one topic and one content chunk."""
    from core.database import SessionLocal, Topic, ContentChunk

    doc_id = indexed_doc["doc_id"]
    db = SessionLocal()
    try:
        topics = db.query(Topic).filter(Topic.document_id == doc_id).all()
        chunk_count = db.query(ContentChunk).filter(ContentChunk.document_id == doc_id).count()
    finally:
        db.close()

    assert len(topics) > 0, "Indexing should produce at least one topic"
    assert chunk_count > 0, "Indexing should produce at least one content chunk"


@pytest.mark.slow
def test_generation_produces_flashcards_for_target_topic(generated_cards):
    """Generation should produce flashcards.
    Uses the session-scoped generated_cards fixture — no extra LLM calls."""
    cards = generated_cards["cards"]
    assert len(cards) > 0, "No flashcards generated during the generation pass"
