"""
Integration test: verifies that topic-filtered generation produces flashcards.

Uses the session-scoped `indexed_doc` fixture from conftest.py so indexing
runs only once. Skipped when the PDF fixture is absent.
"""
import os
import uuid
import pytest  # noqa: F401 — needed for @pytest.mark.slow

TARGET_TAG = "PySpark"  # broad tag guaranteed to match page-1 content


def _build_generation_state(doc_id: str, subject_id: int, target_topics: list) -> dict:
    return {
        "mode": "GENERATION",
        "doc_id": doc_id,
        "subject_id": subject_id,
        "target_topics": target_topics,
        "chunks": [],
        "current_chunk_index": 0,
        "hierarchy": [],
        "pending_vector_docs": [],
        "current_new_cards": [],
        "generated_flashcards": [],
        "file_path": None,
        "total_pages": 0,
        "current_page": 0,
        "status_message": "Starting generation...",
        "matched_subtopic_ids": None,
    }


@pytest.mark.slow
def test_filtered_generation_produces_relevant_cards(indexed_doc):
    """Filtered generation must produce at least one flashcard for the target topic."""
    from core.database import SessionLocal, Subject, Flashcard
    from workflows.phase1_ingestion import phase1_graph

    doc_id = indexed_doc["doc_id"]
    topic_names = indexed_doc["topic_names"]
    assert topic_names, "No topics indexed — cannot test filtered generation"

    # Use the first actual indexed topic name for reliable matching
    target = topic_names[0]

    db = SessionLocal()
    subj = Subject(name=f"Filtered-{uuid.uuid4().hex[:8]}")
    db.add(subj)
    db.commit()
    subject_id = subj.id
    db.close()

    filt_state = _build_generation_state(doc_id, subject_id, [target])
    phase1_graph.invoke(filt_state)

    db = SessionLocal()
    try:
        total_cards = db.query(Flashcard).filter(Flashcard.subject_id == subject_id).count()
        assert total_cards > 0, (
            f"Filtered generation for topic '{target}' produced no flashcards"
        )
    finally:
        db.close()
