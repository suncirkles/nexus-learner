"""
tests/conftest.py
-----------------
Session-scoped fixtures for the integration test suite.

Overrides PRIMARY_MODEL → gpt-4o-mini at session start so that the full
suite stays well within the 30k TPM limit for gpt-4o. gpt-4o-mini has
a 10M TPM limit and is sufficient for correctness tests.
"""
import os
import uuid
import pytest

# ── Model override ────────────────────────────────────────────────────────────
# Must happen before any agent or workflow module is imported so that
# core/config.py picks up the override via pydantic-settings env parsing.
os.environ.setdefault("PRIMARY_MODEL", "gpt-4o-mini")

SAMPLE_PDF = os.path.abspath("documents/pyspark.pdf")
MAX_PAGES = 1  # 1 page ≈ 4-5 chunks — fast enough for CI


@pytest.fixture(scope="session")
def indexed_doc():
    """
    Indexes 1 page of pyspark.pdf once per test session.
    Returns dict with keys: doc_id, subject_id, topic_names.
    Skipped if the PDF fixture is absent.
    """
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip(f"PDF fixture not found: {SAMPLE_PDF}")

    from core.database import (
        SessionLocal, Subject, Topic, SubjectDocumentAssociation,
    )
    from workflows.phase1_ingestion import phase1_graph

    doc_id = str(uuid.uuid4())
    db = SessionLocal()
    subj = Subject(name=f"SharedIndex-{doc_id[:8]}")
    db.add(subj)
    db.commit()
    subject_id = subj.id
    db.close()

    final = phase1_graph.invoke({
        "mode": "INDEXING",
        "file_path": SAMPLE_PDF,
        "doc_id": doc_id,
        "subject_id": None,
        "target_topics": [],
        "total_pages": MAX_PAGES,
        "current_page": 0,
        "chunks": [],
        "current_chunk_index": 0,
        "hierarchy": [],
        "pending_qdrant_docs": [],
        "current_new_cards": [],
        "generated_flashcards": [],
        "status_message": "start",
        "matched_subtopic_ids": None,
    })

    actual_doc_id = final.get("doc_id", doc_id)

    db = SessionLocal()
    try:
        topics = db.query(Topic).filter(Topic.document_id == actual_doc_id).all()
        topic_names = [t.name for t in topics]
        exists = db.query(SubjectDocumentAssociation).filter_by(
            subject_id=subject_id, document_id=actual_doc_id
        ).first()
        if not exists:
            db.add(SubjectDocumentAssociation(subject_id=subject_id, document_id=actual_doc_id))
            db.commit()
    finally:
        db.close()

    return {"doc_id": actual_doc_id, "subject_id": subject_id, "topic_names": topic_names}


@pytest.fixture(scope="session")
def generated_cards(indexed_doc):
    """
    Runs one unfiltered GENERATION pass on the indexed_doc once per session.
    Returns dict with keys: subject_id, cards (list of Flashcard ORM rows read as dicts).
    Tests that only need to verify cards exist/have correct fields should use this
    fixture instead of running their own generation — avoids hitting rate limits.
    """
    from core.database import SessionLocal, Subject, Flashcard
    from workflows.phase1_ingestion import phase1_graph
    import uuid

    doc_id = indexed_doc["doc_id"]

    db = SessionLocal()
    subj = Subject(name=f"SharedGen-{uuid.uuid4().hex[:8]}")
    db.add(subj)
    db.commit()
    subject_id = subj.id
    db.close()

    phase1_graph.invoke({
        "mode": "GENERATION",
        "file_path": None,
        "doc_id": doc_id,
        "subject_id": subject_id,
        "target_topics": [],
        "total_pages": 0,
        "current_page": 0,
        "chunks": [],
        "current_chunk_index": 0,
        "hierarchy": [],
        "pending_qdrant_docs": [],
        "current_new_cards": [],
        "generated_flashcards": [],
        "status_message": "start",
        "matched_subtopic_ids": None,
    })

    db = SessionLocal()
    try:
        cards = db.query(Flashcard).filter(Flashcard.subject_id == subject_id).all()
        # Snapshot fields as plain dicts so session can be closed safely
        card_dicts = [
            {"id": c.id, "subject_id": c.subject_id, "question": c.question,
             "answer": c.answer, "status": c.status, "critic_score": c.critic_score}
            for c in cards
        ]
    finally:
        db.close()

    return {"subject_id": subject_id, "cards": card_dicts}
