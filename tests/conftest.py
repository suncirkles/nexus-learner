"""
tests/conftest.py
-----------------
Session-scoped fixtures for the integration test suite.

Which env file to use is controlled by --env-file (default: .env):

    # Use production Supabase (default)
    PYTHONPATH=. pytest tests/ -v

    # Use local test database
    PYTHONPATH=. pytest tests/ -v --env-file=.env.test

Start the test database before using .env.test:
    docker-compose up -d test-db
"""
import os
import uuid
import pytest
from dotenv import load_dotenv


def pytest_addoption(parser):
    parser.addoption(
        "--env-file",
        default=".env",
        help="Environment file to load (default: .env). Use .env.test for the local test DB.",
    )


def pytest_configure(config):
    """Load the selected env file before any test module is imported.

    pytest_configure runs before collection, so DB_URL is set before
    core.database creates its engine.
    """
    try:
        env_file = config.getoption("--env-file")
    except ValueError:
        env_file = ".env"   # fallback during early plugin init

    load_dotenv(env_file, override=True)

    # Pin to mini model to stay within TPM limits across the test suite.
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
        "question_type": "active_recall",
        "total_pages": MAX_PAGES,
        "current_page": 0,
        "chunks": [],
        "current_chunk_index": 0,
        "hierarchy": [],
        "pending_vector_docs": [],
        "current_new_cards": [],
        "generated_flashcards": [],
        "status_message": "start",
        "matched_subtopic_ids": None,
        "subtopic_embeddings": [],
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
    """
    from core.database import SessionLocal, Subject, Flashcard
    from workflows.phase1_ingestion import phase1_graph

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
        "question_type": "active_recall",
        "total_pages": 0,
        "current_page": 0,
        "chunks": [],
        "current_chunk_index": 0,
        "hierarchy": [],
        "pending_vector_docs": [],
        "current_new_cards": [],
        "generated_flashcards": [],
        "status_message": "start",
        "matched_subtopic_ids": None,
        "subtopic_embeddings": [],
    })

    db = SessionLocal()
    try:
        cards = db.query(Flashcard).filter(Flashcard.subject_id == subject_id).all()
        card_dicts = [
            {"id": c.id, "subject_id": c.subject_id, "question": c.question,
             "answer": c.answer, "status": c.status, "critic_score": c.critic_score}
            for c in cards
        ]
    finally:
        db.close()

    return {"subject_id": subject_id, "cards": card_dicts}
