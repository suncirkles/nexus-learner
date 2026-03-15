"""
E2E backend test: full INDEXING → GENERATION pipeline.

Uses documents/pyspark.pdf as the fixture (small, well-structured PDF).
Runs against live SQLite + Qdrant + LLM APIs — not a unit test.

Usage:
    PYTHONPATH=. python tests/e2e_backend_ingestion.py
    PYTHONPATH=. pytest tests/e2e_backend_ingestion.py -v -s
"""
import logging
import os
import uuid
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("e2e")

SAMPLE_PDF = os.path.abspath("documents/pyspark.pdf")
TARGET_TOPICS = ["RDD", "DataFrame"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _indexing_state(file_path: str, doc_id: str) -> dict:
    return {
        "mode": "INDEXING",
        "file_path": file_path,
        "doc_id": doc_id,
        "subject_id": None,
        "target_topics": [],
        "total_pages": 0,
        "current_page": 0,
        "chunks": [],
        "current_chunk_index": 0,
        "hierarchy": [],
        "pending_qdrant_docs": [],
        "current_new_cards": [],
        "generated_flashcards": [],
        "status_message": "Starting INDEXING…",
    }


def _generation_state(doc_id: str, subject_id: int, target_topics: list) -> dict:
    return {
        "mode": "GENERATION",
        "file_path": None,
        "doc_id": doc_id,
        "subject_id": subject_id,
        "target_topics": target_topics,
        "total_pages": 0,
        "current_page": 0,
        "chunks": [],
        "current_chunk_index": 0,
        "hierarchy": [],
        "pending_qdrant_docs": [],
        "current_new_cards": [],
        "generated_flashcards": [],
        "status_message": "Starting GENERATION…",
    }


# ── phase runners ─────────────────────────────────────────────────────────────

def run_indexing(graph, file_path: str, doc_id: str) -> dict:
    logger.info("═══ PHASE 1 — INDEXING ═══")
    state = _indexing_state(file_path, doc_id)
    final = {}
    for event in graph.stream(state):
        for node_name, update in event.items():
            if isinstance(update, dict):
                final.update(update)
                msg = update.get("status_message", "")
                if msg:
                    logger.info(f"  [{node_name}] {msg}")
    logger.info(f"  Hierarchy built: {len(final.get('hierarchy', []))} topic(s)")
    return final


def run_generation(graph, doc_id: str, subject_id: int, target_topics: list) -> dict:
    logger.info(f"═══ PHASE 2 — GENERATION (topics={target_topics}) ═══")
    state = _generation_state(doc_id, subject_id, target_topics)
    final = {}
    for event in graph.stream(state):
        for node_name, update in event.items():
            if isinstance(update, dict):
                final.update(update)
                msg = update.get("status_message", "")
                if msg:
                    logger.info(f"  [{node_name}] {msg}")
    cards = final.get("generated_flashcards", [])
    logger.info(f"  Cards produced this run: {len(cards)}")
    return final


# ── assertions ────────────────────────────────────────────────────────────────

def assert_indexing(db, actual_doc_id: str):
    from core.database import Document as DBDocument, Topic, Subtopic, ContentChunk

    doc = db.query(DBDocument).filter(DBDocument.id == actual_doc_id).first()
    assert doc is not None, f"Document record missing for id={actual_doc_id}"
    logger.info(f"  ✓ Document record exists: '{doc.title or doc.filename}'")

    topics = db.query(Topic).filter(Topic.document_id == actual_doc_id).all()
    assert len(topics) > 0, "No topics created during indexing"
    topic_names = [t.name for t in topics]
    logger.info(f"  ✓ {len(topics)} topic(s): {topic_names}")

    subtopics = (
        db.query(Subtopic)
        .join(Topic)
        .filter(Topic.document_id == actual_doc_id)
        .all()
    )
    assert len(subtopics) > 0, "No subtopics created during indexing"
    logger.info(f"  ✓ {len(subtopics)} subtopic(s) created")

    chunks = db.query(ContentChunk).filter(ContentChunk.document_id == actual_doc_id).all()
    assert len(chunks) > 0, "No content chunks created during indexing"
    chunks_with_subtopic = [c for c in chunks if c.subtopic_id is not None]
    logger.info(f"  ✓ {len(chunks)} chunk(s) indexed ({len(chunks_with_subtopic)} assigned to subtopics)")

    return topics


def assert_generation(db, subject_id: int, topics):
    from core.database import Flashcard

    cards = db.query(Flashcard).filter(Flashcard.subject_id == subject_id).all()
    assert len(cards) > 0, f"No flashcards generated for subject_id={subject_id}"

    scored = [c for c in cards if c.critic_score > 0]
    rejected = [c for c in cards if c.status == "rejected"]
    logger.info(f"  ✓ {len(cards)} flashcard(s) generated")
    logger.info(f"  ✓ {len(scored)} scored by critic, {len(rejected)} auto-rejected (score < 3)")

    # Sample output
    sample = cards[0]
    logger.info(f"  ✓ Sample Q: {sample.question[:80]!r}")
    logger.info(f"         A: {sample.answer[:80]!r}")
    logger.info(f"    Critic: {sample.critic_score}/5 — {(sample.critic_feedback or '')[:60]}")

    return cards


# ── main test ─────────────────────────────────────────────────────────────────

def run_e2e():
    if not os.path.exists(SAMPLE_PDF):
        logger.error(f"PDF fixture not found: {SAMPLE_PDF}")
        sys.exit(1)

    from core.database import SessionLocal, Subject, SubjectDocumentAssociation
    from workflows.phase1_ingestion import phase1_graph

    doc_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        subj = Subject(name=f"E2E-PySpark-{doc_id[:8]}")
        db.add(subj)
        db.commit()
        subject_id = subj.id
        logger.info(f"Created subject '{subj.name}' (id={subject_id})")
    finally:
        db.close()

    # ── Phase 1: INDEXING ──
    idx_final = run_indexing(phase1_graph, SAMPLE_PDF, doc_id)
    actual_doc_id = idx_final.get("doc_id", doc_id)

    db = SessionLocal()
    try:
        topics = assert_indexing(db, actual_doc_id)

        # Link document to subject (mirrors what app.py does on attachment)
        assoc = SubjectDocumentAssociation(subject_id=subject_id, document_id=actual_doc_id)
        db.add(assoc)
        db.commit()
    finally:
        db.close()

    # ── Phase 2: GENERATION ──
    run_generation(phase1_graph, actual_doc_id, subject_id, TARGET_TOPICS)

    db = SessionLocal()
    try:
        cards = assert_generation(db, subject_id, topics)
    finally:
        db.close()

    logger.info("\n" + "═" * 60)
    logger.info("  E2E TEST PASSED")
    logger.info("═" * 60)
    return True


# ── pytest entry point ────────────────────────────────────────────────────────

def test_e2e_indexing_and_generation():
    """Full pipeline e2e: indexing creates topics+chunks, generation creates flashcards."""
    import pytest
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip(f"PDF fixture not found: {SAMPLE_PDF}")
    assert run_e2e()


if __name__ == "__main__":
    success = run_e2e()
    sys.exit(0 if success else 1)
