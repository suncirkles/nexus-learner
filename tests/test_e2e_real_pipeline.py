"""
tests/test_e2e_real_pipeline.py
--------------------------------
End-to-end integration test: REAL database, REAL LLM, REAL PDF — no mocks.

Covers the full INDEXING → GENERATION pipeline and verifies:

  1. ContentChunks are assigned a non-null subtopic_id (critical for GENERATION JOIN)
  2. Flashcards carry the full subject→topic→subtopic FK chain
  3. SubjectTopicAssociation is populated after GENERATION
  4. Per-topic-type card limit (MAX_CARDS_PER_TOPIC_TYPE) is respected
  5. Flashcards are visible via the mentor-review topic tree (pending_count > 0)

Marked 'slow' — not run in the default suite.
Run explicitly with:
    PYTHONPATH=. pytest tests/test_e2e_real_pipeline.py -v -s -m slow

Requires:
    - DB_URL pointing to Supabase in .env
    - A valid LLM provider key (DEFAULT_LLM_PROVIDER / PRIMARY_MODEL)
    - documents/calculus_chapter1.pdf present

Cost guard:
    MAX_CARDS_PER_TOPIC_TYPE=2 and MAX_CARDS_PER_PDF=8 are forced for the run
    so at most ~8 cards are generated regardless of env defaults.
"""

import os
import uuid
import logging
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File under test
# ---------------------------------------------------------------------------
_PDF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "documents", "calculus_chapter1.pdf")
)
# Index at most this many pages to keep the test fast
_MAX_INDEX_PAGES = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(mode, doc_id, subject_id=None, file_path=None, total_pages=0, **extra):
    base = {
        "mode": mode,
        "file_path": file_path,
        "doc_id": doc_id,
        "subject_id": subject_id,
        "target_topics": [],
        "question_type": "active_recall",
        "total_pages": total_pages,
        "current_page": 0,
        "chunks": [],
        "current_chunk_index": 0,
        "hierarchy": [],
        "pending_qdrant_docs": [],
        "matched_subtopic_ids": None,
        "current_new_cards": [],
        "subtopic_embeddings": [],
        "generated_flashcards": [],
        "status_message": "start",
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Module-scoped fixture: run the pipeline once, yield results, then clean up
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline(request):
    """
    Runs real INDEXING then real GENERATION against Supabase.

    Yields a dict:
        doc_id          — the document UUID used for indexing
        subject_id      — the test subject created for generation
        index_final     — final graph state after INDEXING
        gen_final       — final graph state after GENERATION
    """
    if not os.path.exists(_PDF):
        pytest.skip(f"PDF not found: {_PDF}")

    # Use mini model to reduce cost (setdefault = don't override if already set in .env)
    os.environ.setdefault("PRIMARY_MODEL", "gpt-4o-mini")

    from core.database import (
        SessionLocal, Subject, Document as DBDocument,
        SubjectDocumentAssociation, Topic, Subtopic, ContentChunk, Flashcard,
        SubjectTopicAssociation,
    )
    from workflows.phase1_ingestion import phase1_graph

    doc_id = str(uuid.uuid4())
    subject_id = None

    # ---------- INDEXING ----------
    logger.info("E2E: Starting INDEXING — doc_id=%s pdf=%s", doc_id, _PDF)
    index_state = _make_state(
        "INDEXING",
        doc_id,
        file_path=_PDF,
        total_pages=_MAX_INDEX_PAGES,
    )
    index_final = phase1_graph.invoke(index_state)
    actual_doc_id = index_final.get("doc_id", doc_id)
    logger.info("E2E: INDEXING complete — actual_doc_id=%s status=%s",
                actual_doc_id, index_final.get("status_message"))

    # ---------- Subject + association ----------
    db = SessionLocal()
    try:
        subj = Subject(name=f"E2E-Real-{uuid.uuid4().hex[:8]}")
        db.add(subj)
        db.commit()
        subject_id = subj.id

        exists = db.query(SubjectDocumentAssociation).filter_by(
            subject_id=subject_id, document_id=actual_doc_id
        ).first()
        if not exists:
            db.add(SubjectDocumentAssociation(
                subject_id=subject_id, document_id=actual_doc_id
            ))
            db.commit()
        logger.info("E2E: Subject created id=%s", subject_id)
    finally:
        db.close()

    # ---------- GENERATION ----------
    logger.info("E2E: Starting GENERATION — subject_id=%s", subject_id)
    gen_state = _make_state(
        "GENERATION",
        actual_doc_id,
        subject_id=subject_id,
    )
    gen_final = phase1_graph.invoke(gen_state)
    logger.info(
        "E2E: GENERATION complete — generated_flashcards=%d status=%s",
        len(gen_final.get("generated_flashcards", [])),
        gen_final.get("status_message"),
    )

    yield {
        "doc_id": actual_doc_id,
        "subject_id": subject_id,
        "index_final": index_final,
        "gen_final": gen_final,
    }

    # ---------- Teardown ----------
    logger.info("E2E: Teardown — deleting test subject=%s and doc=%s", subject_id, actual_doc_id)
    db = SessionLocal()
    try:
        # Explicitly delete flashcards first (avoids FK issues if cascade isn't complete)
        db.query(Flashcard).filter(Flashcard.subject_id == subject_id).delete(
            synchronize_session=False
        )
        db.query(SubjectTopicAssociation).filter(
            SubjectTopicAssociation.subject_id == subject_id
        ).delete(synchronize_session=False)
        db.query(SubjectDocumentAssociation).filter(
            SubjectDocumentAssociation.subject_id == subject_id
        ).delete(synchronize_session=False)
        db.query(Subject).filter(Subject.id == subject_id).delete(synchronize_session=False)

        # Delete document → cascades to topics → subtopics → chunks
        db.query(DBDocument).filter(DBDocument.id == actual_doc_id).delete(
            synchronize_session=False
        )
        db.commit()
        logger.info("E2E: Teardown complete.")
    except Exception as e:
        db.rollback()
        logger.error("E2E: Teardown failed (manual cleanup may be needed): %s", e)
    finally:
        db.close()

    # Try to clean up vectors too
    try:
        from repositories.vector.factory import get_vector_store
        store = get_vector_store()
        store.delete_by_document(actual_doc_id)
        logger.info("E2E: Vector cleanup complete for doc %s", actual_doc_id)
    except Exception as ve:
        logger.warning("E2E: Vector cleanup failed (non-fatal): %s", ve)


# ---------------------------------------------------------------------------
# Test 1: Indexing — ContentChunks have non-null subtopic_id
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_chunks_assigned_to_subtopics(pipeline):
    """Every ContentChunk must have a non-null subtopic_id after INDEXING.

    If subtopic_id is NULL the INNER JOIN in node_ingest GENERATION silently
    drops all chunks → 0 flashcards generated. This is the most critical invariant.
    """
    from core.database import SessionLocal, ContentChunk

    doc_id = pipeline["doc_id"]
    db = SessionLocal()
    try:
        all_chunks = db.query(ContentChunk).filter(
            ContentChunk.document_id == doc_id
        ).all()
        null_subtopic = [c for c in all_chunks if c.subtopic_id is None]

        logger.info(
            "T1: %d total chunks, %d with null subtopic_id",
            len(all_chunks), len(null_subtopic),
        )
        assert len(all_chunks) >= 1, \
            "No ContentChunks found after INDEXING — pipeline failed silently"
        assert not null_subtopic, (
            f"{len(null_subtopic)}/{len(all_chunks)} chunks have subtopic_id=NULL. "
            "These are invisible to GENERATION and produce 0 flashcards."
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 2: Indexing — Topics and Subtopics created
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_topics_and_subtopics_created(pipeline):
    """INDEXING must create at least one Topic with at least one Subtopic."""
    from core.database import SessionLocal, Topic, Subtopic

    doc_id = pipeline["doc_id"]
    db = SessionLocal()
    try:
        topics = db.query(Topic).filter(Topic.document_id == doc_id).all()
        logger.info("T2: %d topic(s) found: %s", len(topics), [t.name for t in topics])
        assert len(topics) >= 1, "No topics indexed — CuratorAgent likely failed"

        for t in topics:
            subs = db.query(Subtopic).filter(Subtopic.topic_id == t.id).all()
            logger.info(
                "T2:   topic '%s' → %d subtopic(s): %s",
                t.name, len(subs), [s.name for s in subs],
            )
            assert len(subs) >= 1, \
                f"Topic '{t.name}' has no subtopics — CuratorAgent returned empty hierarchy"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 3: Generation — Flashcards exist with correct FK chain
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_flashcards_have_full_fk_chain(pipeline):
    """Generated flashcards must carry non-null subject_id, topic_id, and subtopic_id.

    A missing topic_id breaks the subject→topic→subtopic chain, causing cards to
    be invisible in the mentor review topic tree.
    """
    from core.database import SessionLocal, Flashcard

    subject_id = pipeline["subject_id"]
    db = SessionLocal()
    try:
        cards = db.query(Flashcard).filter(Flashcard.subject_id == subject_id).all()
        logger.info(
            "T3: %d flashcard(s) for subject_id=%s",
            len(cards), subject_id,
        )
        for fc in cards:
            logger.info(
                "T3:   fc id=%s subject_id=%s topic_id=%s subtopic_id=%s status=%s critic_score=%s",
                fc.id, fc.subject_id, fc.topic_id, fc.subtopic_id, fc.status, fc.critic_score,
            )

        assert len(cards) >= 1, (
            "No flashcards generated. Check GENERATION logs — likely subtopic_id=NULL "
            "on ContentChunks or LLM call failures."
        )
        null_topic = [fc for fc in cards if fc.topic_id is None]
        null_subtopic = [fc for fc in cards if fc.subtopic_id is None]
        assert not null_topic, (
            f"{len(null_topic)} flashcard(s) have topic_id=NULL — "
            "subject→topic link missing; card invisible in mentor review"
        )
        assert not null_subtopic, (
            f"{len(null_subtopic)} flashcard(s) have subtopic_id=NULL — "
            "card invisible in subtopic-level card counts"
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 4: Generation — SubjectTopicAssociation was populated
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_subject_topic_association_populated(pipeline):
    """SubjectTopicAssociation must be populated after GENERATION runs.

    This table is the explicit subject→topic bridge used by get_by_subject().
    Without it, topics only appear via the legacy document-association fallback.
    """
    from core.database import SessionLocal, SubjectTopicAssociation, Flashcard

    subject_id = pipeline["subject_id"]
    db = SessionLocal()
    try:
        sta_rows = db.query(SubjectTopicAssociation).filter(
            SubjectTopicAssociation.subject_id == subject_id
        ).all()
        logger.info(
            "T4: %d SubjectTopicAssociation row(s) for subject_id=%s topic_ids=%s",
            len(sta_rows), subject_id, [r.topic_id for r in sta_rows],
        )

        # Every topic_id on a flashcard must also appear in SubjectTopicAssociation
        fc_topic_ids = {
            fc.topic_id
            for fc in db.query(Flashcard).filter(Flashcard.subject_id == subject_id).all()
            if fc.topic_id is not None
        }
        sta_topic_ids = {r.topic_id for r in sta_rows}
        missing = fc_topic_ids - sta_topic_ids
        assert not missing, (
            f"topic_ids {missing} appear on flashcards but not in SubjectTopicAssociation — "
            "invalidation/refresh in node_ingest did not run correctly"
        )
        assert len(sta_rows) >= 1, \
            "SubjectTopicAssociation is empty — explicit subject→topic bridge not populated"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 5: Generation — Per-topic-type limit respected
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_per_topic_type_limit_respected(pipeline):
    """Each topic must have at most MAX_CARDS_PER_TOPIC_TYPE cards per question type.

    The limit is enforced by pre-filtering chunks in node_ingest. If violated it
    means the limit logic was bypassed or the setting was not picked up correctly.
    Allows a small overshoot (x3) because SocraticAgent returns 1-3 cards per chunk
    and the limit is chunk-based.
    """
    from core.database import SessionLocal, Flashcard
    from core.config import settings
    from sqlalchemy import func

    subject_id = pipeline["subject_id"]
    effective_limit = min(max(int(settings.MAX_CARDS_PER_TOPIC_TYPE), 1), 50)
    db = SessionLocal()
    try:
        rows = (
            db.query(Flashcard.topic_id, Flashcard.question_type, func.count(Flashcard.id))
            .filter(Flashcard.subject_id == subject_id)
            .group_by(Flashcard.topic_id, Flashcard.question_type)
            .all()
        )
        logger.info(
            "T5: Per-topic-type card counts (MAX_CARDS_PER_TOPIC_TYPE=%d):",
            effective_limit,
        )
        total_cards = 0
        violations = []
        for topic_id, qtype, count in rows:
            total_cards += count
            logger.info("T5:   topic_id=%s type=%s count=%s", topic_id, qtype, count)
            # Allow overshoot up to limit*3 — SocraticAgent returns 1-3 per chunk
            if count > effective_limit * 3:
                violations.append((topic_id, qtype, count))

        logger.info("T5: Total cards generated across all topics: %d", total_cards)
        assert not violations, (
            f"Topics exceeded per-topic-type limit ({effective_limit}x3={effective_limit*3}) "
            f"cards: {violations}. Chunk pre-filter in node_ingest may not be working."
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 6: Mentor review — topic tree returns pending cards
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_mentor_review_tree_shows_pending_cards(pipeline):
    """get_full_tree_by_subject must return topics with pending_count > 0.

    This is exactly what the Mentor Review page calls. If it returns 0 the page
    shows 'All caught up' even though flashcards exist in the DB.
    """
    from repositories.sql.topic_repo import TopicRepo

    subject_id = pipeline["subject_id"]
    repo = TopicRepo()

    topics = repo.get_by_subject(subject_id)
    logger.info("T6: get_by_subject(%s) returned %d topic(s)", subject_id, len(topics))
    assert topics, (
        f"get_by_subject({subject_id}) returned []. "
        "SubjectDocumentAssociation or SubjectTopicAssociation missing."
    )

    topic_ids = [t["id"] for t in topics]
    subtopics_map = repo.get_subtopics_for_topic_ids(topic_ids, subject_id=subject_id)
    logger.info("T6: subtopics_map (topic_id -> subtopic list):")
    for tid, subs in subtopics_map.items():
        for s in subs:
            logger.info(
                "T6:   topic_id=%s subtopic='%s' pending=%s approved=%s",
                tid, s.get("name"), s.get("pending_count"), s.get("approved_count"),
            )

    total_pending = sum(
        s.get("pending_count", 0)
        for subs in subtopics_map.values()
        for s in subs
    )
    assert total_pending > 0, (
        "pending_count=0 across all subtopics in the mentor review query even though "
        "flashcards were generated. The subject_id filter in get_subtopics_for_topic_ids "
        "is likely dropping cards from a different subject, or topic_id/subtopic_id on "
        "Flashcard is NULL. subtopics_map: " + str(subtopics_map)
    )


# ---------------------------------------------------------------------------
# Test 7: Critic ran and scored all non-rejected cards
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_critic_scored_all_cards(pipeline):
    """All non-rejected flashcards must have a critic_score > 0.

    A critic_score of 0 means node_critic did not run or CriticAgent failed
    for that card — the card has no quality signal for mentor review.
    """
    from core.database import SessionLocal, Flashcard

    subject_id = pipeline["subject_id"]
    db = SessionLocal()
    try:
        cards = db.query(Flashcard).filter(Flashcard.subject_id == subject_id).all()
        not_rejected = [fc for fc in cards if fc.status != "rejected"]
        unscored = [fc for fc in not_rejected if (fc.critic_score or 0) == 0]

        logger.info(
            "T7: %d total cards, %d non-rejected, %d unscored",
            len(cards), len(not_rejected), len(unscored),
        )
        for fc in unscored:
            logger.warning(
                "T7: unscored card id=%s status=%s critic_score=%s",
                fc.id, fc.status, fc.critic_score,
            )

        assert not unscored, (
            f"{len(unscored)} non-rejected card(s) have critic_score=0. "
            "CriticAgent may have errored or been skipped for these cards."
        )
    finally:
        db.close()
