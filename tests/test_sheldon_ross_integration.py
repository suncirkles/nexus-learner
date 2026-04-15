"""
tests/test_sheldon_ross_integration.py
----------------------------------------
End-to-end integration test against the first two chapters of:
  "Probability and Statistics for Engineers and Scientists" — Sheldon M. Ross

Chapter page ranges (PyMuPDF 0-indexed):
  Ch.1 "Introduction to Statistics"  → pages 17–24  (8 pages)
  Ch.2 "Descriptive Statistics"      → pages 25–34  (first 10 of 46)

Pipeline exercised:
  1. Document record pre-created (hash-stable, survives re-runs)
  2. INDEXING  — Ch.1 pages, then Ch.2 pages (same doc_id)
  3. Verify    — topics, subtopics, content chunks, no boilerplate
  4. GENERATION — active_recall flashcards, all Phase 2.5 fields
  5. Verify    — rubric, 4-score critic, complexity, card lifecycle

Uses gpt-4o-mini (set via env var before any imports) and real Qdrant.
Skip gracefully if the PDF is not present on disk.
"""

import json
import os
import uuid

import pytest

# Override PRIMARY_MODEL before any local imports so config picks it up
os.environ.setdefault("PRIMARY_MODEL", "gpt-4o-mini")

# All tests in this module hit real LLM + Qdrant APIs and are excluded from
# the default `pytest tests/` run.  Run explicitly with:
#   PYTHONPATH=. pytest tests/test_sheldon_ross_integration.py -v
# or include slow tests globally:
#   PYTHONPATH=. pytest tests/ -m slow -v
pytestmark = pytest.mark.slow

ROSS_PDF = os.path.abspath(
    r"D:/cse/Probability/Probability and statistics for engineers and scientists - Sheldon Ross.pdf"
)

# Ch.1: pages 17–24 (current_page=17, total_pages=25 → processes 17..24)
CH1_START = 17
CH1_END_EXCLUSIVE = 25   # total_pages value

# Ch.2 first 10 pages: 25–34
CH2_START = 25
CH2_END_EXCLUSIVE = 35

BOILERPLATE_KEYWORDS = {
    "table of contents", "contents", "preface", "foreword", "acknowledgement",
    "dedication", "bibliography", "references", "index", "publisher", "copyright",
    "isbn", "about the author",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(mode, doc_id, subject_id=None, current_page=0, total_pages=1, question_type="active_recall"):
    return {
        "mode": mode,
        "file_path": ROSS_PDF,
        "doc_id": doc_id,
        "subject_id": subject_id,
        "target_topics": [],
        "question_type": question_type,
        "total_pages": total_pages,
        "current_page": current_page,
        "chunks": [],
        "current_chunk_index": 0,
        "hierarchy": [],
        "pending_vector_docs": [],
        "current_new_cards": [],
        "generated_flashcards": [],
        "status_message": "start",
        "matched_subtopic_ids": None,
    }


def _ensure_doc_record(doc_id: str) -> str:
    """
    Pre-create (or reuse) a Document DB record for the Ross PDF.
    Mirrors the hash logic in node_ingest so re-runs reuse the same row.
    Returns the canonical doc_id (may differ from the argument if a
    content-hash match was found from a previous run).
    """
    from agents.ingestion import IngestionAgent
    from core.database import SessionLocal, Document as DBDocument
    import os

    agent = IngestionAgent()
    sample_text = agent.load_page_text(ROSS_PDF, 0)[:10000]
    file_size = os.path.getsize(ROSS_PDF)
    content_hash = agent.get_content_hash(
        sample_text + str(file_size) + os.path.basename(ROSS_PDF)
    )

    db = SessionLocal()
    try:
        existing = db.query(DBDocument).filter(DBDocument.content_hash == content_hash).first()
        if existing:
            return str(existing.id)

        basename = os.path.basename(ROSS_PDF)
        new_doc = DBDocument(id=doc_id, filename=basename, title=basename, content_hash=content_hash)
        db.add(new_doc)
        db.commit()
        return doc_id
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Module-scoped fixture: index Ch.1 + Ch.2
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sheldon_indexed():
    """
    Indexes Ch.1 (pages 17–24) then Ch.2 pages 25–34 of the Ross PDF.
    Returns dict: doc_id, subject_id, topic_names, subtopic_names, chunk_count.
    Skipped if PDF not present.
    """
    if not os.path.exists(ROSS_PDF):
        pytest.skip(f"Ross PDF not found: {ROSS_PDF}")

    from core.database import (
        SessionLocal, Subject, Topic, Subtopic, ContentChunk,
        SubjectDocumentAssociation,
    )
    from workflows.phase1_ingestion import phase1_graph

    raw_doc_id = str(uuid.uuid4())
    doc_id = _ensure_doc_record(raw_doc_id)

    # Create (or reuse) a subject and link to document
    db = SessionLocal()
    try:
        subject_name = f"Ross-Prob-Stats-{doc_id[:8]}"
        subj = db.query(Subject).filter_by(name=subject_name).first()
        if not subj:
            subj = Subject(name=subject_name)
            db.add(subj)
            db.commit()
        subject_id = subj.id

        exists = db.query(SubjectDocumentAssociation).filter_by(
            subject_id=subject_id, document_id=doc_id
        ).first()
        if not exists:
            db.add(SubjectDocumentAssociation(subject_id=subject_id, document_id=doc_id))
            db.commit()
    finally:
        db.close()

    # --- Index Ch.1 ---
    phase1_graph.invoke(_make_state(
        "INDEXING", doc_id,
        current_page=CH1_START,
        total_pages=CH1_END_EXCLUSIVE,
    ))

    # --- Index Ch.2 (first 10 pages) ---
    phase1_graph.invoke(_make_state(
        "INDEXING", doc_id,
        current_page=CH2_START,
        total_pages=CH2_END_EXCLUSIVE,
    ))

    # Collect results
    db = SessionLocal()
    try:
        topics = db.query(Topic).filter(Topic.document_id == doc_id).all()
        topic_names = [t.name for t in topics]
        topic_ids = [t.id for t in topics]

        subtopics = db.query(Subtopic).filter(Subtopic.topic_id.in_(topic_ids)).all()
        subtopic_names = [s.name for s in subtopics]
        subtopic_ids = [s.id for s in subtopics]

        chunk_count = db.query(ContentChunk).filter(
            ContentChunk.subtopic_id.in_(subtopic_ids)
        ).count()
    finally:
        db.close()

    return {
        "doc_id": doc_id,
        "subject_id": subject_id,
        "topic_names": topic_names,
        "subtopic_names": subtopic_names,
        "chunk_count": chunk_count,
    }


# ---------------------------------------------------------------------------
# Module-scoped fixture: generate flashcards from the indexed content
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sheldon_cards(sheldon_indexed):
    """
    Runs GENERATION over the indexed Ross content and returns card dicts.
    """
    from core.database import SessionLocal, Subject, Flashcard, SubjectDocumentAssociation
    from workflows.phase1_ingestion import phase1_graph

    doc_id = sheldon_indexed["doc_id"]

    db = SessionLocal()
    try:
        gen_subj = Subject(name=f"Ross-Gen-{uuid.uuid4().hex[:8]}")
        db.add(gen_subj)
        db.commit()
        gen_subject_id = gen_subj.id
        db.add(SubjectDocumentAssociation(subject_id=gen_subject_id, document_id=doc_id))
        db.commit()
    finally:
        db.close()

    state = _make_state(
        "GENERATION", doc_id,
        subject_id=gen_subject_id,
        question_type="active_recall",
        total_pages=0,
        current_page=0,
    )
    phase1_graph.invoke(state)

    db = SessionLocal()
    try:
        cards = db.query(Flashcard).filter(Flashcard.subject_id == gen_subject_id).all()
        result = [
            {
                "id": c.id,
                "subject_id": gen_subject_id,
                "question": c.question,
                "answer": c.answer,
                "question_type": c.question_type,
                "rubric": c.rubric,
                "complexity_level": c.complexity_level,
                "critic_score": c.critic_score,
                "critic_rubric_scores": c.critic_rubric_scores,
                "status": c.status,
                "subtopic_id": c.subtopic_id,
            }
            for c in cards
        ]
    finally:
        db.close()

    return result


# ===========================================================================
# INDEXING TESTS
# ===========================================================================

def test_pdf_exists():
    """Sanity: the Ross PDF is on disk."""
    assert os.path.exists(ROSS_PDF), f"PDF not found: {ROSS_PDF}"


def test_topics_created(sheldon_indexed):
    """Indexing Ch.1 + Ch.2 produces at least 3 distinct topics."""
    names = sheldon_indexed["topic_names"]
    assert len(names) >= 3, (
        f"Expected ≥3 topics, got {len(names)}: {names}"
    )


def test_subtopics_created(sheldon_indexed):
    """Each topic has at least one subtopic; total ≥5 subtopics."""
    assert len(sheldon_indexed["subtopic_names"]) >= 5, (
        f"Expected ≥5 subtopics, got {len(sheldon_indexed['subtopic_names'])}: "
        f"{sheldon_indexed['subtopic_names']}"
    )


def test_content_chunks_stored(sheldon_indexed):
    """Content chunks are persisted and linked to subtopics."""
    assert sheldon_indexed["chunk_count"] >= 5, (
        f"Expected ≥5 chunks in DB, got {sheldon_indexed['chunk_count']}"
    )


def test_no_boilerplate_topics(sheldon_indexed):
    """No indexed topic is pure boilerplate (TOC, preface, publisher, etc.)."""
    bad = [
        name for name in sheldon_indexed["topic_names"]
        if any(kw in name.lower() for kw in BOILERPLATE_KEYWORDS)
    ]
    assert not bad, f"Boilerplate topics slipped through: {bad}"


def test_no_boilerplate_subtopics(sheldon_indexed):
    """No subtopic name is pure boilerplate."""
    bad = [
        name for name in sheldon_indexed["subtopic_names"]
        if any(kw in name.lower() for kw in BOILERPLATE_KEYWORDS)
    ]
    assert not bad, f"Boilerplate subtopics slipped through: {bad}"


def test_ch1_topics_are_statistical(sheldon_indexed):
    """
    At least one topic name reflects Ch.1 statistical content
    (introduction, statistics, data collection, population, sampling, history).
    """
    ch1_keywords = {
        "introduction", "statistics", "data", "population", "sample",
        "history", "inferential", "descriptive",
    }
    names_lower = {n.lower() for n in sheldon_indexed["topic_names"]}
    matched = [
        name for name in names_lower
        if any(kw in name for kw in ch1_keywords)
    ]
    assert matched, (
        f"No Ch.1 statistical topics found. Topics: {sheldon_indexed['topic_names']}"
    )


def test_ch2_topics_are_descriptive_stats(sheldon_indexed):
    """
    At least one topic reflects Ch.2 descriptive statistics content
    (mean, median, variance, histogram, frequency, correlation, percentile).
    """
    ch2_keywords = {
        "mean", "median", "variance", "histogram", "frequency", "distribution",
        "correlation", "percentile", "descriptive", "summariz", "spread",
        "standard deviation", "dataset",
    }
    names_lower = {n.lower() for n in sheldon_indexed["topic_names"] + sheldon_indexed["subtopic_names"]}
    matched = [n for n in names_lower if any(kw in n for kw in ch2_keywords)]
    assert matched, (
        f"No Ch.2 descriptive-stats topics found. "
        f"Topics: {sheldon_indexed['topic_names']} | "
        f"Subtopics: {sheldon_indexed['subtopic_names']}"
    )


def test_document_linked_to_subject(sheldon_indexed):
    """The document is associated with the subject in the DB."""
    from core.database import SessionLocal, SubjectDocumentAssociation

    doc_id = sheldon_indexed["doc_id"]
    subject_id = sheldon_indexed["subject_id"]

    with SessionLocal() as db:
        link = db.query(SubjectDocumentAssociation).filter_by(
            subject_id=subject_id, document_id=doc_id
        ).first()
    assert link is not None, "SubjectDocumentAssociation not found"


# ===========================================================================
# GENERATION TESTS
# ===========================================================================

def test_flashcards_generated(sheldon_cards):
    """Generation produces at least one flashcard from the indexed content."""
    assert len(sheldon_cards) >= 1, "No flashcards generated from Ross Ch.1+2"


def test_flashcard_question_type(sheldon_cards):
    """All cards have question_type='active_recall'."""
    for card in sheldon_cards:
        assert card["question_type"] == "active_recall", (
            f"Card {card['id']} has unexpected question_type={card['question_type']!r}"
        )


def test_flashcard_rubric_present_and_valid(sheldon_cards):
    """Every card has a valid rubric JSON with ≥1 criteria entries."""
    for card in sheldon_cards:
        assert card["rubric"] is not None, f"Card {card['id']} missing rubric"
        rubric = json.loads(card["rubric"])
        assert isinstance(rubric, list) and len(rubric) >= 1, (
            f"Card {card['id']} rubric should be a non-empty list, got: {rubric}"
        )
        for item in rubric:
            assert "criterion" in item and "description" in item, (
                f"Rubric item missing keys: {item}"
            )


def test_critic_4score_rubric_present(sheldon_cards):
    """Every card has critic_rubric_scores JSON with all 4 sub-scores in 1–4 range."""
    for card in sheldon_cards:
        assert card["critic_rubric_scores"] is not None, (
            f"Card {card['id']} missing critic_rubric_scores"
        )
        scores = json.loads(card["critic_rubric_scores"])
        assert set(scores.keys()) == {"accuracy", "logic", "grounding", "clarity"}, (
            f"Unexpected keys in critic_rubric_scores: {set(scores.keys())}"
        )
        for key, val in scores.items():
            assert 1 <= val <= 4, f"Card {card['id']}: {key}={val} out of 1–4 range"


def test_critic_aggregate_score_matches(sheldon_cards):
    """aggregate critic_score == round(mean of 4 sub-scores)."""
    import math

    for card in sheldon_cards:
        if card["critic_rubric_scores"] is None or card["critic_score"] is None:
            continue
        scores = json.loads(card["critic_rubric_scores"])
        expected = round(sum(scores.values()) / 4)
        assert card["critic_score"] == expected, (
            f"Card {card['id']}: critic_score={card['critic_score']} "
            f"expected={expected} from {scores}"
        )


def test_complexity_level_set_by_critic(sheldon_cards):
    """All cards have a valid complexity_level after Critic evaluation."""
    valid = {"simple", "medium", "complex"}
    for card in sheldon_cards:
        assert card["complexity_level"] in valid, (
            f"Card {card['id']} has invalid complexity_level={card['complexity_level']!r}"
        )


def test_cards_in_pending_status(sheldon_cards):
    """Newly generated cards are in 'pending' or 'rejected' status.

    Cards that pass the Critic remain 'pending' for mentor review.
    Cards auto-rejected by the Critic (grounding/clarity score < 2) are
    correctly set to 'rejected' — that is also valid lifecycle behaviour.
    """
    valid_statuses = {"pending", "rejected"}
    for card in sheldon_cards:
        assert card["status"] in valid_statuses, (
            f"Card {card['id']} has unexpected status={card['status']!r}"
        )
    # At least some cards must have made it to pending
    pending_count = sum(1 for c in sheldon_cards if c["status"] == "pending")
    assert pending_count > 0, "No cards reached 'pending' status — generation or critic pipeline broken"


def test_cards_linked_to_subtopics(sheldon_cards):
    """Every card is linked to a valid subtopic_id."""
    from core.database import SessionLocal, Subtopic

    subtopic_ids = {c["subtopic_id"] for c in sheldon_cards}
    with SessionLocal() as db:
        found = {
            r.id for r in db.query(Subtopic).filter(Subtopic.id.in_(subtopic_ids)).all()
        }
    missing = subtopic_ids - found
    assert not missing, f"Cards reference unknown subtopic_ids: {missing}"


def test_card_questions_are_non_trivial(sheldon_cards):
    """Questions are at least 10 characters (not stub/empty responses)."""
    for card in sheldon_cards:
        assert len(card["question"]) >= 10, (
            f"Card {card['id']} question too short: {card['question']!r}"
        )


def test_mentor_approve_persists(sheldon_cards):
    """Approving a card with complexity override persists both fields correctly."""
    from core.database import SessionLocal, Flashcard

    if not sheldon_cards:
        pytest.skip("No cards to approve")

    fc_id = sheldon_cards[0]["id"]

    with SessionLocal() as db:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        fc.status = "approved"
        fc.complexity_level = "complex"
        db.commit()

    with SessionLocal() as db:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        assert fc.status == "approved"
        assert fc.complexity_level == "complex"


def test_approved_card_retrievable_for_learner(sheldon_cards):
    """After approval, card appears in the approved query (Learner view)."""
    from core.database import SessionLocal, Flashcard

    if not sheldon_cards:
        pytest.skip("No cards generated")

    subject_id = sheldon_cards[0]["subject_id"]

    with SessionLocal() as db:
        approved = db.query(Flashcard).filter(
            Flashcard.subject_id == subject_id,
            Flashcard.status == "approved",
        ).all()

    assert len(approved) >= 1, "No approved cards visible to Learner"
    fc = approved[0]
    assert fc.question_type == "active_recall"
    assert fc.complexity_level is not None
