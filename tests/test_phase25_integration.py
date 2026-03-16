"""
tests/test_phase25_integration.py
-----------------------------------
Phase 2.5 end-to-end integration test.

Pipeline:  Index Ch.1 page of AEM calculus PDF
        → Generate a 'numerical' flashcard
        → Critic grades with 4-score rubric
        → Mentor tags complexity='medium'
        → Verify card appears in DB with all Phase 2.5 fields

Uses the real LangGraph workflow but overrides PRIMARY_MODEL → gpt-4o-mini.
Requires:
  - OpenAI API key in .env
  - Qdrant running on port 6333
  - D:/cse/calculus/Advanced Engineering Mathematics 10th Edition.pdf
"""

import json
import os
import uuid

import pytest

# Model override must precede any local imports
os.environ.setdefault("PRIMARY_MODEL", "gpt-4o-mini")

CALCULUS_PDF = os.path.abspath("D:/cse/calculus/Advanced Engineering Mathematics 10th Edition.pdf")
MAX_PAGES = 1   # index only the first content page (page index 1 to avoid image cover)


def _initial_state(mode: str, doc_id: str, subject_id=None, question_type="active_recall", **overrides):
    base = {
        "mode": mode,
        "file_path": CALCULUS_PDF,
        "doc_id": doc_id,
        "subject_id": subject_id,
        "target_topics": [],
        "question_type": question_type,
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
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Session-scoped: index the calculus PDF once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def calculus_indexed():
    """Index page 1 of the AEM calculus book. Returns doc_id and subject_id."""
    if not os.path.exists(CALCULUS_PDF):
        pytest.skip(f"Calculus PDF not found: {CALCULUS_PDF}")

    from core.database import SessionLocal, Subject, Topic, SubjectDocumentAssociation
    from workflows.phase1_ingestion import phase1_graph

    doc_id = str(uuid.uuid4())
    db = SessionLocal()
    subj = Subject(name=f"AEM-Calculus-{doc_id[:8]}")
    db.add(subj)
    db.commit()
    subject_id = subj.id
    db.close()

    # Index starting at page 1 (skip image-only cover page 0)
    state = _initial_state(
        "INDEXING", doc_id,
        current_page=1,   # start from page 1
        total_pages=2,    # index 1 page only
    )
    final = phase1_graph.invoke(state)
    actual_doc_id = final.get("doc_id", doc_id)

    # Attach subject → document
    db = SessionLocal()
    try:
        exists = db.query(SubjectDocumentAssociation).filter_by(
            subject_id=subject_id, document_id=actual_doc_id
        ).first()
        if not exists:
            db.add(SubjectDocumentAssociation(subject_id=subject_id, document_id=actual_doc_id))
            db.commit()
        topics = db.query(Topic).filter(Topic.document_id == actual_doc_id).all()
        topic_names = [t.name for t in topics]
    finally:
        db.close()

    return {
        "doc_id": actual_doc_id,
        "subject_id": subject_id,
        "topic_names": topic_names,
    }


# ---------------------------------------------------------------------------
# Session-scoped: generate a 'numerical' card using the calculus index
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def numerical_cards(calculus_indexed):
    """Generate numerical flashcards and return as plain dicts."""
    from core.database import SessionLocal, Subject, Flashcard, SubjectDocumentAssociation
    from workflows.phase1_ingestion import phase1_graph

    doc_id = calculus_indexed["doc_id"]
    db = SessionLocal()
    subj = Subject(name=f"AEM-Numerical-{uuid.uuid4().hex[:8]}")
    db.add(subj)
    db.commit()
    subject_id = subj.id
    # Share the same indexed document
    db.add(SubjectDocumentAssociation(subject_id=subject_id, document_id=doc_id))
    db.commit()
    db.close()

    state = _initial_state(
        "GENERATION", doc_id,
        subject_id=subject_id,
        question_type="numerical",
        file_path=None,
        total_pages=0,
        current_page=0,
        current_chunk_index=0,
    )
    phase1_graph.invoke(state)

    db = SessionLocal()
    try:
        cards = db.query(Flashcard).filter(Flashcard.subject_id == subject_id).all()
        result = [
            {
                "id": c.id,
                "question_type": c.question_type,
                "question": c.question,
                "answer": c.answer,
                "rubric": c.rubric,
                "complexity_level": c.complexity_level,
                "critic_score": c.critic_score,
                "critic_rubric_scores": c.critic_rubric_scores,
                "status": c.status,
                "subject_id": subject_id,
            }
            for c in cards
        ]
    finally:
        db.close()

    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_calculus_pdf_indexed_has_topics(calculus_indexed):
    """Indexing the calculus PDF produces at least one topic."""
    assert len(calculus_indexed["topic_names"]) >= 1, (
        "Expected at least 1 topic indexed from AEM Ch.1"
    )


def test_numerical_cards_generated(numerical_cards):
    """At least one numerical card was generated from the calculus source."""
    assert len(numerical_cards) >= 1, "No numerical cards generated from calculus PDF"


def test_numerical_question_type_in_db(numerical_cards):
    """Every card generated has question_type='numerical' in the DB."""
    for card in numerical_cards:
        assert card["question_type"] == "numerical", (
            f"Card {card['id']} has question_type={card['question_type']!r}, expected 'numerical'"
        )


def test_rubric_json_valid_and_has_3_criteria(numerical_cards):
    """Rubric JSON is parseable and contains exactly 3 criteria."""
    for card in numerical_cards:
        assert card["rubric"] is not None, f"Card {card['id']} missing rubric"
        rubric = json.loads(card["rubric"])
        assert isinstance(rubric, list), "Rubric should be a list"
        assert len(rubric) == 3, f"Expected 3 rubric items, got {len(rubric)}"
        for item in rubric:
            assert "criterion" in item, "Rubric item missing 'criterion'"
            assert "description" in item, "Rubric item missing 'description'"


def test_critic_rubric_scores_json_valid(numerical_cards):
    """critic_rubric_scores JSON is present and has the 4 expected keys."""
    for card in numerical_cards:
        assert card["critic_rubric_scores"] is not None, (
            f"Card {card['id']} missing critic_rubric_scores"
        )
        scores = json.loads(card["critic_rubric_scores"])
        assert set(scores.keys()) == {"accuracy", "logic", "grounding", "clarity"}
        for key, val in scores.items():
            assert 1 <= val <= 4, f"{key}={val} out of range 1-4"


def test_suggested_complexity_set_by_critic(numerical_cards):
    """complexity_level is set (to a valid value) after Critic evaluation."""
    valid = {"simple", "medium", "complex"}
    for card in numerical_cards:
        assert card["complexity_level"] in valid, (
            f"Card {card['id']} has invalid complexity_level={card['complexity_level']!r}"
        )


def test_mentor_approve_with_complexity_tag(numerical_cards):
    """Mentor approve + complexity override persists complexity_level='medium'."""
    from core.database import SessionLocal, Flashcard

    card = numerical_cards[0]
    fc_id = card["id"]

    with SessionLocal() as db:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        fc.status = "approved"
        fc.complexity_level = "medium"
        db.commit()

    with SessionLocal() as db:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        assert fc.status == "approved"
        assert fc.complexity_level == "medium"


def test_approved_card_visible_to_learner(numerical_cards):
    """Approved numerical card is retrievable as an approved card."""
    from core.database import SessionLocal, Flashcard

    card = numerical_cards[0]
    subject_id = card["subject_id"]

    with SessionLocal() as db:
        approved = db.query(Flashcard).filter(
            Flashcard.subject_id == subject_id,
            Flashcard.status == "approved",
        ).all()
        assert len(approved) >= 1, "No approved cards found for Learner view"
        fc = approved[0]
        assert fc.question_type == "numerical"
        assert fc.complexity_level is not None
