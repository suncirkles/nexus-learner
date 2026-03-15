"""
tests/test_hitl_complexity.py
-------------------------------
Tests that the Mentor approve action correctly persists complexity_level
to the Flashcard row in the database.

These are pure DB-layer tests — no Streamlit, no LLM calls.
The "approve with tag" logic mirrors what render_flashcard_review_card()
does in app.py when the Mentor clicks Approve.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ---------------------------------------------------------------------------
# In-memory DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mem_db():
    from core.database import Base
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


# ---------------------------------------------------------------------------
# Helper: create a pending flashcard row
# ---------------------------------------------------------------------------

def _make_pending_card(Session, complexity_level=None, question_type="active_recall"):
    from core.database import Flashcard
    db = Session()
    try:
        fc = Flashcard(
            question="What is an integral?",
            answer="The antiderivative.",
            question_type=question_type,
            status="pending",
            complexity_level=complexity_level,
            subject_id=None,
            subtopic_id=None,
            chunk_id=None,
        )
        db.add(fc)
        db.commit()
        db.refresh(fc)
        return fc.id
    finally:
        db.close()


def _approve_with_complexity(Session, fc_id, complexity: str):
    """Mirrors the app.py approve + complexity-tag logic."""
    from core.database import Flashcard
    db = Session()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        if fc:
            fc.status = "approved"
            fc.complexity_level = None if complexity == "(unset)" else complexity
            db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("complexity", ["simple", "medium", "complex"])
def test_approve_saves_complexity_level(mem_db, complexity):
    """Approving with a complexity tag persists complexity_level."""
    from core.database import Flashcard

    fc_id = _make_pending_card(mem_db)
    _approve_with_complexity(mem_db, fc_id, complexity)

    db = mem_db()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        assert fc.status == "approved"
        assert fc.complexity_level == complexity
    finally:
        db.close()


def test_approve_with_unset_saves_none(mem_db):
    """Approving with '(unset)' leaves complexity_level as None."""
    from core.database import Flashcard

    fc_id = _make_pending_card(mem_db, complexity_level="medium")  # pre-filled by Critic
    _approve_with_complexity(mem_db, fc_id, "(unset)")

    db = mem_db()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        assert fc.status == "approved"
        assert fc.complexity_level is None
    finally:
        db.close()


def test_critic_suggestion_pre_populates(mem_db):
    """A complexity_level set by the Critic is readable before Mentor acts."""
    from core.database import Flashcard

    fc_id = _make_pending_card(mem_db, complexity_level="complex")

    db = mem_db()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        assert fc.complexity_level == "complex"   # Critic's suggestion is present
        assert fc.status == "pending"             # not yet approved by Mentor
    finally:
        db.close()


def test_reject_does_not_save_complexity(mem_db):
    """Rejecting a card leaves complexity_level unchanged (None by default)."""
    from core.database import Flashcard

    fc_id = _make_pending_card(mem_db)

    db = mem_db()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        fc.status = "rejected"
        db.commit()

        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        assert fc.status == "rejected"
        assert fc.complexity_level is None   # not touched during reject
    finally:
        db.close()


@pytest.mark.parametrize("qtype", ["active_recall", "numerical", "scenario"])
def test_complexity_persisted_regardless_of_question_type(mem_db, qtype):
    """complexity_level is saved correctly for any question_type."""
    from core.database import Flashcard

    fc_id = _make_pending_card(mem_db, question_type=qtype)
    _approve_with_complexity(mem_db, fc_id, "medium")

    db = mem_db()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        assert fc.complexity_level == "medium"
        assert fc.question_type == qtype
    finally:
        db.close()
