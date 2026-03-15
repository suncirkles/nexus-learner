"""
tests/test_critic_rubric.py
-----------------------------
Unit tests for the enhanced CriticAgent with 4-score rubric evaluation.

All LLM calls are mocked — no API calls required.
Tests:
  1. RubricEvaluation scores are integers in 1-4 range
  2. auto-reject fires when grounding_score < 2 (Flashcard.status → "rejected")
  3. aggregate critic_score = round(mean([acc, log, grd, cla]))
  4. high scores do NOT trigger auto-reject
  5. suggested_complexity is written to Flashcard.complexity_level
  6. critic_rubric_scores JSON is persisted with all 4 keys
  7. AUTO_ACCEPT_CONTENT=True skips auto-reject even for grounding < 2
"""

import json
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("PRIMARY_MODEL", "gpt-4o-mini")


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
# Helpers
# ---------------------------------------------------------------------------

def _insert_flashcard(Session, **kwargs):
    """Insert a minimal Flashcard row, return its id."""
    from core.database import Flashcard
    db = Session()
    try:
        fc = Flashcard(
            question=kwargs.get("question", "What is a derivative?"),
            answer=kwargs.get("answer", "Rate of change."),
            question_type=kwargs.get("question_type", "active_recall"),
            status=kwargs.get("status", "pending"),
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


def _make_eval_result(acc=4, log=4, grd=4, cla=4, complexity="complex"):
    from agents.critic import RubricEvaluation
    return RubricEvaluation(
        accuracy_score=acc,
        logic_score=log,
        grounding_score=grd,
        clarity_score=cla,
        feedback="Test feedback.",
        suggested_complexity=complexity,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scores_are_1_to_4():
    """Pydantic model rejects out-of-range values."""
    from agents.critic import RubricEvaluation
    import pydantic

    with pytest.raises((pydantic.ValidationError, ValueError)):
        RubricEvaluation(
            accuracy_score=5,  # invalid: max is 4
            logic_score=4,
            grounding_score=4,
            clarity_score=4,
            feedback="bad",
            suggested_complexity="medium",
        )

    with pytest.raises((pydantic.ValidationError, ValueError)):
        RubricEvaluation(
            accuracy_score=4,
            logic_score=0,  # invalid: min is 1
            grounding_score=4,
            clarity_score=4,
            feedback="bad",
            suggested_complexity="medium",
        )


def test_aggregate_score_is_rounded_mean():
    """Verify _aggregate helper computes round(mean) correctly."""
    from agents.critic import _aggregate
    assert _aggregate(4, 4, 4, 4) == 4
    assert _aggregate(1, 1, 1, 1) == 1
    assert _aggregate(4, 3, 2, 1) == round((4+3+2+1)/4)  # round(2.5) = 2
    assert _aggregate(3, 3, 3, 2) == round((3+3+3+2)/4)  # round(2.75) = 3


def test_auto_reject_fires_when_grounding_lt_2(mem_db):
    """grounding_score < 2 → status = "rejected" in DB."""
    from agents.critic import CriticAgent
    from core.database import Flashcard

    fc_id = _insert_flashcard(mem_db)

    agent = CriticAgent.__new__(CriticAgent)
    agent.llm = MagicMock()
    agent.chain = MagicMock()
    agent.chain.invoke.return_value = _make_eval_result(acc=4, log=3, grd=1, cla=3, complexity="simple")

    with patch("agents.critic.SessionLocal", mem_db), \
         patch("agents.critic.settings") as mock_settings:
        mock_settings.AUTO_ACCEPT_CONTENT = False
        result = agent.evaluate_flashcard(fc_id, "source", "question?", "answer.")

    assert "error" not in result
    assert result["rubric_scores"]["grounding"] == 1

    db = mem_db()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        assert fc.status == "rejected"
    finally:
        db.close()


def test_no_auto_reject_when_grounding_gte_2(mem_db):
    """grounding_score >= 2 → card is NOT auto-rejected."""
    from agents.critic import CriticAgent
    from core.database import Flashcard

    fc_id = _insert_flashcard(mem_db)

    agent = CriticAgent.__new__(CriticAgent)
    agent.llm = MagicMock()
    agent.chain = MagicMock()
    agent.chain.invoke.return_value = _make_eval_result(acc=4, log=3, grd=2, cla=3, complexity="medium")

    with patch("agents.critic.SessionLocal", mem_db), \
         patch("agents.critic.settings") as mock_settings:
        mock_settings.AUTO_ACCEPT_CONTENT = False
        result = agent.evaluate_flashcard(fc_id, "source", "question?", "answer.")

    db = mem_db()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        assert fc.status != "rejected"
    finally:
        db.close()


def test_auto_accept_bypasses_auto_reject(mem_db):
    """When AUTO_ACCEPT_CONTENT=True, low grounding card is kept approved."""
    from agents.critic import CriticAgent
    from core.database import Flashcard

    fc_id = _insert_flashcard(mem_db, status="approved")

    agent = CriticAgent.__new__(CriticAgent)
    agent.llm = MagicMock()
    agent.chain = MagicMock()
    agent.chain.invoke.return_value = _make_eval_result(acc=2, log=2, grd=1, cla=2, complexity="simple")

    with patch("agents.critic.SessionLocal", mem_db), \
         patch("agents.critic.settings") as mock_settings:
        mock_settings.AUTO_ACCEPT_CONTENT = True
        agent.evaluate_flashcard(fc_id, "source", "question?", "answer.")

    db = mem_db()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        assert fc.status == "approved"
    finally:
        db.close()


def test_complexity_written_to_db(mem_db):
    """suggested_complexity from Critic is stored in Flashcard.complexity_level."""
    from agents.critic import CriticAgent
    from core.database import Flashcard

    fc_id = _insert_flashcard(mem_db)

    agent = CriticAgent.__new__(CriticAgent)
    agent.llm = MagicMock()
    agent.chain = MagicMock()
    agent.chain.invoke.return_value = _make_eval_result(acc=4, log=4, grd=4, cla=4, complexity="complex")

    with patch("agents.critic.SessionLocal", mem_db), \
         patch("agents.critic.settings") as mock_settings:
        mock_settings.AUTO_ACCEPT_CONTENT = False
        result = agent.evaluate_flashcard(fc_id, "source", "question?", "answer.")

    assert result["suggested_complexity"] == "complex"
    db = mem_db()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        assert fc.complexity_level == "complex"
    finally:
        db.close()


def test_critic_rubric_scores_json_persisted(mem_db):
    """critic_rubric_scores JSON contains all 4 keys with correct values."""
    from agents.critic import CriticAgent
    from core.database import Flashcard

    fc_id = _insert_flashcard(mem_db)

    agent = CriticAgent.__new__(CriticAgent)
    agent.llm = MagicMock()
    agent.chain = MagicMock()
    agent.chain.invoke.return_value = _make_eval_result(acc=3, log=2, grd=4, cla=3, complexity="medium")

    with patch("agents.critic.SessionLocal", mem_db), \
         patch("agents.critic.settings") as mock_settings:
        mock_settings.AUTO_ACCEPT_CONTENT = False
        result = agent.evaluate_flashcard(fc_id, "source", "question?", "answer.")

    db = mem_db()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        scores = json.loads(fc.critic_rubric_scores)
        assert set(scores.keys()) == {"accuracy", "logic", "grounding", "clarity"}
        assert scores["accuracy"] == 3
        assert scores["logic"] == 2
        assert scores["grounding"] == 4
        assert scores["clarity"] == 3
        # aggregate = round((3+2+4+3)/4) = round(3.0) = 3
        assert fc.critic_score == 3
    finally:
        db.close()


def test_unknown_complexity_normalised_to_medium(mem_db):
    """Critic normalises unknown complexity labels to 'medium'."""
    from agents.critic import CriticAgent
    from core.database import Flashcard

    fc_id = _insert_flashcard(mem_db)

    agent = CriticAgent.__new__(CriticAgent)
    agent.llm = MagicMock()
    agent.chain = MagicMock()
    agent.chain.invoke.return_value = _make_eval_result(complexity="hard")  # invalid label

    with patch("agents.critic.SessionLocal", mem_db), \
         patch("agents.critic.settings") as mock_settings:
        mock_settings.AUTO_ACCEPT_CONTENT = False
        agent.evaluate_flashcard(fc_id, "source", "question?", "answer.")

    db = mem_db()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        assert fc.complexity_level == "medium"
    finally:
        db.close()
