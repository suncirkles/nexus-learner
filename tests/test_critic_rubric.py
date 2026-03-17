"""
tests/test_critic_rubric.py
-----------------------------
Unit tests for CriticAgent with 4-score rubric evaluation (Phase 2b API).

Phase 2b: evaluate_flashcard() returns CriticResult dataclass — no DB writes.
DB persistence is the workflow node's responsibility (via FlashcardRepo).

Tests:
  1. RubricEvaluation scores are integers in 1-4 range (pydantic validation)
  2. _aggregate helper computes round(mean) correctly
  3. auto-reject fires when grounding_score < 2 → CriticResult.should_reject=True
  4. grounding_score >= 2 → should_reject=False
  5. AUTO_ACCEPT_CONTENT=True bypasses auto-reject even for grounding < 2
  6. suggested_complexity is normalised (unknown → "medium")
  7. rubric_scores_json contains all 4 keys with correct values
  8. FlashcardRepo correctly persists CriticResult scores to the DB
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
def mem_engine():
    from core.database import Base
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture(scope="module")
def mem_db(mem_engine):
    return sessionmaker(bind=mem_engine)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_flashcard(mem_engine, **kwargs):
    """Insert a minimal Flashcard row using the engine directly; return its id."""
    from sqlalchemy.orm import sessionmaker
    from core.database import Flashcard
    Session = sessionmaker(bind=mem_engine)
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


def _make_agent_with_mock(eval_result):
    """Return a CriticAgent whose chain returns the given RubricEvaluation."""
    from agents.critic import CriticAgent
    agent = CriticAgent.__new__(CriticAgent)
    agent._llm = MagicMock()
    agent._chain = MagicMock()
    agent._chain.invoke.return_value = eval_result
    return agent


# ---------------------------------------------------------------------------
# Pydantic validation tests — no API change needed
# ---------------------------------------------------------------------------

def test_scores_are_1_to_4():
    """Pydantic model rejects out-of-range values."""
    from agents.critic import RubricEvaluation
    import pydantic

    with pytest.raises((pydantic.ValidationError, ValueError)):
        RubricEvaluation(
            accuracy_score=5,  # invalid: max is 4
            logic_score=4, grounding_score=4, clarity_score=4,
            feedback="bad", suggested_complexity="medium",
        )

    with pytest.raises((pydantic.ValidationError, ValueError)):
        RubricEvaluation(
            accuracy_score=4,
            logic_score=0,  # invalid: min is 1
            grounding_score=4, clarity_score=4,
            feedback="bad", suggested_complexity="medium",
        )


def test_aggregate_score_is_rounded_mean():
    """Verify _aggregate helper computes round(mean) correctly."""
    from agents.critic import _aggregate
    assert _aggregate(4, 4, 4, 4) == 4
    assert _aggregate(1, 1, 1, 1) == 1
    assert _aggregate(4, 3, 2, 1) == round((4+3+2+1)/4)  # round(2.5) = 2
    assert _aggregate(3, 3, 3, 2) == round((3+3+3+2)/4)  # round(2.75) = 3


# ---------------------------------------------------------------------------
# Phase 2b: CriticResult field tests
# ---------------------------------------------------------------------------

def test_auto_reject_fires_when_grounding_lt_2():
    """grounding_score < 2 → CriticResult.should_reject is True."""
    from agents.critic import CriticAgent, CriticResult

    agent = _make_agent_with_mock(_make_eval_result(acc=4, log=3, grd=1, cla=3))

    with patch("agents.critic.call_structured_chain") as mock_call, \
         patch("agents.critic.settings") as mock_settings:
        mock_call.return_value = _make_eval_result(acc=4, log=3, grd=1, cla=3)
        mock_settings.AUTO_ACCEPT_CONTENT = False
        mock_settings.MODEL_HOP_ENABLED = False
        result = agent.evaluate_flashcard("source text", "question?", "answer.")

    assert isinstance(result, CriticResult)
    assert result.should_reject is True
    assert result.rubric_scores["grounding"] == 1
    assert "grounding" in result.reject_reason


def test_no_auto_reject_when_grounding_gte_2():
    """grounding_score >= 2 → should_reject is False."""
    from agents.critic import CriticAgent, CriticResult

    agent = _make_agent_with_mock(_make_eval_result(acc=4, log=3, grd=2, cla=3))

    with patch("agents.critic.call_structured_chain") as mock_call, \
         patch("agents.critic.settings") as mock_settings:
        mock_call.return_value = _make_eval_result(acc=4, log=3, grd=2, cla=3)
        mock_settings.AUTO_ACCEPT_CONTENT = False
        mock_settings.MODEL_HOP_ENABLED = False
        result = agent.evaluate_flashcard("source text", "question?", "answer.")

    assert isinstance(result, CriticResult)
    assert result.should_reject is False


def test_auto_accept_bypasses_auto_reject():
    """AUTO_ACCEPT_CONTENT=True → should_reject=False even for grounding < 2."""
    from agents.critic import CriticAgent, CriticResult

    agent = _make_agent_with_mock(_make_eval_result(acc=2, log=2, grd=1, cla=2))

    with patch("agents.critic.call_structured_chain") as mock_call, \
         patch("agents.critic.settings") as mock_settings:
        mock_call.return_value = _make_eval_result(acc=2, log=2, grd=1, cla=2)
        mock_settings.AUTO_ACCEPT_CONTENT = True
        mock_settings.MODEL_HOP_ENABLED = False
        result = agent.evaluate_flashcard("source text", "question?", "answer.")

    assert isinstance(result, CriticResult)
    assert result.should_reject is False


def test_complexity_returned_in_result():
    """suggested_complexity from Critic is available on CriticResult."""
    from agents.critic import CriticAgent, CriticResult

    agent = _make_agent_with_mock(_make_eval_result(acc=4, log=4, grd=4, cla=4, complexity="complex"))

    with patch("agents.critic.call_structured_chain") as mock_call, \
         patch("agents.critic.settings") as mock_settings:
        mock_call.return_value = _make_eval_result(acc=4, log=4, grd=4, cla=4, complexity="complex")
        mock_settings.AUTO_ACCEPT_CONTENT = False
        mock_settings.MODEL_HOP_ENABLED = False
        result = agent.evaluate_flashcard("source text", "question?", "answer.")

    assert isinstance(result, CriticResult)
    assert result.suggested_complexity == "complex"


def test_critic_rubric_scores_json_has_all_keys():
    """rubric_scores_json contains all 4 keys with correct values."""
    from agents.critic import CriticAgent, CriticResult

    agent = _make_agent_with_mock(_make_eval_result(acc=3, log=2, grd=4, cla=3))

    with patch("agents.critic.call_structured_chain") as mock_call, \
         patch("agents.critic.settings") as mock_settings:
        mock_call.return_value = _make_eval_result(acc=3, log=2, grd=4, cla=3)
        mock_settings.AUTO_ACCEPT_CONTENT = False
        mock_settings.MODEL_HOP_ENABLED = False
        result = agent.evaluate_flashcard("source text", "question?", "answer.")

    assert isinstance(result, CriticResult)
    scores = json.loads(result.rubric_scores_json)
    assert set(scores.keys()) == {"accuracy", "logic", "grounding", "clarity"}
    assert scores["accuracy"] == 3
    assert scores["logic"] == 2
    assert scores["grounding"] == 4
    assert scores["clarity"] == 3
    assert result.aggregate_score == round((3+2+4+3)/4)  # 3


def test_unknown_complexity_normalised_to_medium():
    """Critic normalises unknown complexity labels to 'medium'."""
    from agents.critic import CriticAgent, CriticResult

    agent = _make_agent_with_mock(_make_eval_result(complexity="hard"))

    with patch("agents.critic.call_structured_chain") as mock_call, \
         patch("agents.critic.settings") as mock_settings:
        mock_call.return_value = _make_eval_result(complexity="hard")
        mock_settings.AUTO_ACCEPT_CONTENT = False
        mock_settings.MODEL_HOP_ENABLED = False
        result = agent.evaluate_flashcard("source text", "question?", "answer.")

    assert isinstance(result, CriticResult)
    assert result.suggested_complexity == "medium"


# ---------------------------------------------------------------------------
# Phase 2b: FlashcardRepo persists CriticResult correctly
# ---------------------------------------------------------------------------

def test_flashcard_repo_persists_critic_scores(mem_engine, mem_db):
    """FlashcardRepo.update_critic_scores() writes rubric scores to the DB."""
    from core.database import Flashcard
    from repositories.sql.flashcard_repo import FlashcardRepo
    from agents.critic import CriticAgent, CriticResult

    fc_id = _insert_flashcard(mem_engine)

    agent = _make_agent_with_mock(_make_eval_result(acc=3, log=2, grd=4, cla=3, complexity="medium"))

    with patch("agents.critic.call_structured_chain") as mock_call, \
         patch("agents.critic.settings") as mock_settings:
        mock_call.return_value = _make_eval_result(acc=3, log=2, grd=4, cla=3, complexity="medium")
        mock_settings.AUTO_ACCEPT_CONTENT = False
        mock_settings.MODEL_HOP_ENABLED = False
        result = agent.evaluate_flashcard("source text", "question?", "answer.")

    # Simulate what the workflow node does: persist via FlashcardRepo
    with patch("repositories.sql.flashcard_repo.SessionLocal", mem_db):
        repo = FlashcardRepo()
        repo.update_critic_scores(
            flashcard_id=fc_id,
            aggregate_score=result.aggregate_score,
            rubric_scores_json=result.rubric_scores_json,
            feedback=result.feedback,
            complexity_level=result.suggested_complexity,
        )
        if result.should_reject:
            repo.update_status(fc_id, "rejected")

    db = mem_db()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        scores = json.loads(fc.critic_rubric_scores)
        assert set(scores.keys()) == {"accuracy", "logic", "grounding", "clarity"}
        assert scores["accuracy"] == 3
        assert scores["logic"] == 2
        assert fc.critic_score == 3
        assert fc.complexity_level == "medium"
    finally:
        db.close()


def test_flashcard_repo_rejects_when_should_reject(mem_engine, mem_db):
    """When should_reject=True, workflow calls update_status('rejected')."""
    from core.database import Flashcard
    from repositories.sql.flashcard_repo import FlashcardRepo
    from agents.critic import CriticAgent

    fc_id = _insert_flashcard(mem_engine)

    with patch("agents.critic.call_structured_chain") as mock_call, \
         patch("agents.critic.settings") as mock_settings:
        mock_call.return_value = _make_eval_result(acc=4, log=3, grd=1, cla=3)
        mock_settings.AUTO_ACCEPT_CONTENT = False
        mock_settings.MODEL_HOP_ENABLED = False
        agent = _make_agent_with_mock(_make_eval_result(acc=4, log=3, grd=1, cla=3))
        result = agent.evaluate_flashcard("source text", "question?", "answer.")

    assert result.should_reject is True

    with patch("repositories.sql.flashcard_repo.SessionLocal", mem_db):
        repo = FlashcardRepo()
        repo.update_critic_scores(fc_id, result.aggregate_score, result.rubric_scores_json, result.feedback, result.suggested_complexity)
        repo.update_status(fc_id, "rejected")

    db = mem_db()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        assert fc.status == "rejected"
    finally:
        db.close()
