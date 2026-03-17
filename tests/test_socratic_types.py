"""
tests/test_socratic_types.py
------------------------------
Unit tests for multi-format SocraticAgent (Phase 2b API).

Phase 2b: generate_flashcard() returns List[FlashcardDraft] — no DB writes.
DB persistence is the workflow node's responsibility (via FlashcardRepo).

Tests:
  - FlashcardDraft schema fields are present and typed correctly
  - rubric_json contains exactly 3 criteria with criterion + description
  - suggested_complexity is one of {simple, medium, complex}
  - question_type on the draft reflects the caller-requested type
  - FlashcardRepo correctly persists drafts to the DB
  - Unknown question_type falls back to active_recall chain
  - Empty LLM response returns empty list
"""

import json
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("PRIMARY_MODEL", "gpt-4o-mini")

CALCULUS_PDF = os.path.abspath("D:/cse/calculus/Advanced Engineering Mathematics 10th Edition.pdf")
CALCULUS_AVAILABLE = os.path.exists(CALCULUS_PDF)

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

def _make_fake_chain_result(question_type: str):
    """Returns a fake FlashcardOutput matching the expected schema."""
    from agents.socratic import FlashcardOutput, FlashcardItem, RubricItem
    return FlashcardOutput(flashcards=[
        FlashcardItem(
            question=f"Sample {question_type} question?",
            answer="Sample answer grounded in source.",
            question_type=question_type,
            rubric=[
                RubricItem(criterion="Accuracy", description="Answer matches source."),
                RubricItem(criterion="Completeness", description="All key points included."),
                RubricItem(criterion="Grounding", description="Cited from source text."),
            ],
            suggested_complexity="medium",
        )
    ])


def _make_agent(question_types=None):
    """Return a SocraticAgent with mocked chains."""
    from agents.socratic import SocraticAgent
    agent = SocraticAgent.__new__(SocraticAgent)
    agent.llm = MagicMock()
    qtypes = question_types or ["active_recall", "fill_blank", "short_answer", "long_answer", "numerical", "scenario"]
    agent._chains = {}
    for qt in qtypes:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _make_fake_chain_result(qt)
        agent._chains[qt] = mock_chain
    return agent


QUESTION_TYPES = ["active_recall", "fill_blank", "short_answer", "long_answer", "numerical", "scenario"]


# ---------------------------------------------------------------------------
# Phase 2b: FlashcardDraft schema tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("qtype", QUESTION_TYPES)
def test_flashcard_item_schema(qtype):
    """Each question type produces correct FlashcardDraft schema."""
    from agents.socratic import FlashcardDraft

    agent = _make_agent()
    # Install the per-type chain mock
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _make_fake_chain_result(qtype)
    agent._chains[qtype] = mock_chain

    result = agent.generate_flashcard(source_text="Sample source text.", question_type=qtype)

    assert isinstance(result, list)
    assert len(result) == 1
    draft = result[0]
    assert isinstance(draft, FlashcardDraft)
    assert draft.question_type == qtype
    assert isinstance(draft.question, str) and draft.question
    assert isinstance(draft.answer, str) and draft.answer
    assert draft.suggested_complexity in ("simple", "medium", "complex")


@pytest.mark.parametrize("qtype", QUESTION_TYPES)
def test_rubric_json_has_three_criteria(qtype):
    """rubric_json contains exactly 3 criteria with criterion + description."""
    agent = _make_agent()
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _make_fake_chain_result(qtype)
    agent._chains[qtype] = mock_chain

    result = agent.generate_flashcard(source_text="Sample source text.", question_type=qtype)

    assert len(result) == 1
    rubric = json.loads(result[0].rubric_json)
    assert isinstance(rubric, list)
    assert len(rubric) == 3
    for item in rubric:
        assert "criterion" in item
        assert "description" in item


def test_unknown_question_type_falls_back_to_active_recall():
    """An unrecognised question_type silently falls back to active_recall chain."""
    from agents.socratic import SocraticAgent, FlashcardDraft

    agent = SocraticAgent.__new__(SocraticAgent)
    agent.llm = MagicMock()
    ar_chain = MagicMock()
    ar_chain.invoke.return_value = _make_fake_chain_result("active_recall")
    agent._chains = {"active_recall": ar_chain}

    result = agent.generate_flashcard(source_text="text", question_type="invalid_type")

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], FlashcardDraft)
    ar_chain.invoke.assert_called_once()


def test_empty_flashcard_list_returns_empty():
    """If LLM returns no flashcards, generate_flashcard() returns empty list."""
    from agents.socratic import SocraticAgent, FlashcardOutput

    agent = SocraticAgent.__new__(SocraticAgent)
    agent.llm = MagicMock()
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = FlashcardOutput(flashcards=[])
    agent._chains = {qt: mock_chain for qt in QUESTION_TYPES}

    result = agent.generate_flashcard(source_text="text", question_type="active_recall")
    assert result == []


# ---------------------------------------------------------------------------
# Phase 2b: FlashcardRepo persists FlashcardDraft correctly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("qtype", QUESTION_TYPES)
def test_rubric_saved_to_db_via_repo(qtype, mem_engine, mem_db):
    """FlashcardRepo.create() correctly persists FlashcardDraft to DB."""
    from agents.socratic import FlashcardDraft
    from repositories.sql.flashcard_repo import FlashcardRepo
    from core.database import Flashcard

    agent = _make_agent()
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _make_fake_chain_result(qtype)
    agent._chains[qtype] = mock_chain

    drafts = agent.generate_flashcard(source_text="Sample source text.", question_type=qtype)
    assert len(drafts) == 1
    draft = drafts[0]

    # Simulate workflow node: persist via FlashcardRepo
    with patch("repositories.sql.flashcard_repo.SessionLocal", mem_db):
        repo = FlashcardRepo()
        saved = repo.create(
            subject_id=None,
            subtopic_id=None,
            chunk_id=None,
            question=draft.question,
            answer=draft.answer,
            question_type=draft.question_type,
            rubric_json=draft.rubric_json,
            status="pending",
        )

    fc_id = saved["id"]
    db = mem_db()
    try:
        fc = db.query(Flashcard).filter(Flashcard.id == fc_id).first()
        assert fc is not None
        assert fc.question_type == qtype
        assert fc.rubric is not None
        rubric = json.loads(fc.rubric)
        assert isinstance(rubric, list)
        assert len(rubric) == 3
        for item in rubric:
            assert "criterion" in item
            assert "description" in item
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Calculus PDF smoke test — verifies IngestionAgent text extraction only
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CALCULUS_AVAILABLE, reason="Calculus PDF not available")
def test_calculus_pdf_chunk_text_nonempty():
    """Confirm IngestionAgent can extract text from Ch.1 of the calculus book."""
    from agents.ingestion import IngestionAgent
    agent = IngestionAgent()
    found_text = ""
    for page_idx in range(1, 6):
        page_text = agent.load_page_text(CALCULUS_PDF, page_idx)
        if len(page_text) > 100:
            found_text = page_text
            break
    assert isinstance(found_text, str)
    assert len(found_text) > 100
