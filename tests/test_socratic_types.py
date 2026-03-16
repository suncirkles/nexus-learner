"""
tests/test_socratic_types.py
------------------------------
Unit tests for the multi-format SocraticAgent.

Strategy: mock the per-type LangChain chain so no LLM API calls are made.
Each test verifies:
  - FlashcardItem schema fields are present and typed correctly
  - rubric contains exactly 3 RubricItems with criterion + description
  - suggested_complexity is one of {simple, medium, complex}
  - question_type is persisted to the DB Flashcard row
  - generate_flashcard() returns {"status": "success"}

A second fixture uses the Calculus PDF (Ch.1, p.1) to confirm the agent
loads correctly with a real academic source — the actual LLM call is still
mocked, so this remains a fast, deterministic unit test.
"""

import json
import os
import types
import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure cheaper model for any live calls (matches conftest convention)
os.environ.setdefault("PRIMARY_MODEL", "gpt-4o-mini")

CALCULUS_PDF = os.path.abspath("D:/cse/calculus/Advanced Engineering Mathematics 10th Edition.pdf")
CALCULUS_AVAILABLE = os.path.exists(CALCULUS_PDF)

# ---------------------------------------------------------------------------
# In-memory DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mem_db():
    """Isolated SQLite DB for these tests (not the project DB)."""
    from core.database import Base
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    return Session


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


def _make_chunk(text: str = "Sample source text about calculus derivatives."):
    """Minimal ORM-like chunk object."""
    chunk = MagicMock()
    chunk.text = text
    chunk.id = 42
    # Ensure hasattr(chunk, 'page_content') is False
    del chunk.page_content
    return chunk


# ---------------------------------------------------------------------------
# Parametrised schema tests (mocked chain)
# ---------------------------------------------------------------------------

QUESTION_TYPES = ["active_recall", "fill_blank", "short_answer", "long_answer", "numerical", "scenario"]


@pytest.mark.parametrize("qtype", QUESTION_TYPES)
def test_flashcard_item_schema(qtype, mem_db):
    """Each question type produces correct FlashcardItem schema."""
    from agents.socratic import SocraticAgent

    agent = SocraticAgent.__new__(SocraticAgent)
    fake_result = _make_fake_chain_result(qtype)

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = fake_result
    agent._chains = {qt: mock_chain for qt in QUESTION_TYPES}
    agent.llm = MagicMock()

    chunk = _make_chunk()

    # Patch SessionLocal to use the in-memory DB
    with patch("agents.socratic.SessionLocal", mem_db):
        result = agent.generate_flashcard(
            doc_id=str(uuid.uuid4()),
            chunk=chunk,
            subtopic_id=None,
            subject_id=None,
            question_type=qtype,
        )

    assert result["status"] == "success", f"Expected success for {qtype}, got: {result}"
    cards = result["flashcards"]
    assert len(cards) == 1

    card = cards[0]
    assert card["question_type"] == qtype
    assert isinstance(card["question"], str) and card["question"]
    assert isinstance(card["answer"], str) and card["answer"]
    assert card["suggested_complexity"] in ("simple", "medium", "complex")


@pytest.mark.parametrize("qtype", QUESTION_TYPES)
def test_rubric_saved_to_db(qtype, mem_db):
    """rubric JSON is persisted to Flashcard row with 3 criteria."""
    from agents.socratic import SocraticAgent
    from core.database import Flashcard

    agent = SocraticAgent.__new__(SocraticAgent)
    fake_result = _make_fake_chain_result(qtype)

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = fake_result
    agent._chains = {qt: mock_chain for qt in QUESTION_TYPES}
    agent.llm = MagicMock()

    chunk = _make_chunk()
    doc_id = str(uuid.uuid4())

    with patch("agents.socratic.SessionLocal", mem_db):
        result = agent.generate_flashcard(
            doc_id=doc_id,
            chunk=chunk,
            subtopic_id=None,
            subject_id=None,
            question_type=qtype,
        )

    fc_id = result["flashcards"][0]["flashcard_id"]
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


def test_unknown_question_type_falls_back_to_active_recall(mem_db):
    """An unrecognised question_type silently falls back to active_recall chain."""
    from agents.socratic import SocraticAgent

    agent = SocraticAgent.__new__(SocraticAgent)
    fake_result = _make_fake_chain_result("active_recall")

    ar_chain = MagicMock()
    ar_chain.invoke.return_value = fake_result
    agent._chains = {"active_recall": ar_chain}
    agent.llm = MagicMock()

    chunk = _make_chunk()
    with patch("agents.socratic.SessionLocal", mem_db):
        result = agent.generate_flashcard(
            doc_id=str(uuid.uuid4()),
            chunk=chunk,
            question_type="invalid_type",
        )

    assert result["status"] == "success"
    ar_chain.invoke.assert_called_once()


def test_empty_flashcard_list_returns_skipped(mem_db):
    """If LLM returns no flashcards, generate_flashcard() returns skipped."""
    from agents.socratic import SocraticAgent, FlashcardOutput

    agent = SocraticAgent.__new__(SocraticAgent)
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = FlashcardOutput(flashcards=[])
    agent._chains = {qt: mock_chain for qt in QUESTION_TYPES}
    agent.llm = MagicMock()

    chunk = _make_chunk()
    with patch("agents.socratic.SessionLocal", mem_db):
        result = agent.generate_flashcard(
            doc_id=str(uuid.uuid4()),
            chunk=chunk,
            question_type="active_recall",
        )

    assert result["status"] == "skipped"


# ---------------------------------------------------------------------------
# Calculus PDF smoke test (no LLM call — just verifies chunk loading)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CALCULUS_AVAILABLE, reason="Calculus PDF not available")
def test_calculus_pdf_chunk_text_nonempty():
    """Confirm IngestionAgent can extract text from Ch.1 of the calculus book.

    Page 0 is an image-only cover — scan pages 1-5 and accept any that has text.
    (Tesseract OCR not required; PyMuPDF handles embedded text.)
    """
    from agents.ingestion import IngestionAgent
    agent = IngestionAgent()
    found_text = ""
    for page_idx in range(1, 6):  # skip cover page, try early chapter pages
        page_text = agent.load_page_text(CALCULUS_PDF, page_idx)
        if len(page_text) > 100:
            found_text = page_text
            break
    assert isinstance(found_text, str)
    assert len(found_text) > 100, (
        "Expected at least one of pages 1-5 of the calculus PDF to contain "
        "embedded text (>100 chars). Check that PyMuPDF is installed."
    )
