"""
tests/test_e2e.py
------------------
End-to-end test for the Phase 1 LangGraph workflow (Phase 2b architecture).

Architecture:
  INDEXING:   Ingest → AssignTopic (TopicAssignerAgent) → NextPage → FlushQdrant
  GENERATION: MatchTopics → Ingest → Generate (SocraticAgent) → Critic (CriticAgent) → Increment → END

Phase 2b mocking strategy:
  - TopicAssignerAgent.assign_topic  → deterministic TopicAssignment (no LLM)
  - _flush_vector_batch              → no-op (vector store not required)
  - SocraticAgent.generate_flashcard → returns List[FlashcardDraft] (no DB write)
  - CriticAgent.evaluate_flashcard   → returns CriticResult (no DB write)
  - FlashcardRepo persists for real in the test DB
"""

import os
import uuid
import json
import pytest
from unittest.mock import patch, MagicMock

import workflows.phase1_ingestion
from core.database import (
    SessionLocal, ContentChunk, Flashcard,
    Topic, Subtopic, Subject, Base, engine,
    SubjectDocumentAssociation, Document as DBDocument,
)
from agents.topic_assigner import TopicAssignment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function", autouse=True)
def setup_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


@pytest.fixture
def dummy_pdf(tmp_path):
    """Create a minimal PDF with embedded text so PyMuPDF can parse it."""
    import fitz
    pdf_path = str(tmp_path / "test_e2e.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "Calculus is the mathematical study of continuous change. "
        "The derivative measures instantaneous rate of change. "
        "Integration is the inverse of differentiation and accumulates quantities.",
    )
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def mock_topic_assigner():
    """Make TopicAssignerAgent return a deterministic assignment without LLM calls."""
    assignment = TopicAssignment(
        topic_name="Calculus",
        subtopic_name="Derivatives",
        reasoning="Chunk discusses rates of change.",
    )
    with patch("agents.topic_assigner.TopicAssignerAgent.assign_topic", return_value=assignment):
        yield assignment


@pytest.fixture
def mock_qdrant():
    """Suppress vector batch flush — vector store not required for this test."""
    with patch("workflows.phase1_ingestion._flush_vector_batch"):
        yield


@pytest.fixture
def mock_title_llm():
    """Suppress LLM title generation inside IngestionAgent.create_document_record."""
    with patch("core.models.get_llm", side_effect=RuntimeError("skip title gen")):
        yield


def _make_state(mode, doc_id, subject_id=None, file_path=None, **overrides):
    base = {
        "mode": mode,
        "file_path": file_path,
        "doc_id": doc_id,
        "subject_id": subject_id,
        "target_topics": [],
        "question_type": "active_recall",
        "total_pages": 1,
        "current_page": 0,
        "chunks": [],
        "current_chunk_index": 0,
        "hierarchy": [],
        "pending_vector_docs": [],
        "current_new_cards": [],
        "generated_flashcards": [],
        "status_message": "start",
        "matched_subtopic_ids": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_e2e_indexing_creates_topics_and_chunks(
    dummy_pdf, mock_topic_assigner, mock_qdrant, mock_title_llm
):
    """INDEXING mode: chunk → assign → persist Topic/Subtopic/ContentChunk in DB."""
    doc_id = str(uuid.uuid4())
    state = _make_state("INDEXING", doc_id, file_path=dummy_pdf)
    workflows.phase1_ingestion.phase1_graph.invoke(state)

    db = SessionLocal()
    try:
        doc = db.query(DBDocument).first()
        assert doc is not None, "node_ingest should have created a Document row"

        actual_doc_id = doc.id
        topics = db.query(Topic).filter(Topic.document_id == actual_doc_id).all()
        assert len(topics) >= 1, "Expected at least 1 topic indexed"
        assert topics[0].name == "Calculus"

        subtopic = db.query(Subtopic).filter(Subtopic.topic_id == topics[0].id).first()
        assert subtopic is not None
        assert subtopic.name == "Derivatives"

        chunks = db.query(ContentChunk).filter(ContentChunk.document_id == actual_doc_id).all()
        assert len(chunks) >= 1, "Expected ContentChunk rows saved to DB"
    finally:
        db.close()


def test_e2e_generation_creates_flashcards(
    dummy_pdf, mock_topic_assigner, mock_qdrant, mock_title_llm
):
    """GENERATION mode: after indexing, mocked agents produce Flashcard rows in DB.

    Phase 2b: generate_flashcard returns List[FlashcardDraft]; node_generate
    persists via FlashcardRepo. evaluate_flashcard returns CriticResult;
    node_critic persists via FlashcardRepo.
    """
    doc_id = str(uuid.uuid4())

    # --- INDEXING first (populates Topics/Subtopics/Chunks) ---
    index_state = _make_state("INDEXING", doc_id, file_path=dummy_pdf)
    workflows.phase1_ingestion.phase1_graph.invoke(index_state)

    db = SessionLocal()
    try:
        actual_doc = db.query(DBDocument).first()
        assert actual_doc is not None
        actual_doc_id = actual_doc.id

        subject = Subject(name=f"E2E-Gen-{uuid.uuid4().hex[:6]}")
        db.add(subject)
        db.commit()
        subject_id = subject.id
        db.add(SubjectDocumentAssociation(subject_id=subject_id, document_id=actual_doc_id))
        db.commit()
    finally:
        db.close()

    # --- Mock generation agents with Phase 2b APIs ---
    from agents.socratic import FlashcardDraft
    from agents.critic import CriticResult

    def fake_generate(self, source_text="", question_type="active_recall", **kwargs):
        """Returns List[FlashcardDraft] — no DB writes (Phase 2b)."""
        return [FlashcardDraft(
            question="What does a derivative measure?",
            answer="Instantaneous rate of change.",
            question_type=question_type,
            rubric_json=json.dumps([
                {"criterion": "Accuracy", "description": "Matches source."},
                {"criterion": "Logic", "description": "Reasoning sound."},
                {"criterion": "Grounding", "description": "From source."},
            ]),
            suggested_complexity="medium",
        )]

    def fake_evaluate(self, source_text="", question="", answer="", flashcard_id=None):
        """Returns CriticResult — no DB writes (Phase 2b)."""
        return CriticResult(
            aggregate_score=4,
            rubric_scores={"accuracy": 4, "logic": 4, "grounding": 4, "clarity": 4},
            rubric_scores_json=json.dumps({"accuracy": 4, "logic": 4, "grounding": 4, "clarity": 4}),
            feedback="Well grounded.",
            suggested_complexity="medium",
            should_reject=False,
            reject_reason="",
        )

    with patch("agents.socratic.SocraticAgent.generate_flashcard", fake_generate), \
         patch("agents.critic.CriticAgent.evaluate_flashcard", fake_evaluate):

        gen_state = _make_state(
            "GENERATION", actual_doc_id, subject_id=subject_id,
            file_path=None, total_pages=0, current_page=0,
        )
        workflows.phase1_ingestion.phase1_graph.invoke(gen_state)

    db = SessionLocal()
    try:
        flashcards = db.query(Flashcard).filter(Flashcard.subject_id == subject_id).all()
        assert len(flashcards) >= 1, "Expected at least 1 Flashcard in DB after generation"
        fc = flashcards[0]
        assert fc.question == "What does a derivative measure?"
        assert fc.status == "pending"
        assert fc.question_type == "active_recall"
        assert fc.critic_score == 4
    finally:
        db.close()
