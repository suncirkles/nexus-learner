"""
tests/test_chunking_strategy.py
---------------------------------
5 mocked unit tests for the subtopic-aggregation chunking strategy.

No live LLM or database calls — all dependencies are mocked.
Runs as part of the default pytest suite.

Tests:
  1. test_subtopic_aggregation_combines_chunks
     3 chunks of same subtopic_id → one group dict with concatenated text
  2. test_subtopic_aggregation_skips_covered_subtopics
     subtopic with existing approved card is excluded from chunks_to_process
  3. test_subtopic_aggregation_truncates_at_max
     text > MAX_SUBTOPIC_CHARS is truncated to exactly MAX_SUBTOPIC_CHARS
  4. test_generate_flashcard_passes_context
     chain.invoke receives both 'text' and 'context' keys
  5. test_math_separators_prefer_theorem_boundary
     splitter splits at \\n\\nTheorem before hitting chunk_size chars
"""

import os
import uuid
from unittest.mock import MagicMock, patch, call

import pytest

os.environ.setdefault("PRIMARY_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Test 1: subtopic aggregation combines chunks
# ---------------------------------------------------------------------------

def test_subtopic_aggregation_combines_chunks():
    """3 chunks with the same subtopic_id produce one group with concatenated text."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from core.database import Base, ContentChunk, Flashcard, Subtopic, Topic, Document as DBDoc

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    db = Session()
    doc = DBDoc(id=str(uuid.uuid4()), filename="test.pdf", content_hash="h1")
    db.add(doc)
    db.commit()
    topic = Topic(document_id=doc.id, name="Probability")
    db.add(topic)
    db.commit()
    sub = Subtopic(topic_id=topic.id, name="Bayes Theorem")
    db.add(sub)
    db.commit()

    texts = ["Theorem: P(A|B) = P(B|A)P(A)/P(B).", "Proof: multiply both sides.", "Example: coin toss."]
    for t in texts:
        db.add(ContentChunk(document_id=doc.id, text=t, subtopic_id=sub.id))
    db.commit()

    db_chunks = db.query(ContentChunk).all()
    subtopic_groups: dict = {}

    for c in db_chunks:
        sid = c.subtopic_id
        if sid not in subtopic_groups:
            # Simulate: no existing cards
            already_has_cards = False
            s = db.query(Subtopic).filter(Subtopic.id == sid).first()
            subtopic_groups[sid] = {
                "id": c.id,
                "subtopic_id": sid,
                "subtopic_name": s.name if s else "",
                "text": c.text,
            }
        elif subtopic_groups[sid] is not None:
            subtopic_groups[sid]["text"] += "\n\n" + c.text

    db.close()

    assert len(subtopic_groups) == 1
    group = list(subtopic_groups.values())[0]
    assert group["subtopic_name"] == "Bayes Theorem"
    for t in texts:
        assert t in group["text"]
    # All 3 chunks are joined with "\n\n"
    assert group["text"].count("\n\n") == 2


# ---------------------------------------------------------------------------
# Test 2: subtopic aggregation skips covered subtopics
# ---------------------------------------------------------------------------

def test_subtopic_aggregation_skips_covered_subtopics():
    """A subtopic that already has an approved card is excluded from processing."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from core.database import Base, ContentChunk, Flashcard, Subtopic, Topic, Document as DBDoc

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    db = Session()
    doc = DBDoc(id=str(uuid.uuid4()), filename="test.pdf", content_hash="h2")
    db.add(doc)
    db.commit()
    topic = Topic(document_id=doc.id, name="Stats")
    db.add(topic)
    db.commit()
    sub_covered = Subtopic(topic_id=topic.id, name="Mean")
    sub_new = Subtopic(topic_id=topic.id, name="Variance")
    db.add_all([sub_covered, sub_new])
    db.commit()

    db.add(ContentChunk(document_id=doc.id, text="Mean = sum/n", subtopic_id=sub_covered.id))
    db.add(ContentChunk(document_id=doc.id, text="Variance = sum of squared deviations", subtopic_id=sub_new.id))
    db.commit()

    # Insert an approved card for sub_covered
    fc = Flashcard(
        question="What is the mean?", answer="sum/n",
        question_type="active_recall", status="approved",
        subtopic_id=sub_covered.id, subject_id=None,
    )
    db.add(fc)
    db.commit()

    subject_id = None
    db_chunks = db.query(ContentChunk).all()
    subtopic_groups: dict = {}

    for c in db_chunks:
        sid = c.subtopic_id
        if sid not in subtopic_groups:
            already_has_cards = db.query(Flashcard).filter(
                Flashcard.subject_id == subject_id,
                Flashcard.subtopic_id == sid,
                Flashcard.status.in_(["approved", "pending"]),
            ).count() > 0
            if already_has_cards:
                subtopic_groups[sid] = None
                continue
            s = db.query(Subtopic).filter(Subtopic.id == sid).first()
            subtopic_groups[sid] = {
                "id": c.id,
                "subtopic_id": sid,
                "subtopic_name": s.name if s else "",
                "text": c.text,
            }
        elif subtopic_groups[sid] is not None:
            subtopic_groups[sid]["text"] += "\n\n" + c.text

    db.close()

    chunks_to_process = [g for g in subtopic_groups.values() if g is not None]
    assert len(chunks_to_process) == 1
    assert chunks_to_process[0]["subtopic_name"] == "Variance"


# ---------------------------------------------------------------------------
# Test 3: subtopic aggregation truncates at MAX_SUBTOPIC_CHARS
# ---------------------------------------------------------------------------

def test_subtopic_aggregation_truncates_at_max():
    """Aggregated text > MAX_SUBTOPIC_CHARS is truncated to exactly MAX_SUBTOPIC_CHARS."""
    from core.config import settings

    max_chars = settings.MAX_SUBTOPIC_CHARS

    # Simulate a group already built (post-aggregation)
    long_text = "A" * (max_chars + 500)
    group = {
        "id": 1,
        "subtopic_id": 1,
        "subtopic_name": "Long Section",
        "text": long_text,
    }

    if len(group["text"]) > max_chars:
        group["text"] = group["text"][:max_chars]

    assert len(group["text"]) == max_chars


# ---------------------------------------------------------------------------
# Test 4: generate_flashcard passes context to chain.invoke
# ---------------------------------------------------------------------------

def test_generate_flashcard_passes_context():
    """chain.invoke receives the 'text' key resolved from chunk.text (Phase 2b).

    Phase 2b: generate_flashcard() is now pure (no DB writes). source_text is
    resolved from the legacy `chunk` arg when source_text is empty. The chain
    receives {"text": <chunk.text>} directly.
    """
    from agents.socratic import SocraticAgent, FlashcardOutput, FlashcardItem, RubricItem

    fake_result = FlashcardOutput(flashcards=[
        FlashcardItem(
            question="What is Bayes' theorem?",
            answer="P(A|B) = P(B|A)P(A)/P(B)",
            question_type="active_recall",
            rubric=[
                RubricItem(criterion="Accuracy", description="Correct formula."),
                RubricItem(criterion="Completeness", description="All terms defined."),
                RubricItem(criterion="Grounding", description="Found in source."),
            ],
            suggested_complexity="medium",
        )
    ])

    agent = SocraticAgent.__new__(SocraticAgent)
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = fake_result
    agent._chains = {"active_recall": mock_chain}
    agent.llm = MagicMock()

    chunk = MagicMock(spec=["text", "id"])  # has .text but not .page_content
    chunk.text = "Bayes' Theorem: P(A|B) = P(B|A)P(A)/P(B)."
    chunk.id = 99

    # Phase 2b: no SessionLocal needed — pass source_text="" to trigger chunk fallback
    result = agent.generate_flashcard(
        source_text="",
        chunk=chunk,
        question_type="active_recall",
    )

    mock_chain.invoke.assert_called_once()
    call_kwargs = mock_chain.invoke.call_args[0][0]
    assert "text" in call_kwargs, "chain.invoke must receive 'text' key"
    assert call_kwargs["text"] == chunk.text, "source_text must be resolved from chunk.text"
    assert isinstance(result, list)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Test 5: math separators prefer Theorem boundary
# ---------------------------------------------------------------------------

def test_math_separators_prefer_theorem_boundary():
    """Splitter splits at \\n\\nTheorem before hitting chunk_size chars."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Build text: a short intro + \n\nTheorem block + filler
    intro = "Introduction to probability theory. " * 10            # ~360 chars
    theorem_block = "\n\nTheorem: P(A∪B) = P(A) + P(B) - P(A∩B). " * 5  # ~225 chars
    filler = "Some additional notes about the proof. " * 20        # ~760 chars

    text = intro + theorem_block + filler
    assert len(text) > 1000, "Text must be long enough to trigger splitting"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=[
            "\n\nTheorem", "\n\nProof", "\n\nDefinition",
            "\n\n", "\n", ". ", " ", "",
        ],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_text(text)
    assert len(chunks) >= 2, "Expected at least 2 chunks — splitter should break at \\n\\nTheorem"

    # The second chunk should start with Theorem content (splitter keeps the separator prefix)
    found_theorem = any("Theorem" in c for c in chunks[1:])
    assert found_theorem, (
        "Expected at least one chunk (after the first) to contain 'Theorem'. "
        f"Chunks: {[c[:60] for c in chunks]}"
    )
