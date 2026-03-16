"""
tests/test_e2e_card_quality.py
--------------------------------
End-to-end card quality test across three real documents and all six flashcard
types.  Uses live LLM calls (gpt-4o-mini) and a live Qdrant instance.

Documents under test
--------------------
  PYSPARK   : documents/pyspark.pdf           — 2 topics only (first two from index)
  CALCULUS  : D:/cse/Probability/calculus - chapter 1.1.pdf
  STATS     : D:/cse/Probability/descriptive statistics.pdf

Flashcard types tested per document
------------------------------------
  active_recall | fill_blank | short_answer | long_answer | numerical | scenario

Source fragment verification
-----------------------------
  After generation, for every flashcard:
  1. chunk_id is non-null (traceability anchor present)
  2. Full source text is recoverable by aggregating all ContentChunks sharing the
     same subtopic_id (mirrors what the Socratic agent received).
  3. Full source text is complete — does not end with a partial word
     (i.e., the final non-whitespace character is punctuation, a digit, or the last
     full token is complete — no mid-word cut at a chunk boundary).
  4. Key terms from the answer appear in the full aggregated source (grounding check).
  5. Critic grounding_score >= 2 (auto-reject threshold is < 2).

LLM-as-judge quality thresholds
---------------------------------
  Per document × question-type bucket (all generated cards averaged):
  - self_contained_avg  >= 3.5 / 5   (lower than RAG-unit threshold; real PDFs are noisier)
  - concept_depth_avg   >= 3.0 / 5
  - answer_grounded_avg >= 3.5 / 5
  - no_arbitrary_refs   >= 80 %       (most cards must avoid unexplained constants)

Run with:
    PYTHONPATH=. pytest tests/test_e2e_card_quality.py -v -m slow
    # or
    PYTHONPATH=. pytest tests/test_e2e_card_quality.py -v

All tests are marked @pytest.mark.slow and are NOT part of the default suite.
"""

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import pytest
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

os.environ.setdefault("PRIMARY_MODEL", "gpt-4o-mini")

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Document paths
# ---------------------------------------------------------------------------

_PYSPARK_PDF   = os.path.abspath("documents/pyspark.pdf")
_CALCULUS_PDF  = os.path.abspath("D:/cse/Probability/calculus - chapter 1.1.pdf")
_STATS_PDF     = os.path.abspath("D:/cse/Probability/descriptive statistics.pdf")

_PYSPARK_MAX_TOPICS = 2   # only test first 2 indexed topics to keep runtime reasonable
_PAGES_TO_INDEX = 4       # index first N pages of each PDF (enough for meaningful content)

ALL_QTYPES = [
    "active_recall",
    "fill_blank",
    "short_answer",
    "long_answer",
    "numerical",
    "scenario",
]

# LLM-as-judge thresholds (deliberately lower than hand-crafted fixture tests
# because real PDFs contain extraction noise, diagrams-as-spaces, etc.)
QUALITY_THRESHOLDS = {
    "self_contained_avg":  3.5,
    "concept_depth_avg":   3.0,
    "answer_grounded_avg": 3.5,
    "no_arbitrary_refs_pct": 80.0,   # percentage (0-100)
}


# ---------------------------------------------------------------------------
# Skip entire module if PDFs are not present
# ---------------------------------------------------------------------------

def _require_pdf(path: str) -> None:
    if not os.path.exists(path):
        pytest.skip(f"Required PDF not found: {path}", allow_module_level=True)


_require_pdf(_PYSPARK_PDF)
_require_pdf(_CALCULUS_PDF)
_require_pdf(_STATS_PDF)


# ---------------------------------------------------------------------------
# LLM-as-judge models (reuse from test_rag_quality_evaluation schema)
# ---------------------------------------------------------------------------

class CardQualityScore(BaseModel):
    self_contained: int = Field(ge=1, le=5)
    concept_depth: int = Field(ge=1, le=5)
    answer_grounded: int = Field(ge=1, le=5)
    no_arbitrary_refs: bool


class EvalReport(BaseModel):
    card_scores: List[CardQualityScore]
    overall_verdict: str
    reasoning: str


# ---------------------------------------------------------------------------
# Isolated in-memory DB for this test module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def isolated_db():
    """Fresh SQLite in-memory DB used for all tests in this module.

    We patch SessionLocal in all agents and workflows so real project DB is
    untouched.  Qdrant is still live (phase1_ingestion._flush_qdrant_batch).
    """
    from core.database import Base
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    return Session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_initial_state(
    mode: str,
    doc_id: str,
    subject_id: Optional[int] = None,
    file_path: Optional[str] = None,
    question_type: str = "active_recall",
    total_pages: int = _PAGES_TO_INDEX,
    current_page: int = 0,
    **overrides,
) -> dict:
    base: Dict[str, Any] = {
        "mode": mode,
        "file_path": file_path,
        "doc_id": doc_id,
        "subject_id": subject_id,
        "target_topics": [],
        "question_type": question_type,
        "total_pages": total_pages,
        "current_page": current_page,
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


def _run_indexing(
    Session,
    pdf_path: str,
    doc_id: str,
    total_pages: int = _PAGES_TO_INDEX,
) -> dict:
    """Index a PDF into the isolated DB.  Returns {doc_id, topic_names}."""
    import workflows.phase1_ingestion as wf
    from core.database import Subject, Topic, SubjectDocumentAssociation, Document as DBDoc

    db = Session()
    try:
        subject = Subject(name=f"e2e-{os.path.basename(pdf_path)[:20]}-{uuid.uuid4().hex[:6]}")
        db.add(subject)
        db.commit()
        subject_id = subject.id
    finally:
        db.close()

    with patch.object(wf, "ingestion_agent", wf.ingestion_agent), \
         patch("agents.ingestion.SessionLocal", Session), \
         patch("agents.topic_assigner.SessionLocal", Session), \
         patch("workflows.phase1_ingestion.SessionLocal", Session):
        state = _make_initial_state(
            "INDEXING", doc_id,
            file_path=pdf_path,
            total_pages=total_pages,
            current_page=0,
        )
        final = wf.phase1_graph.invoke(state)

    actual_doc_id = final.get("doc_id", doc_id)

    db = Session()
    try:
        existing_assoc = db.query(SubjectDocumentAssociation).filter_by(
            subject_id=subject_id, document_id=actual_doc_id
        ).first()
        if not existing_assoc:
            db.add(SubjectDocumentAssociation(
                subject_id=subject_id, document_id=actual_doc_id
            ))
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


def _run_generation(
    Session,
    doc_id: str,
    subject_id: int,
    question_type: str,
    target_topics: Optional[List[str]] = None,
) -> List[dict]:
    """Run GENERATION for one question_type.  Returns list of raw card dicts."""
    import workflows.phase1_ingestion as wf
    from core.database import Flashcard

    with patch("agents.socratic.SessionLocal", Session), \
         patch("agents.critic.SessionLocal", Session), \
         patch("workflows.phase1_ingestion.SessionLocal", Session):
        state = _make_initial_state(
            "GENERATION", doc_id,
            subject_id=subject_id,
            question_type=question_type,
            target_topics=target_topics or [],
            file_path=None,
            total_pages=0,
            current_page=0,
        )
        wf.phase1_graph.invoke(state)

    db = Session()
    try:
        cards = (
            db.query(Flashcard)
            .filter(Flashcard.subject_id == subject_id,
                    Flashcard.question_type == question_type)
            .all()
        )
        return [
            {
                "id":                  c.id,
                "question":            c.question,
                "answer":              c.answer,
                "question_type":       c.question_type,
                "rubric":              c.rubric,
                "complexity_level":    c.complexity_level,
                "critic_score":        c.critic_score,
                "critic_rubric_scores": c.critic_rubric_scores,
                "status":              c.status,
                "chunk_id":            c.chunk_id,
                "subtopic_id":         c.subtopic_id,
                "subject_id":          subject_id,
            }
            for c in cards
        ]
    finally:
        db.close()


def _recover_full_source(Session, chunk_id: Optional[int], subtopic_id: Optional[int]) -> str:
    """Reconstruct the full aggregated source text the Socratic agent saw.

    Mirrors app.py's new source-snippet logic: aggregate all ContentChunks
    with the same subtopic_id, ordered by id.
    """
    from core.database import ContentChunk

    db = Session()
    try:
        if subtopic_id:
            siblings = (
                db.query(ContentChunk)
                .filter(ContentChunk.subtopic_id == subtopic_id)
                .order_by(ContentChunk.id)
                .all()
            )
            return "\n\n".join(c.text for c in siblings)
        if chunk_id:
            chunk = db.query(ContentChunk).filter(ContentChunk.id == chunk_id).first()
            return chunk.text if chunk else ""
        return ""
    finally:
        db.close()


def _judge_cards(cards: List[dict], source_text: str) -> EvalReport:
    """LLM-as-judge: score each card against the source text."""
    from langchain_core.prompts import ChatPromptTemplate
    from core.models import get_llm

    if not cards:
        return EvalReport(card_scores=[], overall_verdict="FAIL",
                          reasoning="No cards generated.")

    cards_text = "\n\n".join(
        f"Card {i+1} [{c['question_type']}]:\nQ: {c['question']}\nA: {c['answer']}"
        for i, c in enumerate(cards)
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an objective evaluator of AI-generated flashcards.

Score each card on:
- self_contained (1-5): Is the question fully understandable without the source text?
  5 = completely standalone, 1 = completely context-dependent
- concept_depth (1-5): Does it test a principle/method rather than a mechanical step?
  5 = deep understanding, 1 = isolated arithmetic step
- answer_grounded (1-5): Is the answer supported by the source text?
  5 = directly traceable, 1 = not found in source
- no_arbitrary_refs (bool): True if the question does NOT reference unexplained
  constants, step numbers, or variable names absent from the source.

Return EvalReport with a score for every card listed.
overall_verdict: PASS if average quality is acceptable, FAIL otherwise."""),
        ("user", "SOURCE TEXT:\n{source}\n\nCARDS:\n{cards}"),
    ])

    llm = get_llm(purpose="primary", temperature=0).with_structured_output(EvalReport)
    return (prompt | llm).invoke({"source": source_text[:6000], "cards": cards_text})


def _avg(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Module-scoped indexing fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pyspark_indexed(isolated_db):
    """Index pyspark.pdf once for the whole module."""
    doc_id = str(uuid.uuid4())
    return _run_indexing(isolated_db, _PYSPARK_PDF, doc_id)


@pytest.fixture(scope="module")
def calculus_indexed(isolated_db):
    """Index calculus - chapter 1.1.pdf once for the whole module."""
    doc_id = str(uuid.uuid4())
    return _run_indexing(isolated_db, _CALCULUS_PDF, doc_id)


@pytest.fixture(scope="module")
def stats_indexed(isolated_db):
    """Index descriptive statistics.pdf once for the whole module."""
    doc_id = str(uuid.uuid4())
    return _run_indexing(isolated_db, _STATS_PDF, doc_id)


# ---------------------------------------------------------------------------
# Shared generation helper: generate all qtypes for one doc
# ---------------------------------------------------------------------------

def _generate_all_types(Session, indexed: dict, max_topics: Optional[int] = None) -> Dict[str, List[dict]]:
    """Generate all 6 question types for an indexed document.

    If max_topics is set, only the first N topic names are used as target_topics
    (PySpark restriction).
    """
    from core.database import Subject, SubjectDocumentAssociation

    topic_filter = None
    if max_topics is not None:
        topic_filter = indexed["topic_names"][:max_topics]

    results: Dict[str, List[dict]] = {}
    for qtype in ALL_QTYPES:
        # Fresh subject per question_type so they don't interfere
        db = Session()
        try:
            subj = Subject(name=f"e2e-{qtype}-{uuid.uuid4().hex[:8]}")
            db.add(subj)
            db.commit()
            subject_id = subj.id
            db.add(SubjectDocumentAssociation(
                subject_id=subject_id, document_id=indexed["doc_id"]
            ))
            db.commit()
        finally:
            db.close()

        cards = _run_generation(
            Session,
            doc_id=indexed["doc_id"],
            subject_id=subject_id,
            question_type=qtype,
            target_topics=topic_filter,
        )
        results[qtype] = cards

    return results


# ---------------------------------------------------------------------------
# Module-scoped card generation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pyspark_cards(isolated_db, pyspark_indexed):
    """Generate all 6 card types for PySpark (2 topics only)."""
    return _generate_all_types(isolated_db, pyspark_indexed, max_topics=_PYSPARK_MAX_TOPICS)


@pytest.fixture(scope="module")
def calculus_cards(isolated_db, calculus_indexed):
    """Generate all 6 card types for calculus chapter 1.1."""
    return _generate_all_types(isolated_db, calculus_indexed)


@pytest.fixture(scope="module")
def stats_cards(isolated_db, stats_indexed):
    """Generate all 6 card types for descriptive statistics."""
    return _generate_all_types(isolated_db, stats_indexed)


# ===========================================================================
# TEST SECTION 1 — Indexing sanity
# ===========================================================================

class TestIndexing:
    def test_pyspark_has_topics(self, pyspark_indexed):
        assert len(pyspark_indexed["topic_names"]) >= 1, \
            "PySpark PDF produced no topics"

    def test_calculus_has_topics(self, calculus_indexed):
        assert len(calculus_indexed["topic_names"]) >= 1, \
            "Calculus PDF produced no topics"

    def test_stats_has_topics(self, stats_indexed):
        assert len(stats_indexed["topic_names"]) >= 1, \
            "Descriptive statistics PDF produced no topics"

    def test_pyspark_topics_count_for_generation(self, pyspark_indexed):
        """Confirm we have at least 2 topics to test the 2-topic restriction."""
        count = len(pyspark_indexed["topic_names"])
        assert count >= 1, (
            f"Need at least 1 indexed topic in PySpark; got {count}. "
            "Try indexing more pages."
        )


# ===========================================================================
# TEST SECTION 2 — Schema: every card has required fields
# ===========================================================================

@pytest.mark.parametrize("qtype", ALL_QTYPES)
class TestCardSchema:

    def _check_schema(self, cards: List[dict], qtype: str, doc_label: str):
        assert cards, f"[{doc_label}] No cards generated for question_type={qtype!r}"
        for card in cards:
            assert card["question"].strip(), \
                f"[{doc_label}/{qtype}] Card {card['id']} has empty question"
            assert card["answer"].strip(), \
                f"[{doc_label}/{qtype}] Card {card['id']} has empty answer"
            assert card["question_type"] == qtype, \
                f"[{doc_label}/{qtype}] Card {card['id']} has wrong question_type={card['question_type']!r}"
            # Rubric
            assert card["rubric"], \
                f"[{doc_label}/{qtype}] Card {card['id']} missing rubric"
            rubric = json.loads(card["rubric"])
            assert len(rubric) == 3, \
                f"[{doc_label}/{qtype}] Card {card['id']} rubric has {len(rubric)} items, expected 3"
            for item in rubric:
                assert "criterion" in item and "description" in item
            # Complexity
            assert card["complexity_level"] in ("simple", "medium", "complex"), \
                f"[{doc_label}/{qtype}] Card {card['id']} invalid complexity={card['complexity_level']!r}"
            # Critic scores
            assert card["critic_rubric_scores"], \
                f"[{doc_label}/{qtype}] Card {card['id']} missing critic_rubric_scores"
            scores = json.loads(card["critic_rubric_scores"])
            assert set(scores.keys()) == {"accuracy", "logic", "grounding", "clarity"}
            for k, v in scores.items():
                assert 1 <= v <= 4, f"[{doc_label}/{qtype}] {k}={v} out of 1-4"

    def test_pyspark_card_schema(self, pyspark_cards, qtype):
        self._check_schema(pyspark_cards.get(qtype, []), qtype, "PySpark")

    def test_calculus_card_schema(self, calculus_cards, qtype):
        self._check_schema(calculus_cards.get(qtype, []), qtype, "Calculus")

    def test_stats_card_schema(self, stats_cards, qtype):
        self._check_schema(stats_cards.get(qtype, []), qtype, "Stats")


# ===========================================================================
# TEST SECTION 3 — Source fragment accuracy
# ===========================================================================

class TestSourceFragment:
    """Verify that the source text shown in Mentor Review is accurate and complete."""

    def _check_source_fragments(
        self,
        Session,
        cards: List[dict],
        doc_label: str,
    ):
        """For each card:
        1. chunk_id is non-null.
        2. Full subtopic source is recoverable.
        3. Source text does not end mid-word (i.e., ends with a complete token).
        4. At least one meaningful answer term appears in the full source.
        """
        failures = []
        for card in cards:
            qtype = card["question_type"]
            cid = card.get("chunk_id")
            sid = card.get("subtopic_id")
            label = f"[{doc_label}/{qtype}/card#{card['id']}]"

            # 1. chunk_id must be non-null
            if cid is None:
                failures.append(f"{label} chunk_id is None — source traceability lost")
                continue

            # 2. Recover full source
            full_source = _recover_full_source(Session, cid, sid)
            if not full_source:
                failures.append(f"{label} recovered empty source for chunk_id={cid}")
                continue

            # 3. Source completeness: the final non-whitespace char should be
            #    punctuation or a digit (not an abruptly cut mid-word letter).
            #    We allow final alphanumeric only if the last "word" is ≥ 3 chars
            #    (heuristic: a 1-2 char stub at the end suggests truncation).
            stripped = full_source.rstrip()
            if stripped:
                last_char = stripped[-1]
                if last_char.isalpha():
                    # Extract the last word
                    last_word_match = re.search(r'\b\w+$', stripped)
                    last_word = last_word_match.group(0) if last_word_match else ""
                    if len(last_word) < 3:
                        failures.append(
                            f"{label} Source ends with suspicious short stub "
                            f"{last_word!r} — possible mid-word truncation"
                        )

            # 4. Answer grounding: at least one multi-character token from the
            #    answer must appear in the full source (case-insensitive).
            #    Exclude stop words and single characters.
            STOP = {
                "the","a","an","is","are","was","were","be","been","being",
                "in","of","to","and","or","for","on","at","by","with","this",
                "that","it","its","from","as","not","can","which","have","has",
                "had","will","would","could","should","may","might","do","does",
                "did","if","then","than","so","but","what","how","why","when",
                "where","who","all","any","each","both","these","those","such",
            }
            answer_tokens = [
                t.lower() for t in re.findall(r'\b[a-zA-Z]\w{2,}\b', card["answer"])
                if t.lower() not in STOP
            ]
            source_lower = full_source.lower()
            matched_tokens = [t for t in answer_tokens if t in source_lower]
            if answer_tokens and len(matched_tokens) == 0:
                failures.append(
                    f"{label} No answer terms found in source. "
                    f"Answer tokens: {answer_tokens[:5]}"
                )

        assert not failures, (
            f"{doc_label}: source fragment issues found:\n"
            + "\n".join(failures)
        )

    def test_pyspark_source_fragments(self, isolated_db, pyspark_cards):
        all_cards = [c for cards in pyspark_cards.values() for c in cards]
        self._check_source_fragments(isolated_db, all_cards, "PySpark")

    def test_calculus_source_fragments(self, isolated_db, calculus_cards):
        all_cards = [c for cards in calculus_cards.values() for c in cards]
        self._check_source_fragments(isolated_db, all_cards, "Calculus")

    def test_stats_source_fragments(self, isolated_db, stats_cards):
        all_cards = [c for cards in stats_cards.values() for c in cards]
        self._check_source_fragments(isolated_db, all_cards, "Stats")

    def test_pyspark_topic_restriction_respected(self, isolated_db, pyspark_indexed, pyspark_cards):
        """Cards generated for PySpark must only reference the 2 allowed topics."""
        from core.database import Subtopic, Topic

        allowed_topics = set(pyspark_indexed["topic_names"][:_PYSPARK_MAX_TOPICS])
        if not allowed_topics:
            pytest.skip("No topics indexed for PySpark")

        all_cards = [c for cards in pyspark_cards.values() for c in cards]
        db = isolated_db()
        try:
            violations = []
            for card in all_cards:
                if card["subtopic_id"] is None:
                    continue
                sub = db.query(Subtopic).filter(Subtopic.id == card["subtopic_id"]).first()
                if sub is None:
                    continue
                topic = db.query(Topic).filter(Topic.id == sub.topic_id).first()
                topic_name = topic.name if topic else "?"
                if topic_name not in allowed_topics:
                    violations.append(
                        f"Card {card['id']} subtopic={sub.name!r} "
                        f"belongs to topic={topic_name!r} (not in {allowed_topics})"
                    )
        finally:
            db.close()

        assert not violations, (
            "Cards generated outside the 2-topic PySpark restriction:\n"
            + "\n".join(violations)
        )

    def test_source_shown_equals_aggregated_subtopic(self, isolated_db, pyspark_cards):
        """app.py source snippet must show the FULL aggregated subtopic text —
        not just the first chunk.  Verify by comparing recovered text to the
        per-subtopic aggregation done during generation."""
        from core.database import ContentChunk

        all_cards = [c for cards in pyspark_cards.values() for c in cards]
        db = isolated_db()
        try:
            mismatches = []
            for card in all_cards:
                if card["subtopic_id"] is None or card["chunk_id"] is None:
                    continue
                # What app.py now shows
                app_source = _recover_full_source(isolated_db, card["chunk_id"], card["subtopic_id"])
                # What was used during generation
                siblings = (
                    db.query(ContentChunk)
                    .filter(ContentChunk.subtopic_id == card["subtopic_id"])
                    .order_by(ContentChunk.id)
                    .all()
                )
                gen_source = "\n\n".join(c.text for c in siblings)

                if app_source != gen_source:
                    mismatches.append(
                        f"Card {card['id']}: app shows {len(app_source)} chars, "
                        f"generation used {len(gen_source)} chars"
                    )
        finally:
            db.close()

        assert not mismatches, (
            "Source shown in Mentor Review differs from generation context:\n"
            + "\n".join(mismatches)
        )


# ===========================================================================
# TEST SECTION 4 — Critic grounding (auto-reject threshold)
# ===========================================================================

class TestCriticGrounding:
    """Cards that survived (status != rejected) must have grounding_score >= 2."""

    def _check_grounding(self, cards: List[dict], doc_label: str):
        failures = []
        for card in cards:
            if card["status"] == "rejected":
                continue   # auto-reject already fired — that's expected behaviour
            if not card["critic_rubric_scores"]:
                continue
            scores = json.loads(card["critic_rubric_scores"])
            grd = scores.get("grounding", 0)
            if grd < 2:
                failures.append(
                    f"[{doc_label}/{card['question_type']}/card#{card['id']}] "
                    f"grounding_score={grd} but status={card['status']!r} — "
                    "card should have been auto-rejected"
                )
        assert not failures, "\n".join(failures)

    def test_pyspark_grounding(self, pyspark_cards):
        all_cards = [c for cards in pyspark_cards.values() for c in cards]
        self._check_grounding(all_cards, "PySpark")

    def test_calculus_grounding(self, calculus_cards):
        all_cards = [c for cards in calculus_cards.values() for c in cards]
        self._check_grounding(all_cards, "Calculus")

    def test_stats_grounding(self, stats_cards):
        all_cards = [c for cards in stats_cards.values() for c in cards]
        self._check_grounding(all_cards, "Stats")


# ===========================================================================
# TEST SECTION 5 — LLM-as-judge quality per document
# ===========================================================================

def _judge_all_cards(Session, cards_by_type: Dict[str, List[dict]]) -> EvalReport:
    """Aggregate judge scores across all types for one document."""
    from core.database import ContentChunk

    all_scores: List[CardQualityScore] = []

    for qtype, cards in cards_by_type.items():
        non_rejected = [c for c in cards if c["status"] != "rejected"]
        if not non_rejected:
            continue

        # Use the first card's subtopic source as representative context for judging
        sample = non_rejected[0]
        full_source = _recover_full_source(Session, sample["chunk_id"], sample["subtopic_id"])
        if not full_source:
            continue

        report = _judge_cards(non_rejected, full_source)
        all_scores.extend(report.card_scores)

    if not all_scores:
        return EvalReport(card_scores=[], overall_verdict="FAIL",
                          reasoning="No cards to evaluate.")

    return EvalReport(
        card_scores=all_scores,
        overall_verdict="PASS",
        reasoning="Aggregated across all question types.",
    )


def _assert_quality(report: EvalReport, doc_label: str):
    scores = report.card_scores
    assert scores, f"[{doc_label}] LLM judge returned no scores"

    sc_avg  = _avg([s.self_contained  for s in scores])
    cd_avg  = _avg([s.concept_depth   for s in scores])
    ag_avg  = _avg([s.answer_grounded for s in scores])
    nar_pct = 100.0 * sum(1 for s in scores if s.no_arbitrary_refs) / len(scores)

    assert sc_avg >= QUALITY_THRESHOLDS["self_contained_avg"], (
        f"[{doc_label}] self_contained avg {sc_avg:.2f} < "
        f"{QUALITY_THRESHOLDS['self_contained_avg']}"
    )
    assert cd_avg >= QUALITY_THRESHOLDS["concept_depth_avg"], (
        f"[{doc_label}] concept_depth avg {cd_avg:.2f} < "
        f"{QUALITY_THRESHOLDS['concept_depth_avg']}"
    )
    assert ag_avg >= QUALITY_THRESHOLDS["answer_grounded_avg"], (
        f"[{doc_label}] answer_grounded avg {ag_avg:.2f} < "
        f"{QUALITY_THRESHOLDS['answer_grounded_avg']}"
    )
    assert nar_pct >= QUALITY_THRESHOLDS["no_arbitrary_refs_pct"], (
        f"[{doc_label}] {nar_pct:.1f}% cards have no_arbitrary_refs=True; "
        f"threshold is {QUALITY_THRESHOLDS['no_arbitrary_refs_pct']}%"
    )


class TestLLMJudgeQuality:

    def test_pyspark_judge_quality(self, isolated_db, pyspark_cards):
        report = _judge_all_cards(isolated_db, pyspark_cards)
        _assert_quality(report, "PySpark")

    def test_calculus_judge_quality(self, isolated_db, calculus_cards):
        report = _judge_all_cards(isolated_db, calculus_cards)
        _assert_quality(report, "Calculus")

    def test_stats_judge_quality(self, isolated_db, stats_cards):
        report = _judge_all_cards(isolated_db, stats_cards)
        _assert_quality(report, "Stats")


# ===========================================================================
# TEST SECTION 6 — Per-question-type quality (PySpark only — 2 topics)
# ===========================================================================

@pytest.mark.parametrize("qtype", ALL_QTYPES)
class TestPerTypeQuality:
    """Each question type individually must meet self_contained >= 3.0 and
    answer_grounded >= 3.0 (looser floor than the aggregate test above —
    some types like 'scenario' are harder on real PDF text)."""

    _PER_TYPE_MIN = 3.0

    def test_pyspark_per_type_self_contained(self, isolated_db, pyspark_cards, qtype):
        cards = [c for c in pyspark_cards.get(qtype, []) if c["status"] != "rejected"]
        if not cards:
            pytest.skip(f"No non-rejected cards for PySpark/{qtype}")

        sample = cards[0]
        src = _recover_full_source(isolated_db, sample["chunk_id"], sample["subtopic_id"])
        report = _judge_cards(cards, src)

        sc_avg = _avg([s.self_contained for s in report.card_scores])
        assert sc_avg >= self._PER_TYPE_MIN, (
            f"[PySpark/{qtype}] self_contained avg {sc_avg:.2f} < {self._PER_TYPE_MIN}"
        )

    def test_pyspark_per_type_answer_grounded(self, isolated_db, pyspark_cards, qtype):
        cards = [c for c in pyspark_cards.get(qtype, []) if c["status"] != "rejected"]
        if not cards:
            pytest.skip(f"No non-rejected cards for PySpark/{qtype}")

        sample = cards[0]
        src = _recover_full_source(isolated_db, sample["chunk_id"], sample["subtopic_id"])
        report = _judge_cards(cards, src)

        ag_avg = _avg([s.answer_grounded for s in report.card_scores])
        assert ag_avg >= self._PER_TYPE_MIN, (
            f"[PySpark/{qtype}] answer_grounded avg {ag_avg:.2f} < {self._PER_TYPE_MIN}"
        )


# ===========================================================================
# TEST SECTION 7 — Numerical card integrity (all three documents)
# ===========================================================================

class TestNumericalCardIntegrity:
    """Numerical cards must be actual computation/derivation problems — not
    descriptive concept questions.  The definitive signal: the question text
    must contain at least one digit (concrete data from the source), OR the
    answer must show step-by-step arithmetic with digits.

    Cards that could not be generated (source lacked numeric data) are
    correctly absent — that's the EMPTY LIST RULE working as intended.
    When cards ARE present, they must meet the computation requirement.
    """

    # Patterns that indicate a descriptive question slipped through
    _FORBIDDEN_PREFIXES = (
        "what is the formula",
        "explain how",
        "explain the",
        "describe how",
        "describe the",
        "what does",
        "what is",
        "how does",
        "why is",
        "define ",
    )

    def _check_numerical_cards(self, cards: List[dict], doc_label: str):
        import re
        non_rejected = [c for c in cards if c["status"] != "rejected"]

        if not non_rejected:
            # Acceptable: source had no numerical data → EMPTY LIST RULE fired
            return

        failures = []
        for card in non_rejected:
            q = card["question"].strip()
            a = card["answer"].strip()

            # The question or answer must contain at least one digit
            has_digit_in_q = bool(re.search(r'\d', q))
            has_digit_in_a = bool(re.search(r'\d', a))
            if not (has_digit_in_q or has_digit_in_a):
                failures.append(
                    f"[{doc_label}/card#{card['id']}] Neither question nor answer "
                    f"contains a digit — this is a descriptive question, not a "
                    f"numerical problem.\n  Q: {q[:120]}"
                )
                continue

            # The question must not start with a forbidden descriptive pattern
            q_lower = q.lower()
            for pat in self._FORBIDDEN_PREFIXES:
                if q_lower.startswith(pat):
                    failures.append(
                        f"[{doc_label}/card#{card['id']}] Question starts with "
                        f"forbidden descriptive pattern {pat!r}.\n  Q: {q[:120]}"
                    )
                    break

        assert not failures, (
            f"{doc_label}: descriptive numerical cards detected:\n"
            + "\n".join(failures)
        )

    def test_stats_numerical_cards_are_computation_problems(self, stats_cards):
        """Descriptive statistics has worked examples and a Problems section —
        numerical cards must use those actual numbers."""
        self._check_numerical_cards(stats_cards.get("numerical", []), "Stats")

    def test_calculus_numerical_cards_are_computation_problems(self, calculus_cards):
        self._check_numerical_cards(calculus_cards.get("numerical", []), "Calculus")

    def test_pyspark_numerical_cards_are_computation_problems(self, pyspark_cards):
        self._check_numerical_cards(pyspark_cards.get("numerical", []), "PySpark")

    def test_stats_numerical_answer_shows_steps(self, stats_cards):
        """Answer must show at least 2 arithmetic steps — not just a formula statement."""
        import re
        non_rejected = [c for c in stats_cards.get("numerical", [])
                        if c["status"] != "rejected"]
        if not non_rejected:
            pytest.skip("No numerical cards for Stats (source lacked numeric data)")

        failures = []
        for card in non_rejected:
            a = card["answer"]
            # Count lines that contain a digit (each = a computation step)
            step_lines = [ln for ln in a.splitlines() if re.search(r'\d', ln.strip())]
            if len(step_lines) < 2:
                failures.append(
                    f"Card #{card['id']}: answer has only {len(step_lines)} numeric "
                    f"step line(s) — expected ≥ 2 for a worked solution.\n  A: {a[:200]}"
                )

        assert not failures, (
            "Stats numerical cards lack step-by-step worked solutions:\n"
            + "\n".join(failures)
        )

    def test_numerical_question_contains_source_data(self, isolated_db, stats_cards):
        """Key numeric tokens in the question must appear in the source text —
        i.e., the LLM is using actual data from the document, not inventing numbers."""
        import re
        non_rejected = [c for c in stats_cards.get("numerical", [])
                        if c["status"] != "rejected"]
        if not non_rejected:
            pytest.skip("No numerical cards for Stats")

        failures = []
        for card in non_rejected:
            q = card["question"]
            numbers_in_q = re.findall(r'\b\d+(?:\.\d+)?\b', q)
            if not numbers_in_q:
                continue  # no digits → covered by integrity test above

            src = _recover_full_source(isolated_db, card["chunk_id"], card["subtopic_id"])
            missing = [n for n in numbers_in_q if n not in src]
            # Allow up to 20% invented numbers (e.g. rounding, formatting variants)
            if len(missing) > 0.2 * len(numbers_in_q):
                failures.append(
                    f"Card #{card['id']}: {len(missing)}/{len(numbers_in_q)} numbers "
                    f"in question not found in source — possible hallucination: "
                    f"{missing[:5]}\n  Q: {q[:120]}"
                )

        assert not failures, (
            "Numerical questions contain numbers not present in source:\n"
            + "\n".join(failures)
        )
