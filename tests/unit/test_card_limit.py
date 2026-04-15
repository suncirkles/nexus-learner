"""
tests/unit/test_card_limit.py
------------------------------
Verifies that MAX_CARDS_PER_TOPIC_TYPE is enforced at the card level inside
node_generate, not just at the chunk level in node_ingest.

The bug: node_ingest limited *chunks* to MAX_CARDS_PER_TOPIC_TYPE, but
SocraticAgent returns 1-3 cards per chunk, so 5 chunks * 3 cards = 15 cards
were written even with the limit set to 5.

These tests are pure unit tests — no LLM calls, no DB connection.
"""
from unittest.mock import MagicMock, patch, call
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(topic_id=10, subtopic_id=20, chunk_id=1):
    return {
        "id": chunk_id,
        "text": "some content about d and f block",
        "topic_id": topic_id,
        "subtopic_id": subtopic_id,
    }


def _make_draft(q="Q?", a="A."):
    from dataclasses import dataclass
    @dataclass
    class Draft:
        question: str = q
        answer: str = a
        question_type: str = "active_recall"
        rubric_json: str = "{}"
        suggested_complexity: str = "medium"
    return Draft()


def _make_state(chunks, subject_id=1, question_type="active_recall", generated=None):
    return {
        "chunks": chunks,
        "current_chunk_index": 0,
        "subject_id": subject_id,
        "question_type": question_type,
        "generated_flashcards": generated or [],
    }


def _run_node_generate(drafts, cards_already_in_db, topic_limit, subject_id=1):
    """
    Call node_generate with a controlled number of existing DB cards and drafts.
    Returns the list of new cards that were saved.
    """
    from workflows import phase1_ingestion

    chunk = _make_chunk()
    state = _make_state([chunk], subject_id=subject_id)

    # SocraticAgent returns `drafts`
    mock_socratic = MagicMock()
    mock_socratic.generate_flashcard.return_value = drafts

    # FlashcardRepo.create returns a fake saved record
    saved_counter = [0]
    def fake_create(**kwargs):
        saved_counter[0] += 1
        return {
            "id": saved_counter[0],
            "subject_id": kwargs["subject_id"],
            "subtopic_id": kwargs["subtopic_id"],
            "status": kwargs["status"],
            "question": kwargs["question"],
            "answer": kwargs["answer"],
            "question_type": kwargs["question_type"],
        }

    mock_fc_repo = MagicMock()
    mock_fc_repo.create.side_effect = fake_create

    # DB query for existing card count returns cards_already_in_db
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.scalar.return_value = cards_already_in_db

    with (
        patch.object(phase1_ingestion, "_socratic_agent", mock_socratic),
        patch("workflows.phase1_ingestion.FlashcardRepo", return_value=mock_fc_repo),
        patch("workflows.phase1_ingestion.SessionLocal", return_value=mock_db),
        patch("workflows.phase1_ingestion.settings") as mock_settings,
    ):
        mock_settings.AUTO_ACCEPT_CONTENT = False
        mock_settings.MAX_CARDS_PER_TOPIC_TYPE = topic_limit

        result = phase1_ingestion.node_generate(state)

    return result["current_new_cards"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cards_capped_when_limit_already_reached():
    """If the topic already has limit cards, no new cards should be saved."""
    drafts = [_make_draft(f"Q{i}") for i in range(3)]
    new_cards = _run_node_generate(drafts, cards_already_in_db=5, topic_limit=5)
    assert new_cards == [], (
        "No cards should be saved when existing cards already equal the limit. "
        f"Got {len(new_cards)} card(s) saved."
    )


def test_cards_capped_partway_through_drafts():
    """If 3 cards exist and limit is 5, only 2 of 4 drafts should be saved."""
    drafts = [_make_draft(f"Q{i}") for i in range(4)]
    new_cards = _run_node_generate(drafts, cards_already_in_db=3, topic_limit=5)
    assert len(new_cards) == 2, (
        f"Expected 2 cards saved (3 existing + 2 = 5 = limit), got {len(new_cards)}. "
        "The per-topic card cap is not being enforced at the card level."
    )


def test_all_drafts_saved_when_under_limit():
    """If 0 cards exist and limit is 5, all 3 drafts should be saved."""
    drafts = [_make_draft(f"Q{i}") for i in range(3)]
    new_cards = _run_node_generate(drafts, cards_already_in_db=0, topic_limit=5)
    assert len(new_cards) == 3, (
        f"Expected all 3 cards saved (0 existing, limit=5), got {len(new_cards)}."
    )


def test_limit_zero_saves_no_cards():
    """Edge case: limit=1 means at most 1 card total; if 1 already exists, save 0."""
    drafts = [_make_draft("Q1"), _make_draft("Q2")]
    new_cards = _run_node_generate(drafts, cards_already_in_db=1, topic_limit=1)
    assert new_cards == [], (
        f"Limit=1 with 1 existing card should save 0 new cards, got {len(new_cards)}."
    )


def test_multiple_topics_capped_independently():
    """
    Cards for topic A hitting the limit must not block cards for topic B.
    This exercises the DB query being per-(subject, topic, question_type).
    """
    from workflows import phase1_ingestion

    chunks = [
        _make_chunk(topic_id=10, subtopic_id=20, chunk_id=1),
        _make_chunk(topic_id=11, subtopic_id=21, chunk_id=2),
    ]

    drafts_per_chunk = [_make_draft(f"Q{i}") for i in range(3)]
    mock_socratic = MagicMock()
    mock_socratic.generate_flashcard.return_value = drafts_per_chunk

    saved_counter = [0]
    def fake_create(**kwargs):
        saved_counter[0] += 1
        return {
            "id": saved_counter[0],
            "subject_id": kwargs["subject_id"],
            "subtopic_id": kwargs["subtopic_id"],
            "status": kwargs["status"],
            "question": kwargs["question"],
            "answer": kwargs["answer"],
            "question_type": kwargs["question_type"],
        }

    mock_fc_repo = MagicMock()
    mock_fc_repo.create.side_effect = fake_create

    # topic 10 already has 5 cards (at limit); topic 11 has 0
    def db_scalar_for_topic(*args, **kwargs):
        # We can't easily inspect which topic is being queried via the mock chain,
        # so just return the counts in sequence: first call for topic 10 = 5, second = 0
        db_scalar_for_topic.call_count = getattr(db_scalar_for_topic, "call_count", 0) + 1
        return 5 if db_scalar_for_topic.call_count == 1 else 0

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.scalar.side_effect = db_scalar_for_topic

    saved_topic10 = []
    saved_topic11 = []

    with (
        patch.object(phase1_ingestion, "_socratic_agent", mock_socratic),
        patch("workflows.phase1_ingestion.FlashcardRepo", return_value=mock_fc_repo),
        patch("workflows.phase1_ingestion.SessionLocal", return_value=mock_db),
        patch("workflows.phase1_ingestion.settings") as mock_settings,
    ):
        mock_settings.AUTO_ACCEPT_CONTENT = False
        mock_settings.MAX_CARDS_PER_TOPIC_TYPE = 5

        # Process chunk for topic 10 (at limit)
        state_t10 = _make_state([chunks[0]])
        r10 = phase1_ingestion.node_generate(state_t10)
        saved_topic10 = r10["current_new_cards"]

        # Process chunk for topic 11 (under limit)
        state_t11 = _make_state([chunks[1]])
        r11 = phase1_ingestion.node_generate(state_t11)
        saved_topic11 = r11["current_new_cards"]

    assert saved_topic10 == [], (
        f"topic 10 is at limit (5 existing) — 0 cards should be saved, got {len(saved_topic10)}"
    )
    assert len(saved_topic11) == 3, (
        f"topic 11 is under limit (0 existing, limit=5) — all 3 drafts should be saved, "
        f"got {len(saved_topic11)}"
    )
