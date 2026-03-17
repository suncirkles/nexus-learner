"""
tests/unit/repos/test_flashcard_repo.py
-----------------------------------------
Unit tests for FlashcardRepo using in-memory SQLite.
Tests bulk_update_status, has_active_cards_for_subtopic, create.
"""

import json
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

from core.database import Base, Subject, Flashcard, Subtopic, Topic


@pytest.fixture
def in_memory_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def session_factory(in_memory_engine):
    return sessionmaker(bind=in_memory_engine)


@pytest.fixture
def flashcard_repo(session_factory):
    from repositories.sql.flashcard_repo import FlashcardRepo
    repo = FlashcardRepo()
    with patch("repositories.sql.flashcard_repo.SessionLocal", session_factory):
        yield repo


@pytest.fixture
def subject_id(session_factory):
    with session_factory() as db:
        subj = Subject(name="Test Subject")
        db.add(subj)
        db.commit()
        db.refresh(subj)
        return subj.id


@pytest.fixture
def subtopic_id(session_factory, subject_id):
    with session_factory() as db:
        topic = Topic(document_id="doc-1", name="Test Topic")
        db.add(topic)
        db.commit()
        db.refresh(topic)
        sub = Subtopic(topic_id=topic.id, name="Test Subtopic")
        db.add(sub)
        db.commit()
        db.refresh(sub)
        return sub.id


def _make_flashcard(session_factory, subject_id, subtopic_id=None, status="pending"):
    with session_factory() as db:
        fc = Flashcard(
            subject_id=subject_id,
            subtopic_id=subtopic_id,
            question="Q?",
            answer="A.",
            status=status,
        )
        db.add(fc)
        db.commit()
        db.refresh(fc)
        return fc.id


class TestFlashcardRepoCreate:
    def test_create_returns_dict_with_id(self, flashcard_repo, subject_id, subtopic_id):
        result = flashcard_repo.create(
            subject_id=subject_id,
            subtopic_id=subtopic_id,
            chunk_id=None,
            question="What is X?",
            answer="X is Y.",
            question_type="active_recall",
            rubric_json=json.dumps([{"criterion": "c1", "description": "d1"}]),
            status="pending",
        )
        assert result["id"] is not None
        assert result["question"] == "What is X?"
        assert result["status"] == "pending"

    def test_create_with_approved_status(self, flashcard_repo, subject_id):
        result = flashcard_repo.create(
            subject_id=subject_id,
            subtopic_id=None,
            chunk_id=None,
            question="Q",
            answer="A",
            question_type="fill_blank",
            rubric_json="[]",
            status="approved",
        )
        assert result["status"] == "approved"


class TestFlashcardRepoBulkUpdateStatus:
    def test_bulk_update_changes_status(self, flashcard_repo, session_factory, subject_id):
        id1 = _make_flashcard(session_factory, subject_id, status="pending")
        id2 = _make_flashcard(session_factory, subject_id, status="pending")

        count = flashcard_repo.bulk_update_status([id1, id2], "approved")
        assert count == 2

        with session_factory() as db:
            fc1 = db.get(Flashcard, id1)
            fc2 = db.get(Flashcard, id2)
        assert fc1.status == "approved"
        assert fc2.status == "approved"

    def test_bulk_update_empty_list_returns_zero(self, flashcard_repo):
        count = flashcard_repo.bulk_update_status([], "approved")
        assert count == 0

    def test_bulk_update_does_not_affect_other_cards(self, flashcard_repo, session_factory, subject_id):
        id1 = _make_flashcard(session_factory, subject_id, status="pending")
        id2 = _make_flashcard(session_factory, subject_id, status="pending")

        flashcard_repo.bulk_update_status([id1], "approved")

        with session_factory() as db:
            fc2 = db.get(Flashcard, id2)
        assert fc2.status == "pending"


class TestFlashcardRepoHasActiveCards:
    def test_returns_true_when_approved_card_exists(self, flashcard_repo, session_factory, subject_id, subtopic_id):
        _make_flashcard(session_factory, subject_id, subtopic_id=subtopic_id, status="approved")
        result = flashcard_repo.has_active_cards_for_subtopic(subject_id, subtopic_id)
        assert result is True

    def test_returns_true_when_pending_card_exists(self, flashcard_repo, session_factory, subject_id, subtopic_id):
        _make_flashcard(session_factory, subject_id, subtopic_id=subtopic_id, status="pending")
        result = flashcard_repo.has_active_cards_for_subtopic(subject_id, subtopic_id)
        assert result is True

    def test_returns_false_when_only_rejected_cards(self, flashcard_repo, session_factory, subject_id, subtopic_id):
        _make_flashcard(session_factory, subject_id, subtopic_id=subtopic_id, status="rejected")
        result = flashcard_repo.has_active_cards_for_subtopic(subject_id, subtopic_id)
        assert result is False

    def test_returns_false_when_no_cards(self, flashcard_repo, subject_id, subtopic_id):
        result = flashcard_repo.has_active_cards_for_subtopic(subject_id, subtopic_id)
        assert result is False

    def test_returns_false_for_different_subject(self, flashcard_repo, session_factory, subject_id, subtopic_id):
        # Card exists for subject_id but not for a different subject
        _make_flashcard(session_factory, subject_id, subtopic_id=subtopic_id, status="approved")
        result = flashcard_repo.has_active_cards_for_subtopic(subject_id + 999, subtopic_id)
        assert result is False


class TestFlashcardRepoUpdateCriticScores:
    def test_update_stores_rubric_scores(self, flashcard_repo, session_factory, subject_id):
        fc_id = _make_flashcard(session_factory, subject_id, status="pending")
        rubric_scores = json.dumps({"accuracy": 4, "logic": 3, "grounding": 4, "clarity": 3})

        flashcard_repo.update_critic_scores(
            flashcard_id=fc_id,
            aggregate_score=4,
            rubric_scores_json=rubric_scores,
            feedback="Well grounded.",
            complexity_level="medium",
        )

        with session_factory() as db:
            fc = db.get(Flashcard, fc_id)
        assert fc.critic_score == 4
        assert fc.complexity_level == "medium"
        assert json.loads(fc.critic_rubric_scores)["accuracy"] == 4
