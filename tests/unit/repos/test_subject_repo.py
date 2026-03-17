"""
tests/unit/repos/test_subject_repo.py
---------------------------------------
Unit tests for SubjectRepo using in-memory SQLite.
Tests archive/restore/delete operations and get_flashcard_stats.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

from core.database import Base, Subject, Flashcard


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
def subject_repo(session_factory):
    """Return a SubjectRepo wired to in-memory SQLite."""
    from repositories.sql.subject_repo import SubjectRepo
    repo = SubjectRepo()
    # Patch SessionLocal inside the repo module to use in-memory DB
    with patch("repositories.sql.subject_repo.SessionLocal", session_factory):
        yield repo


def _create_subject(session_factory, name="Test Subject", is_archived=False):
    with session_factory() as db:
        subj = Subject(name=name, is_archived=is_archived)
        db.add(subj)
        db.commit()
        db.refresh(subj)
        return subj.id


class TestSubjectRepoCreate:
    def test_create_returns_dict_with_id(self, subject_repo):
        result = subject_repo.create("Data Science")
        assert result["id"] is not None
        assert result["name"] == "Data Science"
        assert result["is_archived"] is False

    def test_create_persists_to_db(self, subject_repo, session_factory):
        subject_repo.create("Machine Learning")
        with session_factory() as db:
            subj = db.query(Subject).filter(Subject.name == "Machine Learning").first()
        assert subj is not None


class TestSubjectRepoGetAllActive:
    def test_returns_only_active_subjects(self, subject_repo, session_factory):
        _create_subject(session_factory, "Active Subject", is_archived=False)
        _create_subject(session_factory, "Archived Subject", is_archived=True)

        result = subject_repo.get_all_active()
        names = [r["name"] for r in result]
        assert "Active Subject" in names
        assert "Archived Subject" not in names

    def test_empty_when_no_subjects(self, subject_repo):
        result = subject_repo.get_all_active()
        assert result == []


class TestSubjectRepoArchiveRestore:
    def test_archive_sets_is_archived_true(self, subject_repo, session_factory):
        subj_id = _create_subject(session_factory, "To Archive")
        subject_repo.archive(subj_id)
        with session_factory() as db:
            subj = db.get(Subject, subj_id)
        assert subj.is_archived is True

    def test_restore_sets_is_archived_false(self, subject_repo, session_factory):
        subj_id = _create_subject(session_factory, "To Restore", is_archived=True)
        subject_repo.restore(subj_id)
        with session_factory() as db:
            subj = db.get(Subject, subj_id)
        assert subj.is_archived is False

    def test_archive_nonexistent_subject_is_noop(self, subject_repo):
        # Should not raise
        subject_repo.archive(99999)

    def test_restore_nonexistent_subject_is_noop(self, subject_repo):
        subject_repo.restore(99999)


class TestSubjectRepoDelete:
    def test_delete_removes_subject(self, subject_repo, session_factory):
        subj_id = _create_subject(session_factory, "To Delete")
        subject_repo.delete(subj_id)
        with session_factory() as db:
            subj = db.get(Subject, subj_id)
        assert subj is None

    def test_delete_nonexistent_is_noop(self, subject_repo):
        subject_repo.delete(99999)


class TestSubjectRepoGetFlashcardStats:
    def test_returns_zero_stats_when_no_flashcards(self, subject_repo, session_factory):
        subj_id = _create_subject(session_factory, "Empty Subject")
        stats = subject_repo.get_flashcard_stats(subj_id)
        assert stats == {"approved": 0, "pending": 0, "rejected": 0}

    def test_counts_flashcards_by_status(self, subject_repo, session_factory):
        subj_id = _create_subject(session_factory, "Stats Subject")
        with session_factory() as db:
            db.add(Flashcard(subject_id=subj_id, question="Q1", answer="A1", status="approved"))
            db.add(Flashcard(subject_id=subj_id, question="Q2", answer="A2", status="approved"))
            db.add(Flashcard(subject_id=subj_id, question="Q3", answer="A3", status="pending"))
            db.add(Flashcard(subject_id=subj_id, question="Q4", answer="A4", status="rejected"))
            db.commit()

        stats = subject_repo.get_flashcard_stats(subj_id)
        assert stats["approved"] == 2
        assert stats["pending"] == 1
        assert stats["rejected"] == 1
