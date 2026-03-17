"""
tests/unit/services/test_subject_service.py
---------------------------------------------
Unit tests for SubjectService with mocked repos.
Verifies delegation and permanent_delete orchestration logic.
"""

import pytest
from unittest.mock import MagicMock, call

from services.subject_service import SubjectService


@pytest.fixture
def subject_repo():
    return MagicMock()


@pytest.fixture
def flashcard_repo():
    return MagicMock()


@pytest.fixture
def svc(subject_repo, flashcard_repo):
    return SubjectService(subject_repo, flashcard_repo)


class TestSubjectServiceCreate:
    def test_create_delegates_to_repo(self, svc, subject_repo):
        subject_repo.create.return_value = {"id": 1, "name": "ML", "is_archived": False}
        result = svc.create("ML")
        subject_repo.create.assert_called_once_with("ML")
        assert result["name"] == "ML"

    def test_get_all_active_delegates(self, svc, subject_repo):
        subject_repo.get_all_active.return_value = [{"id": 1, "name": "ML"}]
        result = svc.get_all_active()
        subject_repo.get_all_active.assert_called_once()
        assert len(result) == 1


class TestSubjectServiceArchiveRestore:
    def test_archive_delegates(self, svc, subject_repo):
        svc.archive(42)
        subject_repo.archive.assert_called_once_with(42)

    def test_restore_delegates(self, svc, subject_repo):
        svc.restore(42)
        subject_repo.restore.assert_called_once_with(42)


class TestSubjectServicePermanentDelete:
    def test_permanent_delete_deletes_flashcards_then_subject(
        self, svc, subject_repo, flashcard_repo
    ):
        svc.permanent_delete(7)

        # flashcards must be deleted before the subject
        assert flashcard_repo.delete_by_subject.call_args == call(7)
        assert subject_repo.delete.call_args == call(7)

    def test_permanent_delete_order_is_flashcards_first(
        self, svc, subject_repo, flashcard_repo
    ):
        call_order = []
        flashcard_repo.delete_by_subject.side_effect = lambda _: call_order.append("flashcards")
        subject_repo.delete.side_effect = lambda _: call_order.append("subject")

        svc.permanent_delete(1)
        assert call_order == ["flashcards", "subject"]


class TestSubjectServiceStats:
    def test_get_flashcard_stats_delegates(self, svc, subject_repo):
        subject_repo.get_flashcard_stats.return_value = {"approved": 5, "pending": 2, "rejected": 1}
        result = svc.get_flashcard_stats(3)
        subject_repo.get_flashcard_stats.assert_called_once_with(3)
        assert result["approved"] == 5
