"""
tests/unit/services/test_flashcard_service.py
-----------------------------------------------
Unit tests for FlashcardService with mocked repos.
Covers bulk approve/reject logic for subtopics.
"""

import pytest
from unittest.mock import MagicMock

from services.flashcard_service import FlashcardService


@pytest.fixture
def flashcard_repo():
    return MagicMock()


@pytest.fixture
def chunk_repo():
    return MagicMock()


@pytest.fixture
def svc(flashcard_repo, chunk_repo):
    return FlashcardService(flashcard_repo, chunk_repo)


class TestFlashcardServiceBulkApproveSubtopics:
    def test_approve_calls_bulk_update(self, svc, flashcard_repo):
        flashcard_repo.get_pending_ids_for_subtopics.return_value = [10, 11, 12]
        flashcard_repo.bulk_update_status.return_value = 3

        count = svc.bulk_approve_subtopics([1, 2])

        flashcard_repo.get_pending_ids_for_subtopics.assert_called_once_with([1, 2])
        flashcard_repo.bulk_update_status.assert_called_once_with([10, 11, 12], "approved")
        assert count == 3

    def test_approve_empty_subtopic_list_returns_zero(self, svc, flashcard_repo):
        count = svc.bulk_approve_subtopics([])
        flashcard_repo.get_pending_ids_for_subtopics.assert_not_called()
        assert count == 0

    def test_approve_when_no_pending_returns_zero(self, svc, flashcard_repo):
        flashcard_repo.get_pending_ids_for_subtopics.return_value = []
        count = svc.bulk_approve_subtopics([1, 2])
        flashcard_repo.bulk_update_status.assert_not_called()
        assert count == 0


class TestFlashcardServiceBulkRejectSubtopics:
    def test_reject_calls_bulk_update(self, svc, flashcard_repo):
        flashcard_repo.get_pending_ids_for_subtopics.return_value = [5, 6]
        svc.bulk_reject_subtopics([3])
        flashcard_repo.bulk_update_status.assert_called_once_with([5, 6], "rejected")

    def test_reject_empty_subtopic_list_returns_zero(self, svc, flashcard_repo):
        count = svc.bulk_reject_subtopics([])
        assert count == 0


class TestFlashcardServiceUpdateStatus:
    def test_update_status_delegates(self, svc, flashcard_repo):
        svc.update_status(99, "approved", feedback="looks good", complexity_level="medium")
        flashcard_repo.update_status.assert_called_once_with(
            99, "approved", "looks good", "medium"
        )

    def test_bulk_update_delegates(self, svc, flashcard_repo):
        flashcard_repo.bulk_update_status.return_value = 2
        result = svc.bulk_update_status([1, 2], "rejected")
        flashcard_repo.bulk_update_status.assert_called_once_with([1, 2], "rejected")
        assert result == 2
