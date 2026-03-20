"""
tests/layer/test_api_flashcards.py
------------------------------------
Layer tests for the /flashcards FastAPI router.
Uses TestClient with mock FlashcardService.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_flashcard_service


@pytest.fixture
def mock_svc():
    return MagicMock()


@pytest.fixture
def client(mock_svc):
    app.dependency_overrides[get_flashcard_service] = lambda: mock_svc
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestGetFlashcardsBySubject:
    def test_returns_200_with_cards(self, client, mock_svc):
        mock_svc.get_by_subject.return_value = [
            {
                "id": 1, "subject_id": 1, "subtopic_id": None, "chunk_id": None,
                "question": "Q?", "answer": "A.", "question_type": "active_recall",
                "complexity_level": None, "rubric": None, "critic_rubric_scores": None,
                "critic_score": None, "critic_feedback": None, "status": "pending",
                "mentor_feedback": None, "created_at": None,
            }
        ]
        r = client.get("/flashcards/subject/1")
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_passes_status_filter(self, client, mock_svc):
        mock_svc.get_by_subject.return_value = []
        r = client.get("/flashcards/subject/1?status=approved")
        assert r.status_code == 200
        mock_svc.get_by_subject.assert_called_once_with(1, "approved", 0, 50, None)


class TestUpdateFlashcardStatus:
    def test_approve_returns_204(self, client, mock_svc):
        r = client.patch(
            "/flashcards/42/status",
            json={"status": "approved", "feedback": "", "complexity_level": "medium"},
        )
        assert r.status_code == 204
        mock_svc.update_status.assert_called_once_with(42, "approved", "", "medium")

    def test_invalid_status_returns_422(self, client, mock_svc):
        r = client.patch("/flashcards/1/status", json={"status": "unknown"})
        assert r.status_code == 422


class TestBulkStatusUpdate:
    def test_bulk_approve_returns_204(self, client, mock_svc):
        r = client.post(
            "/flashcards/bulk-status",
            json={"flashcard_ids": [1, 2, 3], "status": "approved"},
        )
        assert r.status_code == 204
        mock_svc.bulk_update_status.assert_called_once_with([1, 2, 3], "approved")


class TestBulkSubtopicAction:
    def test_approve_subtopics(self, client, mock_svc):
        r = client.post(
            "/flashcards/bulk-subtopic-action",
            json={"subtopic_ids": [10, 11], "action": "approve"},
        )
        assert r.status_code == 204
        mock_svc.bulk_approve_subtopics.assert_called_once_with([10, 11])

    def test_reject_subtopics(self, client, mock_svc):
        r = client.post(
            "/flashcards/bulk-subtopic-action",
            json={"subtopic_ids": [10], "action": "reject"},
        )
        assert r.status_code == 204
        mock_svc.bulk_reject_subtopics.assert_called_once_with([10])

    def test_invalid_action_returns_422(self, client, mock_svc):
        r = client.post(
            "/flashcards/bulk-subtopic-action",
            json={"subtopic_ids": [1], "action": "delete"},
        )
        assert r.status_code == 422
