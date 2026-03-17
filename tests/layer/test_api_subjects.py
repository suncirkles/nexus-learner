"""
tests/layer/test_api_subjects.py
----------------------------------
Layer tests for the /subjects FastAPI router.
Uses TestClient with mock SubjectService — no real DB or repos.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_subject_service


@pytest.fixture
def mock_svc():
    return MagicMock()


@pytest.fixture
def client(mock_svc):
    app.dependency_overrides[get_subject_service] = lambda: mock_svc
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestListSubjects:
    def test_returns_200_with_subjects(self, client, mock_svc):
        mock_svc.get_all_active.return_value = [
            {"id": 1, "name": "Data Science", "is_archived": False}
        ]
        r = client.get("/subjects/")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["name"] == "Data Science"

    def test_returns_empty_list_when_no_subjects(self, client, mock_svc):
        mock_svc.get_all_active.return_value = []
        r = client.get("/subjects/")
        assert r.status_code == 200
        assert r.json() == []


class TestCreateSubject:
    def test_returns_201_on_success(self, client, mock_svc):
        mock_svc.create.return_value = {"id": 5, "name": "ML", "is_archived": False}
        r = client.post("/subjects/", json={"name": "ML"})
        assert r.status_code == 201
        assert r.json()["id"] == 5

    def test_returns_422_for_empty_name(self, client, mock_svc):
        r = client.post("/subjects/", json={"name": ""})
        assert r.status_code == 422

    def test_returns_422_for_name_too_long(self, client, mock_svc):
        r = client.post("/subjects/", json={"name": "x" * 101})
        assert r.status_code == 422


class TestGetSubject:
    def test_returns_subject_when_found(self, client, mock_svc):
        mock_svc.get_by_id.return_value = {"id": 3, "name": "Stats", "is_archived": False}
        r = client.get("/subjects/3")
        assert r.status_code == 200
        assert r.json()["name"] == "Stats"

    def test_returns_404_when_not_found(self, client, mock_svc):
        mock_svc.get_by_id.return_value = None
        r = client.get("/subjects/999")
        assert r.status_code == 404


class TestArchiveRestore:
    def test_archive_returns_204(self, client, mock_svc):
        r = client.post("/subjects/1/archive")
        assert r.status_code == 204
        mock_svc.archive.assert_called_once_with(1)

    def test_restore_returns_204(self, client, mock_svc):
        r = client.post("/subjects/1/restore")
        assert r.status_code == 204
        mock_svc.restore.assert_called_once_with(1)


class TestDeleteSubject:
    def test_delete_returns_204(self, client, mock_svc):
        r = client.delete("/subjects/7")
        assert r.status_code == 204
        mock_svc.permanent_delete.assert_called_once_with(7)


class TestFlashcardStats:
    def test_returns_stats(self, client, mock_svc):
        mock_svc.get_flashcard_stats.return_value = {
            "approved": 10, "pending": 3, "rejected": 1
        }
        r = client.get("/subjects/2/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["approved"] == 10
