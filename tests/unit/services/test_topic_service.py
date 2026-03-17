"""
tests/unit/services/test_topic_service.py
-------------------------------------------
Unit tests for TopicService — focuses on the H6 cascade-delete invariant
(approved flashcards are preserved when a topic is deleted).
"""

import pytest
from unittest.mock import MagicMock

from services.topic_service import TopicService


@pytest.fixture
def topic_repo():
    return MagicMock()


@pytest.fixture
def chunk_repo():
    return MagicMock()


@pytest.fixture
def vector_store():
    return MagicMock()


@pytest.fixture
def svc(topic_repo, chunk_repo, vector_store):
    return TopicService(topic_repo, chunk_repo, vector_store)


class TestTopicServiceDeleteCascade:
    def test_cascade_calls_repo_then_vector(self, svc, topic_repo, vector_store):
        topic_repo.delete_topic_cascade.return_value = 3  # 3 approved cards preserved

        preserved = svc.delete_topic_cascade(topic_id=10, doc_id="doc-abc")

        topic_repo.delete_topic_cascade.assert_called_once_with(10, "doc-abc")
        vector_store.delete_by_document.assert_called_once_with("doc-abc")
        assert preserved == 3

    def test_cascade_returns_preserved_count(self, svc, topic_repo, vector_store):
        topic_repo.delete_topic_cascade.return_value = 0
        preserved = svc.delete_topic_cascade(1, "doc-xyz")
        assert preserved == 0

    def test_qdrant_failure_does_not_raise(self, svc, topic_repo, vector_store):
        """Qdrant cleanup failure is logged but does not bubble up."""
        topic_repo.delete_topic_cascade.return_value = 2
        vector_store.delete_by_document.side_effect = Exception("Connection refused")

        # Should not raise
        preserved = svc.delete_topic_cascade(5, "doc-1")
        assert preserved == 2

    def test_repo_failure_does_raise(self, svc, topic_repo, vector_store):
        """DB cascade failure must propagate so callers know the delete failed."""
        topic_repo.delete_topic_cascade.side_effect = Exception("DB error")

        with pytest.raises(Exception, match="DB error"):
            svc.delete_topic_cascade(5, "doc-1")

        # Qdrant should NOT be called if repo failed
        vector_store.delete_by_document.assert_not_called()
