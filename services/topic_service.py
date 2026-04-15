"""
services/topic_service.py
---------------------------
Business logic for topic/subtopic lifecycle.
Hosts delete_topic_cascade — the H6 invariant (preserve approved cards).
Previously this was app.py::delete_topic_data().
"""

import logging
from typing import List
from repositories.protocols import TopicRepoProtocol, ChunkRepoProtocol, VectorStoreProtocol

logger = logging.getLogger(__name__)


class TopicService:
    def __init__(
        self,
        topic_repo: TopicRepoProtocol,
        chunk_repo: ChunkRepoProtocol,
        vector_store: VectorStoreProtocol,
    ):
        self._topics = topic_repo
        self._chunks = chunk_repo
        self._vector = vector_store

    def get_by_document(self, doc_id: str) -> List[dict]:
        return self._topics.get_by_document(doc_id)

    def delete_topic_cascade(self, topic_id: int, doc_id: str) -> int:
        """Delete a topic and its subtopics/chunks/vectors, preserving approved cards.

        H6 invariant: approved flashcards have their subtopic_id set to NULL rather
        than being deleted — they remain in the subject's study pool.

        Returns the count of approved cards preserved (callers can surface this in the UI).
        """
        preserved = self._topics.delete_topic_cascade(topic_id, doc_id)
        try:
            self._vector.delete_by_document(doc_id)
        except Exception as e:
            logger.warning(
                "Qdrant cleanup failed for doc %s after topic %d deletion: %s",
                doc_id, topic_id, e,
            )
        return preserved

    def get_by_subject(self, subject_id: int) -> List[dict]:
        return self._topics.get_by_subject(subject_id)

    def get_subtopics(self, topic_id: int) -> List[dict]:
        return self._topics.get_subtopics_by_topic(topic_id)

    def get_subtopics_with_counts(self, topic_id: int) -> List[dict]:
        return self._topics.get_subtopics_with_counts(topic_id)

    def get_full_tree_by_subject(self, subject_id: int) -> List[dict]:
        """Return topics with embedded subtopics (incl. card counts) — 2 queries total.

        Card counts are scoped to subject_id so each subject sees only its own cards.
        """
        topics = self._topics.get_by_subject(subject_id)
        if not topics:
            return []
        topic_ids = [t["id"] for t in topics]
        subtopics_by_topic = self._topics.get_subtopics_for_topic_ids(topic_ids, subject_id=subject_id)
        for t in topics:
            t["subtopics"] = subtopics_by_topic.get(t["id"], [])
        return topics
