"""
services/flashcard_service.py
-------------------------------
Business logic for flashcard lifecycle operations.
Absorbs inline .filter().update() bulk-approve/reject calls from the Mentor Review UI.
"""

import logging
from typing import List, Optional
from repositories.protocols import FlashcardRepoProtocol, ChunkRepoProtocol

logger = logging.getLogger(__name__)


class FlashcardService:
    def __init__(
        self,
        flashcard_repo: FlashcardRepoProtocol,
        chunk_repo: ChunkRepoProtocol,
    ):
        self._flashcards = flashcard_repo
        self._chunks = chunk_repo

    def get_by_subject(
        self, subject_id: int, status: Optional[str] = None
    ) -> List[dict]:
        return self._flashcards.get_by_subject(subject_id, status)

    def update_status(
        self,
        flashcard_id: int,
        status: str,
        feedback: str = "",
        complexity_level: Optional[str] = None,
    ) -> None:
        self._flashcards.update_status(flashcard_id, status, feedback, complexity_level)

    def bulk_update_status(self, flashcard_ids: List[int], status: str) -> int:
        return self._flashcards.bulk_update_status(flashcard_ids, status)

    def bulk_approve_subtopics(self, subtopic_ids: List[int]) -> int:
        """Approve all pending flashcards for the given subtopics.

        Extracted from the inline `.filter().update()` call in the Mentor Review view.
        Returns the count of cards updated.
        """
        if not subtopic_ids:
            return 0
        pending_ids = self._flashcards.get_pending_ids_for_subtopics(subtopic_ids)
        if not pending_ids:
            return 0
        return self._flashcards.bulk_update_status(pending_ids, "approved")

    def bulk_reject_subtopics(self, subtopic_ids: List[int]) -> int:
        """Reject all pending flashcards for the given subtopics."""
        if not subtopic_ids:
            return 0
        pending_ids = self._flashcards.get_pending_ids_for_subtopics(subtopic_ids)
        if not pending_ids:
            return 0
        return self._flashcards.bulk_update_status(pending_ids, "rejected")

    def has_active_cards_for_subtopic(
        self, subject_id: int, subtopic_id: int
    ) -> bool:
        return self._flashcards.has_active_cards_for_subtopic(subject_id, subtopic_id)

    def get_by_subtopic(self, subtopic_id: int, status: Optional[str] = None) -> List[dict]:
        return self._flashcards.get_by_subtopic(subtopic_id, status)

    def delete_one(self, flashcard_id: int) -> None:
        self._flashcards.delete(flashcard_id)

    def get_all_rejected(self) -> List[dict]:
        return self._flashcards.get_all_by_status("rejected")

    def get_chunk_source(self, chunk_id: int) -> Optional[dict]:
        return self._flashcards.get_source_by_chunk(chunk_id)
