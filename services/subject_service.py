"""
services/subject_service.py
-----------------------------
Business logic for subject management.
Absorbs permanent_delete multi-step logic from the archived subjects UI panel.
All DB operations delegate to repos — no SessionLocal here.
"""

from typing import List, Optional
from repositories.protocols import SubjectRepoProtocol, FlashcardRepoProtocol


class SubjectService:
    def __init__(
        self,
        subject_repo: SubjectRepoProtocol,
        flashcard_repo: FlashcardRepoProtocol,
    ):
        self._subjects = subject_repo
        self._flashcards = flashcard_repo

    def get_all_active(self) -> List[dict]:
        return self._subjects.get_all_active()

    def get_by_id(self, subject_id: int) -> Optional[dict]:
        return self._subjects.get_by_id(subject_id)

    def create(self, name: str) -> dict:
        return self._subjects.create(name)

    def archive(self, subject_id: int) -> None:
        self._subjects.archive(subject_id)

    def restore(self, subject_id: int) -> None:
        self._subjects.restore(subject_id)

    def permanent_delete(self, subject_id: int) -> None:
        """Delete a subject and all its data.

        Order: flashcards first (subject-specific), then associations, then subject.
        This mirrors the UI panel logic — moved here so it can be reused by the API.
        """
        self._flashcards.delete_by_subject(subject_id)
        self._subjects.delete(subject_id)

    def get_flashcard_stats(self, subject_id: int) -> dict:
        return self._subjects.get_flashcard_stats(subject_id)
