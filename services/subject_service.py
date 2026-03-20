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
        document_repo=None,   # Optional: DocumentRepo — needed for attach/detach ops
    ):
        self._subjects = subject_repo
        self._flashcards = flashcard_repo
        self._documents = document_repo

    def get_all_active(self) -> List[dict]:
        return self._subjects.get_all_active()

    def get_all_active_with_stats(self) -> List[dict]:
        """Return all active subjects with topic_count, approved, pending, rejected — 3 queries total."""
        subjects = self._subjects.get_all_active()
        if not subjects:
            return []
        subject_ids = [s["id"] for s in subjects]
        topic_counts = self._subjects.get_topic_counts_by_subject_ids(subject_ids)
        fc_stats = self._subjects.get_flashcard_stats_by_subject_ids(subject_ids)
        for s in subjects:
            sid = s["id"]
            s["topic_count"] = topic_counts.get(sid, 0)
            stats = fc_stats.get(sid, {"approved": 0, "pending": 0, "rejected": 0})
            s["approved"] = stats["approved"]
            s["pending"] = stats["pending"]
            s["rejected"] = stats["rejected"]
        return subjects

    def get_by_id(self, subject_id: int) -> Optional[dict]:
        return self._subjects.get_by_id(subject_id)

    def create(self, name: str) -> dict:
        return self._subjects.create(name)

    def archive(self, subject_id: int) -> None:
        self._subjects.archive(subject_id)

    def restore(self, subject_id: int) -> None:
        self._subjects.restore(subject_id)

    def permanent_delete(self, subject_id: int) -> None:
        """Delete a subject and all its data."""
        self._flashcards.delete_by_subject(subject_id)
        self._subjects.delete(subject_id)

    def get_flashcard_stats(self, subject_id: int) -> dict:
        return self._subjects.get_flashcard_stats(subject_id)

    def get_all_archived(self) -> List[dict]:
        return self._subjects.get_all_archived()

    def rename(self, subject_id: int, name: str) -> None:
        self._subjects.rename(subject_id, name)

    def get_global_stats(self) -> dict:
        return self._flashcards.get_global_stats()

    def get_attached_documents(self, subject_id: int) -> List[dict]:
        if self._documents is None:
            return []
        return self._documents.get_attached_to_subject(subject_id)

    def attach_document(self, subject_id: int, doc_id: str) -> None:
        if self._documents is not None:
            self._documents.attach_to_subject(doc_id, subject_id)

    def detach_document(self, subject_id: int, doc_id: str) -> None:
        if self._documents is not None:
            self._documents.detach_from_subject(doc_id, subject_id)

    def get_available_documents(self, subject_id: int) -> List[dict]:
        if self._documents is None:
            return []
        return self._documents.get_not_attached_to_subject(subject_id)
