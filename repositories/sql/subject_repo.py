"""
repositories/sql/subject_repo.py
----------------------------------
SQLAlchemy implementation of SubjectRepoProtocol.
All queries that were previously scattered across app.py are collected here.
"""

from typing import List, Optional
from sqlalchemy import func, case, Integer
from core.database import SessionLocal, Subject, Flashcard


def _subject_to_dict(subj: Subject) -> dict:
    return {
        "id": subj.id,
        "name": subj.name,
        "is_archived": subj.is_archived,
        "created_at": subj.created_at,
    }


class SubjectRepo:
    """Concrete SQL implementation of SubjectRepoProtocol."""

    def get_all_active(self) -> List[dict]:
        with SessionLocal() as db:
            subjects = db.query(Subject).filter(Subject.is_archived == False).all()
            return [_subject_to_dict(s) for s in subjects]

    def get_by_id(self, subject_id: int) -> Optional[dict]:
        with SessionLocal() as db:
            subj = db.query(Subject).filter(Subject.id == subject_id).first()
            return _subject_to_dict(subj) if subj else None

    def create(self, name: str) -> dict:
        with SessionLocal() as db:
            subj = Subject(name=name)
            db.add(subj)
            db.commit()
            db.refresh(subj)
            return _subject_to_dict(subj)

    def archive(self, subject_id: int) -> None:
        with SessionLocal() as db:
            subj = db.query(Subject).filter(Subject.id == subject_id).first()
            if subj:
                subj.is_archived = True
                db.commit()

    def restore(self, subject_id: int) -> None:
        with SessionLocal() as db:
            subj = db.query(Subject).filter(Subject.id == subject_id).first()
            if subj:
                subj.is_archived = False
                db.commit()

    def delete(self, subject_id: int) -> None:
        """Permanently delete a subject and all its flashcards (cascade via ORM)."""
        with SessionLocal() as db:
            subj = db.query(Subject).filter(Subject.id == subject_id).first()
            if subj:
                db.delete(subj)
                db.commit()

    def get_flashcard_stats(self, subject_id: int) -> dict:
        """Returns {approved, pending, rejected} counts for a subject."""
        with SessionLocal() as db:
            rows = db.query(
                func.sum(case((Flashcard.status == "approved", 1), else_=0)).cast(Integer),
                func.sum(case((Flashcard.status == "pending", 1), else_=0)).cast(Integer),
                func.sum(case((Flashcard.status == "rejected", 1), else_=0)).cast(Integer),
            ).filter(Flashcard.subject_id == subject_id).one()
            return {
                "approved": rows[0] or 0,
                "pending": rows[1] or 0,
                "rejected": rows[2] or 0,
            }

    def get_all_archived(self) -> List[dict]:
        with SessionLocal() as db:
            subjects = db.query(Subject).filter(Subject.is_archived == True).all()
            return [_subject_to_dict(s) for s in subjects]

    def rename(self, subject_id: int, name: str) -> None:
        with SessionLocal() as db:
            subj = db.query(Subject).filter(Subject.id == subject_id).first()
            if subj:
                subj.name = name
                db.commit()
