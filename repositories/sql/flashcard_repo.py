"""
repositories/sql/flashcard_repo.py
------------------------------------
SQLAlchemy implementation of FlashcardRepoProtocol.
Consolidates all flashcard persistence logic that was scattered across
agents/socratic.py, agents/critic.py, and app.py.
"""

import json
import logging
from typing import Dict, List, Optional
from core.database import SessionLocal, Flashcard, SubjectTopicAssociation, Subtopic

logger = logging.getLogger(__name__)


def _fc_to_dict(fc: Flashcard) -> dict:
    return {
        "id": fc.id,
        "subject_id": fc.subject_id,
        "subtopic_id": fc.subtopic_id,
        "chunk_id": fc.chunk_id,
        "question": fc.question,
        "answer": fc.answer,
        "question_type": fc.question_type,
        "complexity_level": fc.complexity_level,
        "rubric": fc.rubric,
        "critic_rubric_scores": fc.critic_rubric_scores,
        "critic_score": fc.critic_score,
        "critic_feedback": fc.critic_feedback,
        "status": fc.status,
        "mentor_feedback": fc.mentor_feedback,
        "created_at": fc.created_at,
    }


class FlashcardRepo:
    """Concrete SQL implementation of FlashcardRepoProtocol."""

    def create(
        self,
        subject_id: int,
        subtopic_id: Optional[int],
        chunk_id: Optional[int],
        question: str,
        answer: str,
        question_type: str,
        rubric_json: str,
        status: str = "pending",
        topic_id: Optional[int] = None,
    ) -> dict:
        with SessionLocal() as db:
            # Validate subject→topic→subtopic chain before inserting.
            if topic_id and subject_id:
                assoc = db.query(SubjectTopicAssociation).filter_by(
                    subject_id=subject_id, topic_id=topic_id
                ).first()
                if not assoc:
                    logger.error(
                        "CHAIN VIOLATION: subject_id=%s topic_id=%s not in SubjectTopicAssociation — "
                        "flashcard will still be saved but subject→topic link is missing",
                        subject_id, topic_id,
                    )
            if subtopic_id and topic_id:
                sub = db.query(Subtopic).filter_by(id=subtopic_id).first()
                if sub and sub.topic_id != topic_id:
                    logger.error(
                        "CHAIN VIOLATION: subtopic_id=%s belongs to topic_id=%s, not topic_id=%s",
                        subtopic_id, sub.topic_id, topic_id,
                    )

            fc = Flashcard(
                subject_id=subject_id,
                topic_id=topic_id,
                subtopic_id=subtopic_id,
                chunk_id=chunk_id,
                question=question,
                answer=answer,
                question_type=question_type,
                rubric=rubric_json,
                complexity_level=None,
                status=status,
            )
            db.add(fc)
            db.commit()
            db.refresh(fc)
            return _fc_to_dict(fc)

    def get_by_subject(
        self,
        subject_id: int,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
        question_type: Optional[str] = None,
    ) -> List[dict]:
        with SessionLocal() as db:
            q = db.query(Flashcard).filter(Flashcard.subject_id == subject_id)
            if status is not None:
                q = q.filter(Flashcard.status == status)
            if question_type is not None:
                q = q.filter(Flashcard.question_type == question_type)
            q = q.offset(skip).limit(limit)
            return [_fc_to_dict(fc) for fc in q.all()]

    def get_by_id(self, flashcard_id: int) -> Optional[dict]:
        with SessionLocal() as db:
            fc = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
            return _fc_to_dict(fc) if fc else None

    def update_status(
        self,
        flashcard_id: int,
        status: str,
        feedback: str = "",
        complexity_level: Optional[str] = None,
    ) -> None:
        with SessionLocal() as db:
            fc = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
            if fc:
                fc.status = status
                if feedback:
                    fc.mentor_feedback = feedback
                if complexity_level is not None:
                    fc.complexity_level = complexity_level
                db.commit()

    def update_complexity(self, flashcard_id: int, complexity_level: Optional[str]) -> None:
        with SessionLocal() as db:
            fc = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
            if fc:
                fc.complexity_level = complexity_level
                db.commit()

    def bulk_update_status(self, flashcard_ids: List[int], status: str) -> int:
        """Update status for multiple flashcards in one query. Returns count updated."""
        if not flashcard_ids:
            return 0
        with SessionLocal() as db:
            count = db.query(Flashcard).filter(
                Flashcard.id.in_(flashcard_ids)
            ).update({"status": status}, synchronize_session=False)
            db.commit()
            return count

    def update_critic_scores(
        self,
        flashcard_id: int,
        aggregate_score: int,
        rubric_scores_json: str,
        feedback: str,
        complexity_level: str,
    ) -> None:
        """Write critic evaluation results back to the flashcard record."""
        with SessionLocal() as db:
            fc = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
            if fc:
                fc.critic_score = aggregate_score
                fc.critic_feedback = feedback
                fc.critic_rubric_scores = rubric_scores_json
                fc.complexity_level = complexity_level
                db.commit()

    def has_active_cards_for_subtopic(self, subject_id: int, subtopic_id: int) -> bool:
        """True if subject already has approved or pending cards for this subtopic.

        H22: only skip subtopics with approved/pending cards — if mentor rejected
        all cards for a subtopic, allow re-generation so gaps can be filled.
        """
        with SessionLocal() as db:
            count = db.query(Flashcard).filter(
                Flashcard.subject_id == subject_id,
                Flashcard.subtopic_id == subtopic_id,
                Flashcard.status.in_(["approved", "pending"]),
            ).count()
            return count > 0

    def get_pending_ids_for_subtopics(self, subtopic_ids: List[int]) -> List[int]:
        """Return IDs of all pending flashcards for the given subtopics."""
        if not subtopic_ids:
            return []
        with SessionLocal() as db:
            rows = db.query(Flashcard.id).filter(
                Flashcard.subtopic_id.in_(subtopic_ids),
                Flashcard.status == "pending",
            ).all()
            return [r[0] for r in rows]

    def delete_by_subject(self, subject_id: int) -> int:
        """Delete all flashcards for a subject. Returns count deleted."""
        with SessionLocal() as db:
            count = db.query(Flashcard).filter(
                Flashcard.subject_id == subject_id
            ).delete(synchronize_session=False)
            db.commit()
            return count

    def delete(self, flashcard_id: int) -> None:
        with SessionLocal() as db:
            fc = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
            if fc:
                db.delete(fc)
                db.commit()

    def update_content(
        self,
        flashcard_id: int,
        question: str,
        answer: str,
        question_type: str,
        rubric_json: str,
        mentor_feedback: str,
    ) -> Optional[dict]:
        """Update flashcard content and reset to pending (used by recreate flow)."""
        with SessionLocal() as db:
            fc = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
            if not fc:
                return None
            fc.question = question
            fc.answer = answer
            fc.question_type = question_type
            fc.rubric = rubric_json
            fc.mentor_feedback = mentor_feedback
            fc.status = "pending"
            db.commit()
            db.refresh(fc)
            return _fc_to_dict(fc)

    def get_all_by_status(self, status: str) -> List[dict]:
        with SessionLocal() as db:
            fcs = db.query(Flashcard).filter(Flashcard.status == status).order_by(
                Flashcard.created_at.desc()
            ).all()
            return [_fc_to_dict(fc) for fc in fcs]

    def get_global_stats(self) -> dict:
        """Returns global {total, approved, pending, rejected} counts."""
        from sqlalchemy import func
        with SessionLocal() as db:
            total    = db.query(func.count(Flashcard.id)).scalar() or 0
            approved = db.query(func.count(Flashcard.id)).filter(Flashcard.status == "approved").scalar() or 0
            pending  = db.query(func.count(Flashcard.id)).filter(Flashcard.status == "pending").scalar() or 0
            rejected = db.query(func.count(Flashcard.id)).filter(Flashcard.status == "rejected").scalar() or 0
        return {"total": total, "approved": approved, "pending": pending, "rejected": rejected}

    def get_by_subtopic(
        self,
        subtopic_id: int,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
        question_type: Optional[str] = None,
    ) -> List[dict]:
        with SessionLocal() as db:
            q = db.query(Flashcard).filter(Flashcard.subtopic_id == subtopic_id)
            if status is not None:
                q = q.filter(Flashcard.status == status)
            if question_type is not None:
                q = q.filter(Flashcard.question_type == question_type)
            q = q.offset(skip).limit(limit)
            return [_fc_to_dict(fc) for fc in q.all()]

    def get_sources_by_chunk_ids(self, chunk_ids: List[int]) -> Dict[int, dict]:
        """Single query: fetch source attribution for multiple chunk IDs.

        Returns {chunk_id: source_dict}.
        """
        if not chunk_ids:
            return {}
        from core.database import ContentChunk, Document as DBDocument
        with SessionLocal() as db:
            chunks = db.query(ContentChunk).filter(ContentChunk.id.in_(chunk_ids)).all()
            doc_ids = list({c.document_id for c in chunks if c.document_id})
            docs_by_id = {}
            if doc_ids:
                docs = db.query(DBDocument).filter(DBDocument.id.in_(doc_ids)).all()
                docs_by_id = {d.id: d for d in docs}
            result: Dict[int, dict] = {}
            for chunk in chunks:
                doc = docs_by_id.get(chunk.document_id)
                result[chunk.id] = {
                    "source_type": chunk.source_type,
                    "source_url": getattr(chunk, "source_url", None),
                    "filename": doc.filename if doc else None,
                    "document_id": chunk.document_id,
                    "page_number": getattr(chunk, "page_number", None),
                    "text": chunk.text,
                }
            return result

    def get_source_by_chunk(self, chunk_id: int) -> Optional[dict]:
        """Return source attribution info for a content chunk."""
        from core.database import ContentChunk, Document as DBDocument
        with SessionLocal() as db:
            chunk = db.query(ContentChunk).filter(ContentChunk.id == chunk_id).first()
            if not chunk:
                return None
            doc = None
            if chunk.document_id:
                doc = db.query(DBDocument).filter(DBDocument.id == chunk.document_id).first()
            return {
                "source_type": chunk.source_type,
                "source_url": getattr(chunk, "source_url", None),
                "filename": doc.filename if doc else None,
                "document_id": chunk.document_id,
                "page_number": getattr(chunk, "page_number", None),
                "text": chunk.text,
            }
