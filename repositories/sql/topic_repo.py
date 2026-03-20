"""
repositories/sql/topic_repo.py
--------------------------------
SQLAlchemy implementation of TopicRepoProtocol.
Centralises topic/subtopic CRUD and the H6 cascade-delete invariant
(approved flashcards are preserved when a topic is deleted).
"""

import logging
from typing import Dict, List, Optional
from sqlalchemy import delete, func, update as sa_update
from core.database import SessionLocal, Topic, Subtopic, Flashcard, ContentChunk, SubjectDocumentAssociation

logger = logging.getLogger(__name__)


def _topic_to_dict(topic: Topic) -> dict:
    return {
        "id": topic.id,
        "document_id": topic.document_id,
        "name": topic.name,
        "summary": topic.summary,
        "created_at": topic.created_at,
    }


def _subtopic_to_dict(sub: Subtopic) -> dict:
    return {
        "id": sub.id,
        "topic_id": sub.topic_id,
        "name": sub.name,
        "summary": sub.summary,
        "created_at": sub.created_at,
    }


class TopicRepo:
    """Concrete SQL implementation of TopicRepoProtocol."""

    def get_by_document(self, doc_id: str) -> List[dict]:
        with SessionLocal() as db:
            topics = db.query(Topic).filter(Topic.document_id == doc_id).all()
            return [_topic_to_dict(t) for t in topics]

    def get_subtopics_by_topic(self, topic_id: int) -> List[dict]:
        with SessionLocal() as db:
            subs = db.query(Subtopic).filter(Subtopic.topic_id == topic_id).all()
            return [_subtopic_to_dict(s) for s in subs]

    def get_or_create(self, doc_id: str, topic_name: str, summary: str = "") -> dict:
        """Find an existing topic by name (case-insensitive) or create it."""
        with SessionLocal() as db:
            existing = db.query(Topic).filter(
                Topic.document_id == doc_id,
                Topic.name.ilike(topic_name),
            ).first()
            if existing:
                return _topic_to_dict(existing)
            topic = Topic(document_id=doc_id, name=topic_name, summary=summary)
            db.add(topic)
            db.commit()
            db.refresh(topic)
            return _topic_to_dict(topic)

    def get_or_create_subtopic(self, topic_id: int, name: str, summary: str = "") -> dict:
        """Find an existing subtopic by name (case-insensitive) or create it."""
        with SessionLocal() as db:
            existing = db.query(Subtopic).filter(
                Subtopic.topic_id == topic_id,
                Subtopic.name.ilike(name),
            ).first()
            if existing:
                return _subtopic_to_dict(existing)
            sub = Subtopic(topic_id=topic_id, name=name, summary=summary)
            db.add(sub)
            db.commit()
            db.refresh(sub)
            return _subtopic_to_dict(sub)

    def delete_topic_cascade(self, topic_id: int, doc_id: str) -> int:
        """Delete a topic, its subtopics, associated chunks, and non-approved flashcards.

        H6 invariant: approved flashcards are preserved — their subtopic_id is set to
        NULL so they remain available for study without a category tag.

        Returns:
            int: count of approved flashcards that were preserved (subtopic unlinked).
        """
        with SessionLocal() as db:
            subtopics = db.query(Subtopic).filter(Subtopic.topic_id == topic_id).all()
            subtopic_ids = [s.id for s in subtopics]
            preserved_count = 0

            if subtopic_ids:
                # Count approved cards before touching anything
                preserved_count = db.query(Flashcard).filter(
                    Flashcard.subtopic_id.in_(subtopic_ids),
                    Flashcard.status == "approved",
                ).count()

                if preserved_count > 0:
                    # Unlink approved cards instead of deleting them
                    db.execute(
                        sa_update(Flashcard)
                        .where(
                            Flashcard.subtopic_id.in_(subtopic_ids),
                            Flashcard.status == "approved",
                        )
                        .values(subtopic_id=None),
                        execution_options={"synchronize_session": False},
                    )
                    logger.info(
                        "Topic %d deletion: preserved %d approved flashcard(s) by unlinking subtopic",
                        topic_id, preserved_count,
                    )

                # Delete non-approved flashcards
                db.execute(
                    delete(Flashcard).where(
                        Flashcard.subtopic_id.in_(subtopic_ids),
                        Flashcard.status != "approved",
                    )
                )

            db.execute(delete(Subtopic).where(Subtopic.topic_id == topic_id))
            db.execute(delete(Topic).where(Topic.id == topic_id))
            db.execute(delete(ContentChunk).where(ContentChunk.document_id == doc_id))
            db.commit()

        return preserved_count

    def get_by_subject(self, subject_id: int) -> List[dict]:
        """Return all topics for a subject via SubjectDocumentAssociation."""
        with SessionLocal() as db:
            topics = (
                db.query(Topic)
                .join(SubjectDocumentAssociation, Topic.document_id == SubjectDocumentAssociation.document_id)
                .filter(SubjectDocumentAssociation.subject_id == subject_id)
                .order_by(Topic.created_at.desc())
                .all()
            )
            return [_topic_to_dict(t) for t in topics]

    def get_subtopics_for_topic_ids(self, topic_ids: List[int]) -> Dict[int, List[dict]]:
        """Single query: fetch all subtopics (with card counts) for multiple topic IDs.

        Returns {topic_id: [subtopic_dict, ...]}.
        """
        if not topic_ids:
            return {}
        with SessionLocal() as db:
            subs = db.query(Subtopic).filter(Subtopic.topic_id.in_(topic_ids)).all()
            if not subs:
                return {}

            subtopic_ids = [s.id for s in subs]
            counts: Dict[int, Dict[str, int]] = {}
            rows = (
                db.query(Flashcard.subtopic_id, Flashcard.status, func.count(Flashcard.id))
                .filter(Flashcard.subtopic_id.in_(subtopic_ids))
                .group_by(Flashcard.subtopic_id, Flashcard.status)
                .all()
            )
            for sub_id, status, count in rows:
                if sub_id not in counts:
                    counts[sub_id] = {"approved": 0, "pending": 0}
                if status in counts[sub_id]:
                    counts[sub_id][status] = count

            result: Dict[int, List[dict]] = {}
            for s in subs:
                d = _subtopic_to_dict(s)
                d["approved_count"] = counts.get(s.id, {}).get("approved", 0)
                d["pending_count"] = counts.get(s.id, {}).get("pending", 0)
                result.setdefault(s.topic_id, []).append(d)
            return result

    def get_subtopics_with_counts(self, topic_id: int) -> List[dict]:
        """Return subtopics for a topic with approved and pending card counts."""
        from sqlalchemy import func, case, Integer
        with SessionLocal() as db:
            subs = db.query(Subtopic).filter(Subtopic.topic_id == topic_id).all()
            result = []
            for s in subs:
                approved = db.query(func.count(Flashcard.id)).filter(
                    Flashcard.subtopic_id == s.id, Flashcard.status == "approved"
                ).scalar() or 0
                pending = db.query(func.count(Flashcard.id)).filter(
                    Flashcard.subtopic_id == s.id, Flashcard.status == "pending"
                ).scalar() or 0
                d = _subtopic_to_dict(s)
                d["approved_count"] = approved
                d["pending_count"] = pending
                result.append(d)
            return result
