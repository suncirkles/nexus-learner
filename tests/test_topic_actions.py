"""
tests/test_topic_actions.py
-----------------------------
Tests for Mentor HITL bulk approve/reject actions on Topics.

Topic is now linked to Document (document_id), not Subject directly.
The bulk-action logic only cares about Subtopic → Flashcard relationships.
"""

import pytest
import uuid
from core.database import (
    SessionLocal, Subject, Document as DBDocument,
    Topic, Subtopic, Flashcard, Base, engine,
)


@pytest.fixture(scope="function", autouse=True)
def setup_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


def test_topic_bulk_actions():
    db = SessionLocal()
    try:
        # Setup data — Topic requires a Document (document_id), not a Subject
        doc_id = str(uuid.uuid4())
        doc = DBDocument(id=doc_id, filename="test.pdf", title="Test", content_hash=f"hash_{doc_id}")
        db.add(doc)
        db.commit()

        topic = Topic(name="Test Topic", document_id=doc_id)
        db.add(topic)
        db.commit()

        sub1 = Subtopic(name="Sub 1", topic_id=topic.id)
        sub2 = Subtopic(name="Sub 2", topic_id=topic.id)
        db.add_all([sub1, sub2])
        db.commit()

        fc1 = Flashcard(subtopic_id=sub1.id, question="Q1", answer="A1", status="pending")
        fc2 = Flashcard(subtopic_id=sub2.id, question="Q2", answer="A2", status="pending")
        db.add_all([fc1, fc2])
        db.commit()

        # Verify initial state
        pending_count = db.query(Flashcard).filter(Flashcard.status == "pending").count()
        assert pending_count == 2

        # Simulate "Approve All Topic"
        sub_ids = [sub1.id, sub2.id]
        db.query(Flashcard).filter(
            Flashcard.subtopic_id.in_(sub_ids), Flashcard.status == "pending"
        ).update({"status": "approved"}, synchronize_session=False)
        db.commit()

        approved_count = db.query(Flashcard).filter(Flashcard.status == "approved").count()
        assert approved_count == 2

        # Reset to pending then simulate "Reject All Topic"
        db.query(Flashcard).update({"status": "pending"})
        db.commit()

        db.query(Flashcard).filter(
            Flashcard.subtopic_id.in_(sub_ids), Flashcard.status == "pending"
        ).update({"status": "rejected"}, synchronize_session=False)
        db.commit()

        rejected_count = db.query(Flashcard).filter(Flashcard.status == "rejected").count()
        assert rejected_count == 2

    finally:
        db.close()
