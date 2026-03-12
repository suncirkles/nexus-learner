import pytest
import uuid
from core.database import SessionLocal, Subject, Document, Topic, Subtopic, Flashcard, Base, engine

@pytest.fixture(scope="function", autouse=True)
def setup_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield

def test_topic_bulk_actions():
    db = SessionLocal()
    try:
        # Setup data
        subject = Subject(name="Test Subject")
        db.add(subject)
        db.commit()
        
        topic = Topic(name="Test Topic", subject_id=subject.id)
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
        
        # Mock "Approve All Topic" logic
        sub_ids = [sub1.id, sub2.id]
        db.query(Flashcard).filter(Flashcard.subtopic_id.in_(sub_ids), Flashcard.status == "pending").update({"status": "approved"}, synchronize_session=False)
        db.commit()
        
        # Verify approved state
        approved_count = db.query(Flashcard).filter(Flashcard.status == "approved").count()
        assert approved_count == 2
        
        # Mock "Reject All Topic" logic (bringing back to pending first)
        db.query(Flashcard).update({"status": "pending"})
        db.commit()
        
        db.query(Flashcard).filter(Flashcard.subtopic_id.in_(sub_ids), Flashcard.status == "pending").update({"status": "rejected"}, synchronize_session=False)
        db.commit()
        
        # Verify rejected state
        rejected_count = db.query(Flashcard).filter(Flashcard.status == "rejected").count()
        assert rejected_count == 2
        
    finally:
        db.close()
