
import pytest
import os
import uuid
from agents.ingestion import IngestionAgent
from core.database import SessionLocal, Document, Subject, Base, engine

@pytest.fixture(scope="function", autouse=True)
def setup_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def test_duplicate_detection():
    agent = IngestionAgent()
    doc_id1 = str(uuid.uuid4())
    doc_id2 = str(uuid.uuid4())
    
    # Create a dummy PDF file
    import fitz
    test_file = "test_doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is some unique content for testing duplicate detection.")
    doc.save(test_file)
    doc.close()
    
    db = SessionLocal()
    subject = Subject(name="Test Subject Duplicates")
    db.add(subject)
    db.commit()
    subject_id = subject.id
    db.close()
    
    try:
        # First upload should succeed
        agent.process_document(test_file, doc_id1, subject_id)
        
        # Second upload with same content should fail
        with pytest.raises(ValueError) as excinfo:
            agent.process_document(test_file, doc_id2, subject_id)
        
        assert "Duplicate content detected" in str(excinfo.value)
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    pytest.main([__file__])
