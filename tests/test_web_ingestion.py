import pytest
import uuid
from unittest.mock import MagicMock, patch
from core.database import SessionLocal, Subject, Document, Flashcard
from workflows.phase2_web_ingestion import phase2_graph

@pytest.fixture
def db():
    session = SessionLocal()
    yield session
    session.close()

@patch("agents.safety.SafetyAgent.check_subject_safety")
@patch("agents.web_researcher.WebResearchAgent.research_topics")
@patch("agents.curator.CuratorAgent.curate_structure")
@patch("agents.socratic.SocraticAgent.generate_flashcard")
@patch("agents.critic.CriticAgent.evaluate_flashcard")
def test_web_ingestion_workflow(
    mock_critic, mock_socratic, mock_curator, mock_research, mock_safety, db
):
    # 1. Setup
    subject_name = f"Web Test Subject {uuid.uuid4().hex[:6]}"
    subject = Subject(name=subject_name)
    db.add(subject)
    db.commit()

    # Mock safety check returns safe
    mock_safety.return_value = MagicMock(is_safe=True)

    # Mock research returns one document
    mock_research.return_value = [
        MagicMock(
            url="https://example_test.com/page1",
            title="Test Page",
            domain="example_test.com",
            content="This is test content for web research.",
            content_hash="hash123",
            model_dump=lambda: {
                "url": "https://example_test.com/page1",
                "title": "Test Page",
                "domain": "example_test.com",
                "content": "This is test content for web research.",
                "content_hash": "hash123"
            }
        )
    ]

    # Mock curator returns hierarchy
    mock_curator.return_value = {
        "hierarchy": [{"name": "Topic 1", "subtopics": [{"id": 1, "name": "Sub 1", "summary": "..."}]}],
        "doc_summary": "Summary..."
    }

    # Mock socratic returns a flashcard
    mock_socratic.return_value = {
        "flashcard_id": 1,
        "question": "Web test question?",
        "answer": "Web test answer.",
        "status": "success"
    }

    initial_state = {
        "subject_id": subject.id,
        "subject_name": subject.name,
        "topics": ["Testing"],
        "web_documents": [],
        "current_doc_index": 0,
        "doc_id": "",
        "full_text": "",
        "chunks": [],
        "hierarchy": [],
        "doc_summary": "",
        "current_chunk_index": 0,
        "generated_flashcards": [],
        "status_message": "",
        "safety_blocked": False,
        "safety_reason": "",
        "processed_urls": [],
    }

    # 2. Execute
    final_state = phase2_graph.invoke(initial_state)

    # 3. Verify
    assert final_state["safety_blocked"] is False
    assert len(final_state["processed_urls"]) == 1
    assert len(final_state["generated_flashcards"]) > 0
    
    # Check if document was saved to DB
    doc = db.query(Document).filter(Document.subject_id == subject.id).first()
    assert doc is not None
    assert doc.source_type == "web"
    assert doc.source_url == "https://example_test.com/page1"

    # Cleanup
    db.delete(subject)
    db.commit()
