import pytest
import os
import uuid
import workflows.phase1_ingestion
from unittest.mock import patch, MagicMock

from core.database import SessionLocal, ContentChunk, Flashcard, Topic, Subtopic, Subject, Base, engine
from agents.socratic import FlashcardOutput
from agents.critic import GroundingEvaluation
from agents.curator import DocumentStructure, TopicStructure, SubtopicStructure

# Ensure tables are built for tests
@pytest.fixture(scope="function", autouse=True)
def setup_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield

@pytest.fixture
def mock_embeddings():
    with patch('agents.ingestion.OpenAIEmbeddings') as MockEmbeddings:
        instance = MockEmbeddings.return_value
        instance.embed_documents.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
        instance.embed_query.side_effect = lambda text: [0.1] * 1536
        yield instance

@pytest.fixture
def mock_curator():
    with patch.object(workflows.phase1_ingestion.curator_agent, 'chain') as mock_chain:
        mock_chain.invoke.return_value = DocumentStructure(
            summary="Test document summary",
            topics=[
                TopicStructure(
                    name="Test Topic 1",
                    summary="Description of Topic 1",
                    subtopics=[
                        SubtopicStructure(name="Subtopic A", summary="Summary A"),
                        SubtopicStructure(name="Subtopic B", summary="Summary B")
                    ]
                )
            ]
        )
        yield mock_chain

@pytest.fixture
def mock_socratic_chain():
    with patch.object(workflows.phase1_ingestion.socratic_agent, 'chain') as mock_chain:
        mock_chain.invoke.return_value = FlashcardOutput(
            question="What is the concept discussed in the text?",
            answer="AI Engineering hierarchical test."
        )
        yield mock_chain

@pytest.fixture
def mock_tesseract():
    with patch('pytesseract.image_to_string') as mock_ocr:
        mock_ocr.return_value = "Mocked OCR text."
        yield mock_ocr

@pytest.fixture
def mock_critic():
    with patch.object(workflows.phase1_ingestion.critic_agent, 'chain') as mock_chain:
        mock_chain.invoke.return_value = GroundingEvaluation(
            score=5,
            feedback="Strongly grounded."
        )
        yield mock_chain

@pytest.fixture
def mock_classifier():
    with patch('core.models.get_llm') as mock_get_llm:
        mock_model = MagicMock()
        mock_get_llm.return_value = mock_model
        
        class MockClassification:
            subtopic_id = 1
            
        mock_model.with_structured_output.return_value.invoke.return_value = MockClassification()
        
        # Ensure title generation works whether called via .invoke() or directly (RunnableLambda)
        mock_res = MagicMock()
        mock_res.content = "Test Document Title"
        mock_model.invoke.return_value = mock_res
        mock_model.return_value = mock_res
        
        yield mock_model

def test_e2e_hierarchical_workflow(mock_embeddings, mock_curator, mock_socratic_chain, mock_tesseract, mock_classifier, mock_critic):
    """Verifies the expanded Phase 1 flow: Ingest -> Curate -> Generate -> Critic."""
    file_path = os.path.join("documents", "gemini 3 developer guide march 2026.pdf")
    doc_id = str(uuid.uuid4())
    
    # Ensure the directory exists for fitz
    if not os.path.exists("documents"):
        os.makedirs("documents")
    
    # Create a real dummy PDF so fitz doesn't complain during ingestion
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Test content for E2E flow.")
    doc.save(file_path)
    doc.close()
    
    db = SessionLocal()
    subject = Subject(name="Test Subject E2E")
    db.add(subject)
    db.commit()
    subject_id = subject.id
    db.close()
    
    initial_state = {
        "file_path": file_path,
        "doc_id": doc_id,
        "subject_id": subject_id,
        "chunks": [],
        "hierarchy": [],
        "doc_summary": "",
        "current_chunk_index": 0,
        "generated_flashcards": [],
        "status_message": "Starting E2E Test"
    }
    
    try:
        # Execute Graph
        # We need to handle the fact that graph.stream/invoke might be used
        final_state = workflows.phase1_ingestion.phase1_graph.invoke(initial_state)
        
        # Verify Hierarchy logic
        assert len(final_state["hierarchy"]) == 1
        assert final_state["hierarchy"][0]["name"] == "Test Topic 1"
        assert len(final_state["hierarchy"][0]["subtopics"]) == 2
        
        # Verify DB persistence
        db = SessionLocal()
        try:
            # Check Document table
            from core.database import Document as DBDocument
            db_doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
            assert db_doc is not None
            assert db_doc.filename == "gemini 3 developer guide march 2026.pdf"

            topic = db.query(Topic).filter(Topic.document_id == doc_id).first()
            assert topic is not None
            assert topic.name == "Test Topic 1"
            
            subtopic = db.query(Subtopic).filter(Subtopic.topic_id == topic.id).first()
            assert subtopic is not None
            
            flashcard = db.query(Flashcard).filter(Flashcard.subtopic_id == subtopic.id).first()
            assert flashcard is not None
            assert flashcard.question == "What is the concept discussed in the text?"
            assert flashcard.status == "pending"
        finally:
            db.close()
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
