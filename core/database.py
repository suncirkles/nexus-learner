"""
core/database.py
-----------------
SQLAlchemy ORM models and database session management.
Defines the relational schema for Subjects, Documents, Topics,
Subtopics, and Flashcards. Uses SQLite for the MVP.
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
from .config import settings

engine = create_engine(settings.DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Subject(Base):
    """Broad academic or professional subjects."""
    __tablename__ = "subjects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class ContentChunk(Base):
    __tablename__ = "content_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String, index=True)
    text = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Document(Base):
    """Tracks unique uploaded files to prevent duplicates."""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True) # UUID
    subject_id = Column(Integer, index=True) # Linked to a Subject
    filename = Column(String)
    title = Column(String, nullable=True)  # New field for meaningful names
    content_hash = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Topic(Base):
    """Broad categories identified in the document (now owned by Subject)."""
    __tablename__ = "topics"
    
    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, index=True) # Linked to a Subject
    document_id = Column(String, index=True, nullable=True) # Optional source link
    name = Column(String)
    summary = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Subtopic(Base):
    """Specific lessons or sections within a Topic."""
    __tablename__ = "subtopics"
    
    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, index=True)
    name = Column(String)
    summary = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
class Flashcard(Base):
    """Stores AI-generated Active Recall questions."""
    __tablename__ = "flashcards"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer) # Optional link to source chunk
    subtopic_id = Column(Integer, index=True) # Linked to hierarchy
    question = Column(Text)
    answer = Column(Text)
    
    # Evals / Grounding
    critic_score = Column(Integer, default=0) # 1-5 rating on factual accuracy
    critic_feedback = Column(Text, nullable=True)

    # HITL (Human-in-the-Loop) State
    status = Column(String, default="pending") # pending, approved, rejected
    mentor_feedback = Column(Text, nullable=True) # Manual adjustments
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
