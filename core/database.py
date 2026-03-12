"""
core/database.py
-----------------
SQLAlchemy ORM models and database session management.
Defines the relational schema for Subjects, Documents, Topics,
Subtopics, and Flashcards. Uses SQLite for the MVP.
"""

from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone
from .config import settings

# --- Engine & Session Factory ---

engine = create_engine(settings.DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


@contextmanager
def get_session():
    """Context manager for safe database session usage.
    
    Usage:
        with get_session() as db:
            db.query(Subject).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- ORM Models ---

class Subject(Base):
    """Broad academic or professional subjects."""
    __tablename__ = "subjects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    documents = relationship("Document", back_populates="subject", cascade="all, delete-orphan")
    topics = relationship("Topic", back_populates="subject", cascade="all, delete-orphan")


class Document(Base):
    """Tracks unique uploaded files to prevent duplicates."""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)  # UUID
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    filename = Column(String)
    title = Column(String, nullable=True)
    content_hash = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    subject = relationship("Subject", back_populates="documents")
    chunks = relationship("ContentChunk", back_populates="document", cascade="all, delete-orphan")


class ContentChunk(Base):
    """Individual text chunks extracted from a document."""
    __tablename__ = "content_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    text = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    document = relationship("Document", back_populates="chunks")


class Topic(Base):
    """Broad categories identified in the document (owned by Subject)."""
    __tablename__ = "topics"
    
    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="SET NULL"), index=True, nullable=True)
    name = Column(String)
    summary = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    subject = relationship("Subject", back_populates="topics")
    subtopics = relationship("Subtopic", back_populates="topic", cascade="all, delete-orphan")


class Subtopic(Base):
    """Specific lessons or sections within a Topic."""
    __tablename__ = "subtopics"
    
    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey("topics.id", ondelete="CASCADE"), index=True)
    name = Column(String)
    summary = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    topic = relationship("Topic", back_populates="subtopics")
    flashcards = relationship("Flashcard", back_populates="subtopic", cascade="all, delete-orphan")


class Flashcard(Base):
    """Stores AI-generated Active Recall questions."""
    __tablename__ = "flashcards"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, nullable=True)  # Optional link to source chunk
    subtopic_id = Column(Integer, ForeignKey("subtopics.id", ondelete="CASCADE"), index=True)
    question = Column(Text)
    answer = Column(Text)
    
    # Evals / Grounding
    critic_score = Column(Integer, default=0)  # 1-5 rating on factual accuracy
    critic_feedback = Column(Text, nullable=True)

    # HITL (Human-in-the-Loop) State
    status = Column(String, default="pending")  # pending, approved, rejected
    mentor_feedback = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    subtopic = relationship("Subtopic", back_populates="flashcards")


# Create tables (no-op if they already exist)
Base.metadata.create_all(bind=engine)


def get_db():
    """Generator-based session for dependency injection (e.g., FastAPI)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
