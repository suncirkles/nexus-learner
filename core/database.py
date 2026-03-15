"""
core/database.py
-----------------
SQLAlchemy ORM models and database session management.
Defines the relational schema for Subjects, Documents, Topics,
Subtopics, and Flashcards. Uses SQLite for the MVP.
"""

from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, ForeignKey, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone
from .config import settings
import logging

logger = logging.getLogger(__name__)

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


# --- Bridge Tables ---

class SubjectDocumentAssociation(Base):
    """Many-to-Many link between Subjects and Documents."""
    __tablename__ = "subject_document_association"
    
    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# --- ORM Models ---

class Subject(Base):
    """Broad academic or professional subjects."""
    __tablename__ = "subjects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    is_archived = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    documents = relationship("Document", secondary="subject_document_association", back_populates="subjects")
    # Note: Flashcards are linked to subjects directly now
    flashcards = relationship("Flashcard", back_populates="subject", cascade="all, delete-orphan")


class Document(Base):
    """Tracks unique uploaded files to prevent duplicates globally."""
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)  # UUID
    filename = Column(String)
    title = Column(String, nullable=True)
    content_hash = Column(String, unique=True, index=True)
    source_type = Column(String(20), nullable=False, default="pdf")   # "pdf" | "image" | "web" | "text"
    source_url = Column(String(2048), nullable=True)                   # URL if source_type == "web"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # RAG Metrics
    relevance_rate = Column(Integer, default=0)       # % of chunks accepted
    yield_rate = Column(Integer, default=0)           # cards per 10 chunks (x10 for precision)
    faithfulness_score = Column(Integer, default=0)   # 1-10 average grounding

    # Relationships
    subjects = relationship("Subject", secondary="subject_document_association", back_populates="documents")
    chunks = relationship("ContentChunk", back_populates="document", cascade="all, delete-orphan")
    topics = relationship("Topic", back_populates="document", cascade="all, delete-orphan")


class ContentChunk(Base):
    """Individual text chunks extracted from a document."""
    __tablename__ = "content_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    text = Column(Text)
    source_type = Column(String(20), nullable=False, default="pdf")   # "pdf" | "image" | "web" | "text"
    source_url = Column(String(2048), nullable=True)                   # URL if source_type == "web"
    subtopic_id = Column(Integer, ForeignKey("subtopics.id", ondelete="SET NULL"), index=True, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    document = relationship("Document", back_populates="chunks")
    subtopic = relationship("Subtopic")


class Topic(Base):
    """Broad categories identified in a document (Global to the Library)."""
    __tablename__ = "topics"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    name = Column(String)
    summary = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    document = relationship("Document", back_populates="topics")
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
    """Stores AI-generated Active Recall questions (Linked to a specific Subject)."""
    __tablename__ = "flashcards"

    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    subtopic_id = Column(Integer, ForeignKey("subtopics.id", ondelete="CASCADE"), index=True)
    chunk_id = Column(Integer, nullable=True)  # Link to ContentChunk ID for reference
    
    question = Column(Text)
    answer = Column(Text)
    
    # Evals / Grounding
    critic_score = Column(Integer, default=0)
    critic_feedback = Column(Text, nullable=True)

    # HITL State
    status = Column(String, default="pending")  # pending, approved, rejected
    mentor_feedback = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    subject = relationship("Subject", back_populates="flashcards")
    subtopic = relationship("Subtopic", back_populates="flashcards")


# Create tables (no-op if they already exist)
Base.metadata.create_all(bind=engine)


def _run_migrations():
    """Adds new columns/tables to an existing database without dropping data.

    Safe to call on both fresh and legacy databases. Uses PRAGMA table_info
    to detect missing columns before issuing ALTER TABLE.
    """
    with engine.connect() as conn:
        def column_exists(table: str, column: str) -> bool:
            rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            return any(row[1] == column for row in rows)

        def table_exists(table: str) -> bool:
            row = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
                {"t": table}
            ).fetchone()
            return row is not None

        migrations = [
            # New metric columns on documents
            ("documents", "relevance_rate",    "INTEGER NOT NULL DEFAULT 0"),
            ("documents", "yield_rate",         "INTEGER NOT NULL DEFAULT 0"),
            ("documents", "faithfulness_score", "INTEGER NOT NULL DEFAULT 0"),
            # Flashcards now carry a direct subject FK
            ("flashcards", "subject_id", "INTEGER REFERENCES subjects(id) ON DELETE CASCADE"),
            # Chunks are now assigned to a specific subtopic
            ("content_chunks", "subtopic_id", "INTEGER REFERENCES subtopics(id) ON DELETE SET NULL"),
            # Topics are now owned by a document, not a subject
            ("topics", "document_id", "VARCHAR REFERENCES documents(id) ON DELETE CASCADE"),
        ]

        for table, column, col_def in migrations:
            if table_exists(table) and not column_exists(table, column):
                try:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}"))
                    conn.commit()
                    logger.info(f"Migration: added {table}.{column}")
                except Exception as e:
                    conn.rollback()
                    logger.warning(f"Migration skipped {table}.{column}: {e}")


_run_migrations()


def reset_database():
    """Drops all tables and recreates them, plus clears Qdrant."""
    logger.info("--- RESETTING DATABASE ---")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(settings.QDRANT_URL)
        if client.collection_exists(settings.QDRANT_COLLECTION_NAME):
            client.delete_collection(settings.QDRANT_COLLECTION_NAME)
            logger.info(f"Deleted Qdrant collection: {settings.QDRANT_COLLECTION_NAME}")
    except Exception as e:
        logger.warning(f"Failed to clear Qdrant: {e}")


def get_db():
    """Generator-based session for dependency injection (e.g., FastAPI)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
