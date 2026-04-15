"""
core/database.py
-----------------
SQLAlchemy ORM models and database session management.
Defines the relational schema for Subjects, Documents, Topics,
Subtopics, and Flashcards.
"""

from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, ForeignKey, text, inspect, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone
from .config import settings
import logging

logger = logging.getLogger(__name__)

# --- Engine & Session Factory ---

# Standard SQLAlchemy engine creation. DB_URL should point to a PostgreSQL instance (Supabase).
engine = create_engine(settings.DB_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


@contextmanager
def get_session():
    """Context manager for safe database session usage."""
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


class SubjectTopicAssociation(Base):
    """Explicit Many-to-Many link between Subjects and Topics.

    A topic discovered in a document can be studied by multiple subjects.
    This table is the authoritative source for which topics belong to which
    subject, making subject→topic→subtopic→flashcard a fully explicit chain.

    Populated/refreshed at the start of each GENERATION run so that stale
    topic associations are cleared and current ones are guaranteed present
    before any flashcard FK check.
    """
    __tablename__ = "subject_topic_association"

    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    topic_id = Column(Integer, ForeignKey("topics.id", ondelete="CASCADE"), index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("subject_id", "topic_id", name="uq_subject_topic"),
    )


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
    page_number = Column(Integer, nullable=True)                       # 0-based page index within the source PDF
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
    flashcards = relationship("Flashcard", back_populates="topic")


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


class BatchJob(Base):
    """Tracks a submitted Anthropic Messages Batch API job."""
    __tablename__ = "batch_jobs"

    id = Column(String, primary_key=True, index=True)                  # local UUID
    anthropic_batch_id = Column(String, nullable=True, index=True)     # set after submit
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    status = Column(String(20), default="indexing")
    # enum: "indexing" | "generating" | "submitted" | "collecting" | "completed" | "failed"
    doc_ids = Column(Text)           # JSON list of document UUIDs
    question_types = Column(Text)    # JSON list of question type strings
    request_count = Column(Integer, default=0)
    completed_count = Column(Integer, default=0)
    error = Column(Text, nullable=True)
    submitted_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Progress tracking columns (written by the Modal worker; read by the UI via API poll)
    filename = Column(Text, nullable=True)
    status_message = Column(Text, nullable=True)
    total_pages = Column(Integer, default=0)
    current_page = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    current_chunk_index = Column(Integer, default=0)
    flashcards_count = Column(Integer, default=0)

    subject = relationship("Subject")
    requests = relationship("BatchRequest", back_populates="job", cascade="all, delete-orphan")


class BatchRequest(Base):
    """Tracks one request within a batch job (custom_id → chunk + question_type)."""
    __tablename__ = "batch_requests"

    id = Column(String, primary_key=True)   # custom_id: "{job_id}:{chunk_id}:{qtype}"
    job_id = Column(String, ForeignKey("batch_jobs.id", ondelete="CASCADE"), index=True)
    chunk_id = Column(Integer, nullable=True)
    question_type = Column(String(30))
    status = Column(String(20), default="pending")
    # enum: "pending" | "succeeded" | "errored"
    flashcard_ids = Column(Text, nullable=True)  # JSON list of created flashcard IDs
    error = Column(Text, nullable=True)

    job = relationship("BatchJob", back_populates="requests")


class Flashcard(Base):
    """Stores AI-generated Active Recall questions (Linked to a specific Subject)."""
    __tablename__ = "flashcards"

    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    # Full hierarchy stored explicitly: subject → topic → subtopic → flashcard
    # This makes mentor-review queries direct and avoids the
    # SubjectDocumentAssociation→Document→Topic→Subtopic join for card counts.
    topic_id = Column(Integer, ForeignKey("topics.id", ondelete="SET NULL"), nullable=True, index=True)
    subtopic_id = Column(Integer, ForeignKey("subtopics.id", ondelete="CASCADE"), index=True)
    chunk_id = Column(Integer, nullable=True)  # Link to ContentChunk ID for reference
    
    question = Column(Text)
    answer = Column(Text)
    
    # Question type & complexity
    question_type    = Column(String(30), default="active_recall")
    # enum: active_recall | fill_blank | short_answer | long_answer | numerical | scenario
    complexity_level = Column(String(10), nullable=True)
    # enum: simple | medium | complex — nullable until mentor confirms
    rubric           = Column(Text, nullable=True)
    # JSON: list of {criterion, description} grading criteria for this card
    critic_rubric_scores = Column(Text, nullable=True)
    # JSON: {"accuracy": 1-4, "logic": 1-4, "grounding": 1-4, "clarity": 1-4}

    # Evals / Grounding
    critic_score = Column(Integer, default=0)
    # Backward-compatible aggregate: round(mean([accuracy, logic, grounding, clarity]))
    critic_feedback = Column(Text, nullable=True)

    # HITL State
    status = Column(String, default="pending")  # pending, approved, rejected
    mentor_feedback = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    subject = relationship("Subject", back_populates="flashcards")
    topic = relationship("Topic", back_populates="flashcards")
    subtopic = relationship("Subtopic", back_populates="flashcards")


# Create tables (checkfirst=True makes this a true no-op if tables exist)
Base.metadata.create_all(bind=engine, checkfirst=True)


_migrations_done = False


def _run_migrations():
    """Adds new columns/tables to an existing database without dropping data.
    
    Dialect-agnostic using SQLAlchemy Inspect. Safe for both SQLite and Postgres.
    """
    global _migrations_done
    if _migrations_done:
        return
    _migrations_done = True

    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
    except Exception as e:
        logger.warning(f"Could not inspect database for migrations: {e}")
        return

    migrations = [
        # New metric columns on documents
        ("documents", "relevance_rate",    "INTEGER NOT NULL DEFAULT 0"),
        ("documents", "yield_rate",         "INTEGER NOT NULL DEFAULT 0"),
        ("documents", "faithfulness_score", "INTEGER NOT NULL DEFAULT 0"),
        # Flashcards now carry a direct subject FK
        ("flashcards", "subject_id", "INTEGER REFERENCES subjects(id) ON DELETE CASCADE"),
        # Chunks are now assigned to a specific subtopic
        ("content_chunks", "subtopic_id", "INTEGER REFERENCES subtopics(id) ON DELETE SET NULL"),
        # Page provenance for source image rendering
        ("content_chunks", "page_number", "INTEGER"),
        # Topics are now owned by a document, not a subject
        ("topics", "document_id", "VARCHAR REFERENCES documents(id) ON DELETE CASCADE"),
        # Phase 2.5: atomic content columns on flashcards
        ("flashcards", "question_type",        "VARCHAR(30) NOT NULL DEFAULT 'active_recall'"),
        ("flashcards", "complexity_level",     "VARCHAR(10)"),
        ("flashcards", "rubric",               "TEXT"),
        ("flashcards", "critic_rubric_scores", "TEXT"),
        # Full subject→topic→subtopic hierarchy stored on the flashcard for direct querying
        ("flashcards", "topic_id", "INTEGER REFERENCES topics(id) ON DELETE SET NULL"),
        # Explicit subject↔topic many-to-many (populated/refreshed each GENERATION run)
        # NOTE: table creation is handled separately below (needs CREATE TABLE, not ALTER)

        # Modal deployment: DB-backed progress so worker updates are visible across containers
        ("batch_jobs", "filename",             "TEXT"),
        ("batch_jobs", "status_message",       "TEXT"),
        ("batch_jobs", "total_pages",          "INTEGER DEFAULT 0"),
        ("batch_jobs", "current_page",         "INTEGER DEFAULT 0"),
        ("batch_jobs", "total_chunks",         "INTEGER DEFAULT 0"),
        ("batch_jobs", "current_chunk_index",  "INTEGER DEFAULT 0"),
        ("batch_jobs", "flashcards_count",     "INTEGER DEFAULT 0"),
    ]

    with engine.begin() as conn:
        for table, column, col_def in migrations:
            if table in existing_tables:
                try:
                    cols = [c["name"] for c in inspector.get_columns(table)]
                    if column not in cols:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}"))
                        logger.info(f"Migration: added {table}.{column}")
                except Exception as e:
                    logger.warning(f"Migration failed for {table}.{column}: {e}")


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
