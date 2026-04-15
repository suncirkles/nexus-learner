"""
api/dependencies.py
--------------------
FastAPI Depends() factories that construct service instances with their concrete
repo implementations. No DI framework — plain Python constructor injection.
"""

from functools import lru_cache
from fastapi import Depends
from sqlalchemy.orm import Session

from repositories.sql.subject_repo import SubjectRepo
from repositories.sql.flashcard_repo import FlashcardRepo
from repositories.sql.topic_repo import TopicRepo
from repositories.sql.document_repo import DocumentRepo
from repositories.sql.chunk_repo import ChunkRepo
from repositories.protocols import VectorStoreProtocol
from repositories.vector.factory import get_vector_store as get_vector_store_factory

from services.subject_service import SubjectService
from services.flashcard_service import FlashcardService
from services.topic_service import TopicService
from services.library_service import LibraryService
from services.system_service import SystemService

from core.database import get_db as core_get_db


# ---------------------------------------------------------------------------
# Singleton repo/store instances (cheap to share across requests)
# ---------------------------------------------------------------------------

def get_db():
    """Returns a database session. Re-exported from core.database."""
    yield from core_get_db()


def get_subject_repo() -> SubjectRepo:
    return SubjectRepo()


def get_flashcard_repo() -> FlashcardRepo:
    return FlashcardRepo()


def get_topic_repo() -> TopicRepo:
    return TopicRepo()


def get_document_repo() -> DocumentRepo:
    return DocumentRepo()


def get_chunk_repo() -> ChunkRepo:
    return ChunkRepo()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreProtocol:
    """Shared VectorStore — provider is selected based on settings."""
    return get_vector_store_factory()


# ---------------------------------------------------------------------------
# Service factories (assembled from repos)
# ---------------------------------------------------------------------------

def get_subject_service(
    subject_repo: SubjectRepo = Depends(get_subject_repo),
    flashcard_repo: FlashcardRepo = Depends(get_flashcard_repo),
    document_repo: DocumentRepo = Depends(get_document_repo),
) -> SubjectService:
    return SubjectService(subject_repo, flashcard_repo, document_repo)


def get_flashcard_service(
    flashcard_repo: FlashcardRepo = Depends(get_flashcard_repo),
    chunk_repo: ChunkRepo = Depends(get_chunk_repo),
) -> FlashcardService:
    return FlashcardService(flashcard_repo, chunk_repo)


def get_topic_service(
    topic_repo: TopicRepo = Depends(get_topic_repo),
    chunk_repo: ChunkRepo = Depends(get_chunk_repo),
) -> TopicService:
    return TopicService(topic_repo, chunk_repo, get_vector_store())


def get_library_service(
    doc_repo: DocumentRepo = Depends(get_document_repo),
) -> LibraryService:
    return LibraryService(doc_repo, get_vector_store())


def get_system_service() -> SystemService:
    return SystemService(get_vector_store())
