"""
repositories/sql/chunk_repo.py
--------------------------------
SQLAlchemy implementation of ChunkRepoProtocol.
create_batch() fixes the N+1 commit bug in agents/ingestion.py by
persisting all chunks in a single transaction.
"""

from typing import List, Optional
from core.database import SessionLocal, ContentChunk, Subtopic, Topic


def _chunk_to_dict(chunk: ContentChunk) -> dict:
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "source_type": chunk.source_type,
        "source_url": chunk.source_url,
        "subtopic_id": chunk.subtopic_id,
        "page_number": chunk.page_number,
        "created_at": chunk.created_at,
    }


class ChunkRepo:
    """Concrete SQL implementation of ChunkRepoProtocol."""

    def create_batch(self, doc_id: str, chunks: List[dict]) -> List[dict]:
        """Persist a batch of chunks in one transaction (eliminates N+1 commit bug).

        Each dict in `chunks` may contain:
            text        (required)
            subtopic_id (optional)
            page_number (optional)
            source_type (optional, default "pdf")
            source_url  (optional)

        Returns a list of dicts with the assigned `id` fields filled in.
        """
        if not chunks:
            return []

        with SessionLocal() as db:
            orm_chunks = []
            for c in chunks:
                orm_chunk = ContentChunk(
                    document_id=doc_id,
                    text=c["text"],
                    subtopic_id=c.get("subtopic_id"),
                    page_number=c.get("page_number"),
                    source_type=c.get("source_type", "pdf"),
                    source_url=c.get("source_url"),
                )
                db.add(orm_chunk)
                orm_chunks.append(orm_chunk)

            db.commit()
            # Refresh to get server-assigned IDs
            for orm_chunk in orm_chunks:
                db.refresh(orm_chunk)

            return [_chunk_to_dict(c) for c in orm_chunks]

    def get_by_subtopics(self, subtopic_ids: List[int]) -> List[dict]:
        """Fetch all chunks belonging to the given subtopic IDs."""
        if not subtopic_ids:
            return []
        with SessionLocal() as db:
            chunks = db.query(ContentChunk).filter(
                ContentChunk.subtopic_id.in_(subtopic_ids)
            ).all()
            return [_chunk_to_dict(c) for c in chunks]

    def get_by_document(self, doc_id: str) -> List[dict]:
        with SessionLocal() as db:
            chunks = db.query(ContentChunk).filter(ContentChunk.document_id == doc_id).all()
            return [_chunk_to_dict(c) for c in chunks]

    def get_by_id(self, chunk_id: int) -> Optional[dict]:
        with SessionLocal() as db:
            chunk = db.query(ContentChunk).filter(ContentChunk.id == chunk_id).first()
            return _chunk_to_dict(chunk) if chunk else None
