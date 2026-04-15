"""
repositories/vector/pgvector_store.py
---------------------------------------
PGVector implementation of VectorStoreProvider.
Uses langchain-postgres to store and search vector embeddings in Supabase/PostgreSQL.
"""

import logging
from typing import List, Optional, Dict, Any

from core.config import settings
from repositories.protocols import VectorStoreProtocol
from repositories.vector.qdrant_store import _make_embeddings

logger = logging.getLogger(__name__)


class PGVectorStore(VectorStoreProtocol):
    """PGVector implementation using the same interface as QdrantStore."""

    def __init__(self):
        self._embeddings = None
        self._collection_name: Optional[str] = None
        self._store = None

    def _init_embeddings(self):
        if self._embeddings is None:
            # We reuse the same logic for provider fallback (OpenAI -> HF)
            self._embeddings, _ = _make_embeddings()
            # Use specific PGVector collection name from settings
            self._collection_name = settings.PGVECTOR_COLLECTION_NAME

    def _get_store(self):
        self._init_embeddings()
        if self._store is None:
            from langchain_postgres import PGVector
            
            # langchain-postgres uses a connection string
            self._store = PGVector(
                embeddings=self._embeddings,
                collection_name=self._collection_name,
                connection=settings.DB_URL,
                use_jsonb=True,
            )
        return self._store

    @property
    def collection_name(self) -> str:
        self._init_embeddings()
        return self._collection_name  # type: ignore[return-value]

    @property
    def embeddings(self):
        self._init_embeddings()
        return self._embeddings

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Upsert a batch of chunks into PostgreSQL."""
        if not chunks:
            return

        from langchain_core.documents import Document as LCDoc
        
        lc_docs = [
            LCDoc(page_content=c["text"], metadata=c.get("metadata", {}))
            for c in chunks
        ]

        try:
            store = self._get_store()
            store.add_documents(lc_docs)
            logger.debug("Upserted %d chunk(s) to PGVector table '%s'", len(chunks), self._collection_name)
        except Exception as e:
            logger.error("PGVector upsert failed (%d docs): %s", len(chunks), e, exc_info=True)
            raise

    def delete_by_document(self, document_id: str) -> None:
        """Delete all vectors for a given document_id filter."""
        try:
            store = self._get_store()
            # PGVector doesn't have a direct delete_by_metadata like QdrantClient
            # We use the underlying sync_connection to execute a DELETE or 
            # use store.delete() with IDs if we had them.
            # However, langchain-postgres supports delete by filtered collection.
            # For simplicity, we can use the SQL connection directly.
            
            from sqlalchemy import create_engine, text
            engine = create_engine(settings.DB_URL)
            with engine.connect() as conn:
                # langchain-postgres default table is langchain_pg_embedding
                # but we can check the implementation or use metadata filtering.
                # Actually, langchain_postgres.PGVector.delete(filter={"document_id": document_id})
                # is the intended way if IDs are not known.
                store.delete(filter={"document_id": document_id})
                
            logger.info("Deleted PGVector vectors for document_id=%s", document_id)
        except Exception as e:
            logger.warning("PGVector delete_by_document failed for %s: %s", document_id, e)
            raise

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search returning list of {"text": str, "metadata": dict, "score": float}."""
        store = self._get_store()
        
        filter_dict = None
        if filter_doc_id:
            filter_dict = {"document_id": filter_doc_id}

        # similarity_search_with_score returns (Document, score)
        results = store.similarity_search_with_score(query, k=top_k, filter=filter_dict)
        
        # Note: PGVector scores might have different range/meaning than Qdrant, 
        # but the interface remains consistent.
        return [
            {"text": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in results
        ]

    def drop_collection(self) -> None:
        """Clear all vectors from the pgvector table."""
        try:
            store = self._get_store()
            store.drop_tables() # Or just truncate
            logger.info("Dropped PGVector tables for collection: %s", self._collection_name)
        except Exception as e:
            logger.warning("PGVector drop_collection failed: %s", e)
            raise
