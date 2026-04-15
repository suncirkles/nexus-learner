"""
repositories/vector/pgvector_store.py
---------------------------------------
PGVector implementation of VectorStoreProtocol.
Uses langchain-postgres to store and search vector embeddings in Supabase/PostgreSQL.
"""

import logging
from typing import List, Optional, Dict, Any

from core.config import settings
from repositories.protocols import VectorStoreProtocol

logger = logging.getLogger(__name__)


def _make_embeddings():
    """Return (embeddings_instance, collection_name).

    Tries OpenAI first; falls back to local FastEmbed (all-MiniLM-L6-v2, 384 dims).
    The HuggingFace fallback appends '_hf' to the collection name so OpenAI and
    HuggingFace vectors never share the same table.
    """
    def _is_real_openai_key(key: str) -> bool:
        return bool(key) and key.startswith("sk-") and len(key) > 20

    use_hf = settings.EMBEDDING_PROVIDER.lower() == "huggingface"

    if not use_hf:
        if _is_real_openai_key(settings.OPENAI_API_KEY):
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY), settings.PGVECTOR_COLLECTION_NAME
        logger.warning(
            "EMBEDDING_PROVIDER=openai but key is absent/invalid — "
            "falling back to FastEmbed embeddings"
        )

    try:
        from core.embeddings import FastEmbedEmbeddings
        logger.info("Using FastEmbed all-MiniLM-L6-v2 embeddings (384 dims, ONNX)")
        embeddings = FastEmbedEmbeddings()
        collection = settings.PGVECTOR_COLLECTION_NAME + "_hf"
        return embeddings, collection
    except Exception as e:
        raise ValueError(
            "FastEmbed embeddings unavailable. "
            "Install with: pip install fastembed. "
            f"Original error: {e}"
        )


class PGVectorStore(VectorStoreProtocol):
    """PGVector implementation using the same interface as the old QdrantStore."""

    def __init__(self):
        self._embeddings = None
        self._collection_name: Optional[str] = None
        self._store = None

    def _init_embeddings(self):
        if self._embeddings is None:
            self._embeddings, self._collection_name = _make_embeddings()

    def _get_store(self):
        self._init_embeddings()
        if self._store is None:
            from langchain_postgres import PGVector
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
        """Delete all vectors for a given document_id."""
        try:
            store = self._get_store()
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

        filter_dict = {"document_id": filter_doc_id} if filter_doc_id else None

        results = store.similarity_search_with_score(query, k=top_k, filter=filter_dict)
        return [
            {"text": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in results
        ]

    def drop_collection(self) -> None:
        """Drop the pgvector tables for this collection (used by system reset)."""
        try:
            store = self._get_store()
            store.drop_tables()
            logger.info("Dropped PGVector tables for collection: %s", self._collection_name)
        except Exception as e:
            logger.warning("PGVector drop_collection failed: %s", e)
            raise
