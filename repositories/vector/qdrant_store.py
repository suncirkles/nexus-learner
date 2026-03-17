"""
repositories/vector/qdrant_store.py
-------------------------------------
Unified Qdrant vector-store wrapper.
Consolidates the three QdrantVectorStore.from_documents() call sites
(agents/ingestion.py, workflows/phase1_ingestion.py, workflows/phase2_web_ingestion.py)
into a single place.

Also absorbs the direct QdrantClient.delete_collection() call that was
previously in core/database.reset_database() and app.py.
"""

import logging
from typing import List, Optional

from core.config import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)


def _make_embeddings():
    """Return (embeddings_instance, collection_name).

    Mirrors agents/ingestion.py::_make_embeddings() exactly so existing
    code paths remain unchanged during Phase 1.  In Phase 4 this becomes
    the single canonical location.
    """
    from core.config import settings as _s

    def _is_real_openai_key(key: str) -> bool:
        return bool(key) and key.startswith("sk-") and len(key) > 20

    use_hf = _s.EMBEDDING_PROVIDER.lower() == "huggingface"

    if not use_hf:
        if _is_real_openai_key(_s.OPENAI_API_KEY):
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(api_key=_s.OPENAI_API_KEY), _s.QDRANT_COLLECTION_NAME
        logger.warning(
            "EMBEDDING_PROVIDER=openai but key is absent/invalid — "
            "falling back to HuggingFace embeddings"
        )

    try:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        logger.info("Using HuggingFace all-MiniLM-L6-v2 embeddings (384 dims)")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        collection = _s.QDRANT_COLLECTION_NAME + "_hf"
        return embeddings, collection
    except Exception as e:
        raise ValueError(
            "HuggingFace embeddings unavailable. "
            "Install sentence-transformers: pip install sentence-transformers. "
            f"Original error: {e}"
        )


class QdrantStore:
    """Unified interface for all Qdrant vector-store operations.

    Lazily initialises embeddings on first use so importing this module
    does not trigger model loading at import time.
    """

    def __init__(self):
        self._embeddings = None
        self._collection_name: Optional[str] = None

    def _init_embeddings(self):
        if self._embeddings is None:
            self._embeddings, self._collection_name = _make_embeddings()

    @property
    def collection_name(self) -> str:
        self._init_embeddings()
        return self._collection_name  # type: ignore[return-value]

    @property
    def embeddings(self):
        self._init_embeddings()
        return self._embeddings

    def upsert_chunks(self, chunks: List[dict]) -> None:
        """Upsert a batch of chunks into Qdrant.

        Args:
            chunks: list of {"text": str, "metadata": dict}
        """
        if not chunks:
            return

        self._init_embeddings()

        from langchain_core.documents import Document as LCDoc

        lc_docs = [
            LCDoc(page_content=c["text"], metadata=c.get("metadata", {}))
            for c in chunks
        ]

        try:
            QdrantVectorStore.from_documents(
                lc_docs,
                self._embeddings,
                url=settings.QDRANT_URL,
                collection_name=self._collection_name,
            )
            logger.debug("Upserted %d chunk(s) to Qdrant collection '%s'", len(chunks), self._collection_name)
        except Exception as e:
            logger.error("Qdrant upsert failed (%d docs): %s", len(chunks), e, exc_info=True)
            raise

    def delete_by_document(self, document_id: str) -> None:
        """Delete all vectors for a given document_id filter."""
        try:
            client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            self._init_embeddings()
            collection = self._collection_name or settings.QDRANT_COLLECTION_NAME

            if client.collection_exists(collection):
                client.delete(
                    collection_name=collection,
                    points_selector=rest.FilterSelector(
                        filter=rest.Filter(
                            must=[
                                rest.FieldCondition(
                                    key="document_id",
                                    match=rest.MatchValue(value=document_id),
                                )
                            ]
                        )
                    ),
                )
                logger.info("Deleted Qdrant vectors for document_id=%s", document_id)
        except Exception as e:
            logger.warning("Qdrant delete_by_document failed for %s: %s", document_id, e)
            raise

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_doc_id: Optional[str] = None,
    ) -> List[dict]:
        """Semantic search returning list of {"text": str, "metadata": dict, "score": float}."""
        self._init_embeddings()

        client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        qdrant_filter = None
        if filter_doc_id:
            qdrant_filter = rest.Filter(
                must=[rest.FieldCondition(key="document_id", match=rest.MatchValue(value=filter_doc_id))]
            )

        store = QdrantVectorStore(
            client=client,
            collection_name=self._collection_name,
            embedding=self._embeddings,
        )
        results = store.similarity_search_with_score(query, k=top_k, filter=qdrant_filter)
        return [
            {"text": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in results
        ]

    def drop_collection(self) -> None:
        """Delete the entire Qdrant collection (used by system reset)."""
        try:
            self._init_embeddings()
            collection = self._collection_name or settings.QDRANT_COLLECTION_NAME

            client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            if client.collection_exists(collection):
                client.delete_collection(collection)
                logger.info("Dropped Qdrant collection: %s", collection)
        except Exception as e:
            logger.warning("Qdrant drop_collection failed: %s", e)
            raise
