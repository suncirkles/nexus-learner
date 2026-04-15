"""
tests/unit/repos/test_pgvector_store.py
---------------------------------------
Unit tests for PGVectorStore with mocked dependencies.
Verifies upsert and search logic without requiring a live Postgres instance.
"""

import pytest
from unittest.mock import MagicMock, patch, call


@pytest.fixture
def mock_embeddings():
    emb = MagicMock()
    emb.embed_documents = MagicMock(return_value=[[0.1] * 384])
    emb.embed_query = MagicMock(return_value=[0.1] * 384)
    return emb


@pytest.fixture
def pg_store(mock_embeddings):
    """PGVectorStore with embeddings pre-wired to mocks."""
    from repositories.vector.pgvector_store import PGVectorStore
    store = PGVectorStore()
    store._embeddings = mock_embeddings
    store._collection_name = "nexus_vectors"
    return store


class TestPGVectorStoreUpsert:
    def test_upsert_calls_add_documents(self, pg_store):
        chunks = [
            {"text": "chunk one", "metadata": {"document_id": "doc-1", "db_chunk_id": 1}},
            {"text": "chunk two", "metadata": {"document_id": "doc-1", "db_chunk_id": 2}},
        ]

        mock_store = MagicMock()
        with patch("repositories.vector.pgvector_store.PGVectorStore._get_store", return_value=mock_store):
            pg_store.upsert_chunks(chunks)

        mock_store.add_documents.assert_called_once()
        lc_docs = mock_store.add_documents.call_args[0][0]
        assert len(lc_docs) == 2
        assert lc_docs[0].page_content == "chunk one"
        assert lc_docs[0].metadata["db_chunk_id"] == 1

    def test_upsert_empty_list_is_noop(self, pg_store):
        mock_store = MagicMock()
        with patch("repositories.vector.pgvector_store.PGVectorStore._get_store", return_value=mock_store):
            pg_store.upsert_chunks([])
        mock_store.add_documents.assert_not_called()


class TestPGVectorStoreSearch:
    def test_search_delegates_to_similarity_search(self, pg_store):
        mock_store = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "result text"
        mock_doc.metadata = {"doc_id": "123"}
        mock_store.similarity_search_with_score.return_value = [(mock_doc, 0.95)]

        with patch("repositories.vector.pgvector_store.PGVectorStore._get_store", return_value=mock_store):
            results = pg_store.search("test query", top_k=5, filter_doc_id="doc-abc")

        mock_store.similarity_search_with_score.assert_called_once_with(
            "test query", k=5, filter={"document_id": "doc-abc"}
        )
        assert len(results) == 1
        assert results[0]["text"] == "result text"
        assert results[0]["score"] == 0.95


class TestPGVectorStoreGeneral:
    def test_factory_returns_correct_store(self):
        from repositories.vector.factory import get_vector_store
        from repositories.vector.pgvector_store import PGVectorStore
        from repositories.vector.qdrant_store import QdrantStore
        
        with patch("core.config.settings.VECTOR_STORE_TYPE", "pgvector"):
            store = get_vector_store()
            assert isinstance(store, PGVectorStore)
            
        with patch("core.config.settings.VECTOR_STORE_TYPE", "qdrant"):
            store = get_vector_store()
            assert isinstance(store, QdrantStore)
