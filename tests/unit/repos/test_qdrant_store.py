"""
tests/unit/repos/test_qdrant_store.py
---------------------------------------
Unit tests for QdrantStore with mocked QdrantClient.
Verifies upsert payload structure and delete filter construction
without requiring a live Qdrant instance.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from langchain_core.documents import Document as LCDoc


@pytest.fixture
def mock_embeddings():
    emb = MagicMock()
    emb.embed_documents = MagicMock(return_value=[[0.1] * 384])
    emb.embed_query = MagicMock(return_value=[0.1] * 384)
    return emb


@pytest.fixture
def qdrant_store(mock_embeddings):
    """QdrantStore with embeddings pre-wired to mocks."""
    from repositories.vector.qdrant_store import QdrantStore
    store = QdrantStore()
    store._embeddings = mock_embeddings
    store._collection_name = "test_collection"
    return store


class TestQdrantStoreUpsert:
    def test_upsert_calls_from_documents(self, qdrant_store, mock_embeddings):
        chunks = [
            {"text": "chunk one", "metadata": {"document_id": "doc-1", "db_chunk_id": 1}},
            {"text": "chunk two", "metadata": {"document_id": "doc-1", "db_chunk_id": 2}},
        ]

        with patch("repositories.vector.qdrant_store.QdrantVectorStore") as mock_vs_cls:
            qdrant_store.upsert_chunks(chunks)

        mock_vs_cls.from_documents.assert_called_once()
        call_args = mock_vs_cls.from_documents.call_args
        lc_docs = call_args[0][0]
        assert len(lc_docs) == 2
        assert lc_docs[0].page_content == "chunk one"
        assert lc_docs[0].metadata["db_chunk_id"] == 1

    def test_upsert_empty_list_is_noop(self, qdrant_store):
        with patch("repositories.vector.qdrant_store.QdrantVectorStore") as mock_vs_cls:
            qdrant_store.upsert_chunks([])
        mock_vs_cls.from_documents.assert_not_called()

    def test_upsert_uses_correct_collection_name(self, qdrant_store):
        chunks = [{"text": "test", "metadata": {}}]
        with patch("repositories.vector.qdrant_store.QdrantVectorStore") as mock_vs_cls:
            qdrant_store.upsert_chunks(chunks)

        kwargs = mock_vs_cls.from_documents.call_args[1]
        assert kwargs["collection_name"] == "test_collection"

    def test_upsert_raises_on_qdrant_failure(self, qdrant_store):
        chunks = [{"text": "test", "metadata": {}}]
        with patch("repositories.vector.qdrant_store.QdrantVectorStore") as mock_vs_cls:
            mock_vs_cls.from_documents.side_effect = Exception("Connection refused")
            with pytest.raises(Exception, match="Connection refused"):
                qdrant_store.upsert_chunks(chunks)


class TestQdrantStoreDeleteByDocument:
    def test_delete_constructs_correct_filter(self, qdrant_store):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True

        with patch("repositories.vector.qdrant_store.QdrantClient", return_value=mock_client):
            qdrant_store.delete_by_document("doc-abc")

        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert call_args[1]["collection_name"] == "test_collection"

        # Verify the filter payload contains the document_id
        points_selector = call_args[1]["points_selector"]
        # Access the nested filter structure
        must_conditions = points_selector.filter.must
        assert len(must_conditions) == 1
        assert must_conditions[0].key == "document_id"
        assert must_conditions[0].match.value == "doc-abc"

    def test_delete_skips_when_collection_absent(self, qdrant_store):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False

        with patch("repositories.vector.qdrant_store.QdrantClient", return_value=mock_client):
            qdrant_store.delete_by_document("doc-xyz")

        mock_client.delete.assert_not_called()


class TestQdrantStoreDropCollection:
    def test_drop_calls_delete_collection(self, qdrant_store):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True

        with patch("repositories.vector.qdrant_store.QdrantClient", return_value=mock_client):
            qdrant_store.drop_collection()

        mock_client.delete_collection.assert_called_once_with("test_collection")

    def test_drop_skips_when_collection_absent(self, qdrant_store):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False

        with patch("repositories.vector.qdrant_store.QdrantClient", return_value=mock_client):
            qdrant_store.drop_collection()

        mock_client.delete_collection.assert_not_called()
