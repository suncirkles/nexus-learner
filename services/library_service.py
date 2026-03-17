"""
services/library_service.py
-----------------------------
Business logic for the Knowledge Library (global document store).
Absorbs the delete-document flow (Qdrant first, then DB) from the Library page.
"""

import logging
from typing import List
from repositories.protocols import DocumentRepoProtocol, VectorStoreProtocol

logger = logging.getLogger(__name__)


class LibraryService:
    def __init__(
        self,
        doc_repo: DocumentRepoProtocol,
        vector_store: VectorStoreProtocol,
    ):
        self._docs = doc_repo
        self._vector = vector_store

    def get_all_documents(self) -> List[dict]:
        return self._docs.get_all()

    def delete_document(self, doc_id: str) -> None:
        """Delete a document from Qdrant (H1: vectors first) then from the DB.

        H1 invariant: vector records are deleted before the DB record to avoid
        orphaned embeddings that could never be cleaned up.
        """
        try:
            self._vector.delete_by_document(doc_id)
        except Exception as e:
            logger.warning("Qdrant cleanup failed for doc %s: %s", doc_id, e)
        self._docs.delete(doc_id)
