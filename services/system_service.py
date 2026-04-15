"""
services/system_service.py
----------------------------
System-level operations: global database + vector store reset.
Absorbs app.py::reset_entire_system() and ui/pages/system_tools.py::_reset_entire_system().
"""

import logging
from repositories.protocols import VectorStoreProtocol

logger = logging.getLogger(__name__)


class SystemService:
    def __init__(self, vector_store: VectorStoreProtocol):
        self._vector = vector_store

    def reset(self) -> None:
        """Wipe all relational tables and the vector store index.

        Database: drop_all + create_all via SQLAlchemy metadata.
        VectorStore: drop_collection via VectorStoreProtocol (best-effort).

        Vector store errors are logged but never re-raised — a stale vector
        index is recoverable, but raising here would leave the DB wiped with
        no tables and no way to recover without a manual schema re-init.
        """
        from core.database import Base, engine
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("Relational database tables recreated.")

        try:
            self._vector.drop_collection()
            logger.info("Vector store collection dropped.")
        except Exception as e:
            logger.warning(
                "Vector store drop_collection failed during reset (non-fatal): %s", e
            )
