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
        """Wipe all SQLite tables and the Qdrant collection.

        SQLite: drop_all + create_all via SQLAlchemy metadata (no Alembic yet).
        Qdrant: drop_collection via VectorStoreProtocol so no direct client here.
        """
        from core.database import Base, engine
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("SQLite tables recreated.")

        try:
            self._vector.drop_collection()
        except Exception as e:
            logger.warning("Qdrant drop_collection failed during reset: %s", e)
            raise
