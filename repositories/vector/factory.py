import logging
from repositories.protocols import VectorStoreProtocol

logger = logging.getLogger(__name__)


def get_vector_store() -> VectorStoreProtocol:
    """Returns the configured VectorStoreProvider instance."""
    from repositories.vector.pgvector_store import PGVectorStore
    logger.debug("Instantiating PGVectorStore")
    return PGVectorStore()
