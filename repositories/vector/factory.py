import logging
from core.config import settings
from repositories.protocols import VectorStoreProtocol

logger = logging.getLogger(__name__)


def get_vector_store() -> VectorStoreProtocol:
    """Returns the configured VectorStoreProvider instance."""
    store_type = settings.VECTOR_STORE_TYPE.lower()
    
    if store_type == "pgvector":
        from repositories.vector.pgvector_store import PGVectorStore
        logger.debug("Instantiating PGVectorStore")
        return PGVectorStore()
    
    # Default to Qdrant
    from repositories.vector.qdrant_store import QdrantStore
    if store_type != "qdrant":
        logger.warning(
            "Unknown VECTOR_STORE_TYPE '%s'. Defaulting to Qdrant.", 
            settings.VECTOR_STORE_TYPE
        )
    logger.debug("Instantiating QdrantStore")
    return QdrantStore()
