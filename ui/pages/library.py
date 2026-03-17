"""
ui/pages/library.py
---------------------
Knowledge Library page — centralized document management.
Moved verbatim from app.py::render_knowledge_library() — zero behaviour change.
"""

import os
import uuid
import logging
import streamlit as st

from core.database import SessionLocal, Document as LibDoc, Topic as LibTopic
from core.config import settings

logger = logging.getLogger(__name__)


def _get_qdrant_client():
    """Lazy-init Qdrant client (reuses app-level cache if available)."""
    try:
        from qdrant_client import QdrantClient
        return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    except Exception as e:
        logger.warning("Could not create Qdrant client: %s", e)
        return None


def render_knowledge_library():
    """Centralized management for indexed documents."""
    st.header("📂 Knowledge Library")
    st.markdown("Upload and index documents once to make them available across all subjects.")

    db = SessionLocal()
    try:
        col_main, col_sidebar = st.columns([0.7, 0.3])

        with col_main:
            st.subheader("📑 Processed Documents")
            all_docs = db.query(LibDoc).order_by(LibDoc.created_at.desc()).all()
            if not all_docs:
                st.info("The library is empty. Upload a document to start.")
            else:
                for doc in all_docs:
                    with st.expander(f"📄 {doc.title or doc.filename}"):
                        st.caption(f"Indexed on: {doc.created_at.strftime('%Y-%m-%d %H:%M')}")
                        topics = db.query(LibTopic).filter(LibTopic.document_id == doc.id).all()
                        if topics:
                            st.markdown(f"**Topics Discovered ({len(topics)}):**")
                            st.write(", ".join([t.name for t in topics]))

                        if st.button("🗑️ Delete from Library", key=f"del_lib_{doc.id}"):
                            # H1: delete Qdrant vectors BEFORE the DB record so no
                            # orphaned embeddings linger in the vector store.
                            try:
                                from qdrant_client.http import models as rest
                                qdrant = _get_qdrant_client()
                                if qdrant:
                                    qdrant.delete(
                                        collection_name=settings.QDRANT_COLLECTION_NAME,
                                        points_selector=rest.FilterSelector(
                                            filter=rest.Filter(
                                                must=[rest.FieldCondition(
                                                    key="document_id",
                                                    match=rest.MatchValue(value=doc.id),
                                                )]
                                            )
                                        ),
                                    )
                            except Exception as qe:
                                logger.warning("Qdrant cleanup failed for doc %s: %s", doc.id, qe)
                                st.warning(f"Vector cleanup failed ({qe}) — DB record will still be removed.")
                            db.delete(doc)
                            db.commit()
                            st.rerun()

        with col_sidebar:
            st.subheader("📥 Upload & Index")
            uploaded_file = st.file_uploader("Index new PDF", type=["pdf"], key="lib_uploader")
            if uploaded_file:
                if st.button("🚀 Start Global Indexing", key="lib_index_btn"):
                    doc_id = str(uuid.uuid4())
                    safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in " ._-").strip()
                    os.makedirs("temp_uploads", exist_ok=True)
                    file_path = os.path.join("temp_uploads", f"{doc_id}_{safe_filename}")
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    state = {
                        "mode": "INDEXING",
                        "file_path": file_path,
                        "doc_id": doc_id,
                        "subject_id": None,
                        "chunks": [],
                        "current_page": 0,
                        "current_chunk_index": 0,
                        "hierarchy": [],
                        "status_message": "Initializing global indexing...",
                    }
                    from core.background import start_background_task
                    start_background_task(state, doc_id, filename=uploaded_file.name)
                    st.success("Global indexing started!")
                    st.rerun()

    finally:
        db.close()
