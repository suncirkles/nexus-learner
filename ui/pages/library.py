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

from ui import api_client

logger = logging.getLogger(__name__)


def render_knowledge_library():
    """Centralized management for indexed documents."""
    st.header("📂 Knowledge Library")
    st.markdown("Upload and index documents once to make them available across all subjects.")

    col_main, col_sidebar = st.columns([0.7, 0.3])

    with col_main:
        st.subheader("📑 Processed Documents")
        all_docs = api_client.list_documents()
        if not all_docs:
            st.info("The library is empty. Upload a document to start.")
        else:
            for doc in all_docs:
                with st.expander(f"📄 {doc.get('title') or doc.get('filename')}"):
                    created_at = doc.get("created_at", "")
                    if created_at:
                        # created_at comes as ISO string from API
                        from datetime import datetime
                        try:
                            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                            st.caption(f"Indexed on: {dt.strftime('%Y-%m-%d %H:%M')}")
                        except Exception:
                            st.caption(f"Indexed on: {created_at}")

                    topics = api_client.get_topics_by_document(doc["id"])
                    if topics:
                        st.markdown(f"**Topics Discovered ({len(topics)}):**")
                        st.write(", ".join([t["name"] for t in topics]))

                    if st.button("🗑️ Delete from Library", key=f"del_lib_{doc['id']}"):
                        # LibraryService handles H1: Qdrant vectors deleted before DB record.
                        try:
                            api_client.delete_document(doc["id"])
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
                        else:
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
