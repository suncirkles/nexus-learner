"""
ui/pages/library.py
---------------------
Knowledge Library page — centralized document management.
Moved verbatim from app.py::render_knowledge_library() — zero behaviour change.
"""

import logging
import streamlit as st

from ui import api_client
from ui.components.background_monitor import render_study_materials_background_monitor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB-polling progress monitor (Modal / multi-container deployments)
# Reads progress from the DB via the API — works across container boundaries.
# ---------------------------------------------------------------------------

@st.fragment(run_every=5)
def _api_job_monitor():
    """Polls /ingestion/status for each active Modal job stored in session state."""
    jobs: dict = st.session_state.get("active_modal_jobs", {})
    if not jobs:
        return

    st.divider()
    st.subheader("⚙️ Background Tasks")
    finished = []

    for job_id, meta in list(jobs.items()):
        # Skip API call for terminal jobs — use cached state from meta
        cached_status = meta.get("_terminal_status")
        if cached_status:
            s = meta.get("_terminal_payload", {})
            status = cached_status
        else:
            try:
                s = api_client.get_ingestion_status(job_id)
            except Exception as e:
                st.error(f"Could not fetch job status: {e}")
                continue
            status = s.get("status", "unknown")

        # Prefer the display name from session meta over the API's filename,
        # since GENERATION jobs have no file_path and the API returns "Unknown".
        api_filename = s.get("filename", "")
        filename = (
            meta.get("filename")
            if (not api_filename or api_filename == "Unknown")
            else api_filename
        ) or job_id[:8]
        mode = meta.get("mode", "INDEXING")

        with st.container(border=True):
            col_t, col_b = st.columns([0.85, 0.15])

            if status in ("queued", "indexing", "generating", "generation"):
                if mode == "GENERATION":
                    cc = s.get("current_chunk_index", 0)
                    tc = s.get("total_chunks", 0)
                    fc = s.get("flashcards_count", 0)
                    pct = min(max(cc / tc if tc > 0 else 0, 0.0), 1.0)
                    col_t.markdown(f"**🧠 Generating: {filename}**")
                    col_t.progress(pct, text=f"Chunk {cc}/{tc} · {fc} card(s) · {s.get('status_message', 'Processing...')}")
                else:
                    tp = s.get("total_pages", 1) or 1
                    cp = s.get("current_page", 0)
                    pct = min(max(cp / tp, 0.0), 1.0)
                    col_t.markdown(f"**⏳ Indexing: {filename}**")
                    col_t.progress(pct, text=f"Page {cp}/{tp} · {s.get('status_message', 'Processing...')}")
                if col_b.button("✖ Dismiss", key=f"api_dismiss_{job_id}"):
                    finished.append(job_id)

            elif status == "completed":
                # Cache terminal state so we stop hitting the API
                if not cached_status:
                    meta["_terminal_status"] = "completed"
                    meta["_terminal_payload"] = s
                    meta["_terminal_polls"] = 0
                fc = s.get("flashcards_count", 0)
                label = f"✅ **Done**: {filename}" + (f" — {fc} card(s)" if fc else "")
                col_t.success(label)
                # Auto-clear after 3 more display cycles (~15 s) so polling stops
                meta["_terminal_polls"] = meta.get("_terminal_polls", 0) + 1
                if meta["_terminal_polls"] >= 3:
                    finished.append(job_id)
                elif col_b.button("✖ Clear", key=f"api_clear_{job_id}"):
                    finished.append(job_id)

            elif status == "failed":
                if not cached_status:
                    meta["_terminal_status"] = "failed"
                    meta["_terminal_payload"] = s
                col_t.error(f"❌ **Failed**: {filename} — {s.get('error', 'Unknown error')}")
                if col_b.button("✖ Clear", key=f"api_clear_{job_id}"):
                    finished.append(job_id)

            else:
                col_t.info(f"⏳ {filename}: {status} — {s.get('status_message', '')}")

    for job_id in finished:
        st.session_state.active_modal_jobs.pop(job_id, None)
        # Clean up any terminal-poll counters
        st.session_state.pop(f"_term_polls_{job_id}", None)


def _render_api_job_monitor():
    """Mount the polling fragment only when there are active Modal jobs."""
    jobs = st.session_state.get("active_modal_jobs", {})
    if jobs:
        _api_job_monitor()


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
                _queued = False
                with st.spinner(f"Uploading {uploaded_file.name}…"):
                    try:
                        response = api_client.upload_and_spawn_ingestion(
                            file_bytes=uploaded_file.getvalue(),
                            filename=uploaded_file.name,
                            subject_id=None,
                        )
                        returned_job_id = response.get("job_id")
                        job_status = response.get("status", "")

                        if job_status == "failed":
                            st.error(f"Failed to start indexing: {response.get('error', 'Unknown error')}")
                        elif returned_job_id:
                            if "active_modal_jobs" not in st.session_state:
                                st.session_state.active_modal_jobs = {}
                            st.session_state.active_modal_jobs[returned_job_id] = {
                                "filename": uploaded_file.name,
                                "mode": "INDEXING",
                            }
                            _queued = True
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

                # Only rerun on success — keeps the error visible on failure
                if _queued:
                    st.rerun()

    # Progress monitor: DB-polling (Modal/multi-container) or in-memory (local dev)
    _render_api_job_monitor()
    render_study_materials_background_monitor()
