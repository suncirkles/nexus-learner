"""
ui/pages/system_tools.py
--------------------------
System Tools page — administrative controls (reset, edit subjects/topics).
Moved verbatim from app.py::render_system_tools() — zero behaviour change.
"""

import time
import logging
import streamlit as st

from ui import api_client

logger = logging.getLogger(__name__)


def render_system_tools():
    st.header("⚙️ Administrative Controls")

    st.warning("These actions are destructive and cannot be undone.")

    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False

    if not st.session_state.confirm_reset:
        if st.button("🚨 Global Reset: Wipe Database & Qdrant Collections"):
            st.session_state.confirm_reset = True
            st.rerun()
    else:
        st.error("⚠️ This will permanently delete ALL subjects, documents, and flashcards.")
        confirm_text = st.text_input(
            "Type **RESET** to confirm:",
            key="reset_confirm_input",
            placeholder="RESET",
        )
        col_ok, col_cancel = st.columns([0.2, 0.8])
        if col_ok.button("Confirm Reset", type="primary"):
            if confirm_text.strip() == "RESET":
                st.session_state.confirm_reset = False
                try:
                    api_client.reset_system()
                    st.success("System completely reset!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Reset failed: {e}")
            else:
                st.error("Incorrect confirmation text. Type exactly: RESET")
        if col_cancel.button("Cancel"):
            st.session_state.confirm_reset = False
            st.rerun()

    st.divider()

    sys_tabs = st.tabs(["🟢 Active Subjects", "📦 Archived Subjects", "📊 Observability Metrics"])

    with sys_tabs[0]:
        st.subheader("Manage Active Subjects & Topics")
        active_subjects = api_client.list_active_subjects()

        if not active_subjects:
            st.info("No active subjects.")

        for subj in active_subjects:
            with st.container(border=True):
                col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
                new_subj_name = col1.text_input("Edit Subject Name", value=subj["name"], key=f"edit_subj_{subj['id']}")
                if col2.button("Update Name", key=f"upd_subj_{subj['id']}"):
                    try:
                        api_client.rename_subject(subj["id"], new_subj_name)
                        st.success("Subject updated!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Rename failed: {e}")
                if col3.button("📦 Archive Subject", key=f"arch_subj_{subj['id']}"):
                    try:
                        api_client.archive_subject(subj["id"])
                        st.success(f"Archived '{subj['name']}'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Archive failed: {e}")

                doc_assocs = api_client.list_attached_documents(subj["id"])
                if doc_assocs:
                    st.markdown("**(Attached Documents)**")
                for doc in doc_assocs:
                    dcol1, dcol2 = st.columns([0.8, 0.2])
                    dcol1.write(f"📄 {doc.get('filename', doc.get('id'))}")
                    if dcol2.button("Detach", key=f"det_sys_{subj['id']}_{doc['id']}"):
                        try:
                            api_client.detach_document(subj["id"], doc["id"])
                            st.rerun()
                        except Exception as e:
                            st.error(f"Detach failed: {e}")

    with sys_tabs[1]:
        st.subheader("Manage Archived Subjects")
        archived_subjects = api_client.list_archived_subjects()

        if not archived_subjects:
            st.info("No archived subjects.")

        for subj in archived_subjects:
            with st.container(border=True):
                st.markdown(f"**{subj['name']}**")
                col1, col2 = st.columns(2)

                if col1.button("♻️ Restore Subject", key=f"rest_subj_{subj['id']}"):
                    try:
                        api_client.restore_subject(subj["id"])
                        st.success(f"Restored '{subj['name']}'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Restore failed: {e}")

                if col2.button("🚨 Permanently Delete", type="primary", key=f"perm_del_{subj['id']}"):
                    try:
                        api_client.delete_subject(subj["id"])
                        st.success(f"Permanently deleted '{subj['name']}'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

    with sys_tabs[2]:
        st.subheader("📊 RAG Observability Metrics")
        st.info("Metrics tracked per document upload/web research session.")

        all_docs = api_client.list_documents()

        if not all_docs:
            st.info("No processing metrics available yet.")
        else:
            # Build doc_id → subject_names map via attached-documents per subject
            active_subjects = api_client.list_active_subjects()
            doc_subject_map: dict = {}
            for subj in active_subjects:
                for doc in api_client.list_attached_documents(subj["id"]):
                    doc_subject_map.setdefault(doc["id"], []).append(subj["name"])

            for doc in all_docs:
                with st.container(border=True):
                    st.markdown(f"**📄 {doc.get('filename')}**")
                    subj_names = doc_subject_map.get(doc["id"], [])
                    created_at = doc.get("created_at", "")
                    date_str = created_at[:16].replace("T", " ") if created_at else ""
                    st.caption(f"Shared in: {', '.join(subj_names) if subj_names else 'Library Only'} | Date: {date_str}")

                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric("Relevance Rate", f"{doc.get('relevance_rate') or 0:.0f}%")
                    yield_rate = doc.get("yield_rate") or 0
                    mcol2.metric("Yield Rate", f"{yield_rate/10:.1f} cards/chunk")
                    mcol3.metric("Grounding (Faithfulness)", f"{doc.get('faithfulness_score') or 0}/5")

                    st.progress((doc.get("relevance_rate") or 0) / 100, text="Filtering efficiency")
