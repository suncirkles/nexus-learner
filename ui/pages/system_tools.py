"""
ui/pages/system_tools.py
--------------------------
System Tools page — administrative controls (reset, edit subjects/topics).
Moved verbatim from app.py::render_system_tools() — zero behaviour change.
"""

import time
import logging
import streamlit as st

from sqlalchemy import delete as sa_delete
from core.database import SessionLocal, Subject, Flashcard, Document as DBDocument, SubjectDocumentAssociation
from core.config import settings

logger = logging.getLogger(__name__)


def _get_qdrant_client():
    try:
        from qdrant_client import QdrantClient
        return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    except Exception as e:
        logger.warning("Could not create Qdrant client: %s", e)
        return None


def render_system_tools():
    st.header("⚙️ Administrative Controls")
    try:
        db = SessionLocal()
        from core.database import Document as DBDocument

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
                    _reset_entire_system()
                else:
                    st.error("Incorrect confirmation text. Type exactly: RESET")
            if col_cancel.button("Cancel"):
                st.session_state.confirm_reset = False
                st.rerun()

        st.divider()
        st.divider()

        sys_tabs = st.tabs(["🟢 Active Subjects", "📦 Archived Subjects", "📊 Observability Metrics"])

        with sys_tabs[0]:
            st.subheader("Manage Active Subjects & Topics")
            active_subjects = db.query(Subject).filter(Subject.is_archived == False).all()

            if not active_subjects:
                st.info("No active subjects.")

            for subj in active_subjects:
                with st.container(border=True):
                    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
                    new_subj_name = col1.text_input(f"Edit Subject Name", value=subj.name, key=f"edit_subj_{subj.id}")
                    if col2.button("Update Name", key=f"upd_subj_{subj.id}"):
                        subj.name = new_subj_name
                        db.commit()
                        st.success("Subject updated!")
                        st.rerun()
                    if col3.button("📦 Archive Subject", key=f"arch_subj_{subj.id}"):
                        subj.is_archived = True
                        db.commit()
                        st.success(f"Archived '{subj.name}'.")
                        st.rerun()

                    doc_assocs = db.query(SubjectDocumentAssociation).filter(SubjectDocumentAssociation.subject_id == subj.id).all()
                    if doc_assocs:
                        st.markdown("**(Attached Documents)**")
                    for assoc in doc_assocs:
                        doc = db.query(DBDocument).filter(DBDocument.id == assoc.document_id).first()
                        if doc:
                            dcol1, dcol2 = st.columns([0.8, 0.2])
                            dcol1.write(f"📄 {doc.filename}")
                            if dcol2.button("Detach", key=f"det_sys_{subj.id}_{doc.id}"):
                                db.delete(assoc)
                                db.commit()
                                st.rerun()

        with sys_tabs[1]:
            st.subheader("Manage Archived Subjects")
            archived_subjects = db.query(Subject).filter(Subject.is_archived == True).all()

            if not archived_subjects:
                st.info("No archived subjects.")

            for subj in archived_subjects:
                with st.container(border=True):
                    st.markdown(f"**{subj.name}**")
                    col1, col2 = st.columns(2)

                    if col1.button("♻️ Restore Subject", key=f"rest_subj_{subj.id}"):
                        subj.is_archived = False
                        db.commit()
                        st.success(f"Restored '{subj.name}'.")
                        st.rerun()

                    if col2.button("🚨 Permanently Delete", type="primary", key=f"perm_del_{subj.id}"):
                        db.query(Flashcard).filter(Flashcard.subject_id == subj.id).delete()
                        db.query(SubjectDocumentAssociation).filter(SubjectDocumentAssociation.subject_id == subj.id).delete()
                        db.delete(subj)
                        db.commit()
                        st.success(f"Permanently deleted '{subj.name}'.")
                        st.rerun()

        with sys_tabs[2]:
            st.subheader("📊 RAG Observability Metrics")
            st.info("Metrics tracked per document upload/web research session.")

            all_docs = db.query(DBDocument).order_by(DBDocument.created_at.desc()).all()

            if not all_docs:
                st.info("No processing metrics available yet.")
            else:
                doc_subjects_raw = (
                    db.query(SubjectDocumentAssociation.document_id, Subject.name)
                    .join(Subject, Subject.id == SubjectDocumentAssociation.subject_id)
                    .all()
                )
                doc_subject_map: dict = {}
                for doc_id_key, subj_name in doc_subjects_raw:
                    doc_subject_map.setdefault(doc_id_key, []).append(subj_name)

                for doc in all_docs:
                    with st.container(border=True):
                        st.markdown(f"**📄 {doc.filename}**")
                        subj_names = doc_subject_map.get(doc.id, [])
                        st.caption(f"Shared in: {', '.join(subj_names) if subj_names else 'Library Only'} | Date: {doc.created_at.strftime('%Y-%m-%d %H:%M')}")

                        mcol1, mcol2, mcol3 = st.columns(3)
                        mcol1.metric("Relevance Rate", f"{doc.relevance_rate}%")
                        mcol2.metric("Yield Rate", f"{doc.yield_rate/10:.1f} cards/chunk")
                        mcol3.metric("Grounding (Faithfulness)", f"{doc.faithfulness_score}/5")

                        st.progress(doc.relevance_rate / 100, text="Filtering efficiency")

    finally:
        db.close()


def _reset_entire_system():
    """Wipes SQLite and Qdrant Collection."""
    from core.database import Base, engine
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    try:
        qdrant = _get_qdrant_client()
        if qdrant:
            qdrant.delete_collection(collection_name=settings.QDRANT_COLLECTION_NAME)
    except Exception as e:
        st.error(f"Failed to clear Qdrant: {e}")

    st.success("System completely reset!")
    time.sleep(1)
    st.rerun()
