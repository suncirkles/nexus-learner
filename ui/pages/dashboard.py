"""
ui/pages/dashboard.py
-----------------------
Dashboard page — subject tiles with stats and quick navigation.
Moved verbatim from app.py::render_dashboard() — zero behaviour change.
"""

import streamlit as st
from sqlalchemy import func, case, Integer

from core.database import SessionLocal, Subject, Topic, Flashcard, SubjectDocumentAssociation
from core.database import get_session


@st.cache_data(ttl=30)
def _get_global_flashcard_stats() -> tuple:
    """Returns (total, approved, pending, rejected) counts. Cached for 30 s."""
    with get_session() as db:
        total    = db.query(func.count(Flashcard.id)).scalar() or 0
        approved = db.query(func.count(Flashcard.id)).filter(Flashcard.status == "approved").scalar() or 0
        pending  = db.query(func.count(Flashcard.id)).filter(Flashcard.status == "pending").scalar() or 0
        rejected = db.query(func.count(Flashcard.id)).filter(Flashcard.status == "rejected").scalar() or 0
    return total, approved, pending, rejected


def render_dashboard():
    st.header("🏠 Agentic Learning Dashboard")
    try:
        db = SessionLocal()

        subjects = db.query(Subject).filter(Subject.is_archived == False).all()

        if not subjects:
            st.info("No subjects found. Head over to 'Study Materials' to define your first subject and upload documents!")
            return

        col_kb, col_stats = st.columns([0.7, 0.3])

        with col_kb:
            st.subheader("📚 My Subjects")

            topic_counts_raw = db.query(SubjectDocumentAssociation.subject_id, func.count(Topic.id)).\
                join(Topic, SubjectDocumentAssociation.document_id == Topic.document_id).\
                group_by(SubjectDocumentAssociation.subject_id).all()
            topic_counts = {r[0]: r[1] for r in topic_counts_raw}

            fc_stats_raw = db.query(
                Flashcard.subject_id,
                func.sum(case((Flashcard.status == 'approved', 1), else_=0)).cast(Integer),
                func.sum(case((Flashcard.status == 'pending', 1), else_=0)).cast(Integer)
            ).group_by(Flashcard.subject_id).all()

            fc_stats = {r[0]: {"approved": r[1] or 0, "pending": r[2] or 0} for r in fc_stats_raw}

            cols_per_row = 2
            for i in range(0, len(subjects), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(subjects):
                        subj = subjects[i + j]

                        topic_count = topic_counts.get(subj.id, 0)
                        stats = fc_stats.get(subj.id, {"approved": 0, "pending": 0})
                        approved_count = stats["approved"]
                        pending_count = stats["pending"]

                        with row_cols[j]:
                            st.markdown(f"""
                            <div class="subject-tile">
                                <div class="subject-title">📁 {subj.name}</div>
                                <div class="stat-row">
                                    <span class="stat-label">📘 Topics:</span>
                                    <span class="stat-value">{topic_count}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">✅ Approved Cards:</span>
                                    <span class="stat-value">{approved_count}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">⏳ Pending Review:</span>
                                    <span class="stat-value">{pending_count}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            if st.button(f"Start Learning {subj.name}", key=f"select_subj_{subj.id}"):
                                st.session_state.study_subject_id = subj.id
                                if "study_topic_id" in st.session_state: del st.session_state.study_topic_id
                                if "study_subtopic_id" in st.session_state: del st.session_state.study_subtopic_id
                                st.session_state.active_nav = "🧠 Learner"
                                st.session_state.sidebar_nav = "🧠 Learner"
                                st.rerun()

        with col_stats:
            st.subheader("Global Stats")
            total_q, approved_q, pending_q, rejected_q = _get_global_flashcard_stats()

            st.metric("Approved (Study Ready)", approved_q)
            st.metric("Pending Review", pending_q)
            st.metric("Review Bin (Rejected)", rejected_q)

            if total_q > 0:
                progress = approved_q / total_q
                st.progress(progress, text=f"{int(progress*100)}% Content Verified")
    finally:
        db.close()
