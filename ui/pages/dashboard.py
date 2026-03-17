"""
ui/pages/dashboard.py
-----------------------
Dashboard page — subject tiles with stats and quick navigation.
Moved verbatim from app.py::render_dashboard() — zero behaviour change.
"""

import streamlit as st

from ui import api_client


@st.cache_data(ttl=30)
def _get_dashboard_data() -> tuple:
    """Returns (subjects_data, topic_counts, fc_stats, global_stats). Cached for 30 s."""
    subjects = api_client.list_active_subjects()
    global_stats = api_client.get_global_stats()

    subjects_data = [(s["id"], s["name"]) for s in subjects]
    topic_counts: dict = {}
    fc_stats: dict = {}

    for s in subjects:
        sid = s["id"]
        topics = api_client.get_topics_by_subject(sid)
        topic_counts[sid] = len(topics)
        stats = api_client.get_flashcard_stats(sid)
        fc_stats[sid] = {
            "approved": stats.get("approved", 0),
            "pending": stats.get("pending", 0),
        }

    return subjects_data, topic_counts, fc_stats, global_stats


def render_dashboard():
    st.header("🏠 Agentic Learning Dashboard")

    subjects_data, topic_counts, fc_stats, global_stats = _get_dashboard_data()

    if not subjects_data:
        st.info("No subjects found. Head over to 'Study Materials' to define your first subject and upload documents!")
        return

    col_kb, col_stats = st.columns([0.7, 0.3])

    with col_kb:
        st.subheader("📚 My Subjects")

        cols_per_row = 2
        for i in range(0, len(subjects_data), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(subjects_data):
                    subj_id, subj_name = subjects_data[i + j]

                    topic_count = topic_counts.get(subj_id, 0)
                    stats = fc_stats.get(subj_id, {"approved": 0, "pending": 0})
                    approved_count = stats["approved"]
                    pending_count = stats["pending"]

                    with row_cols[j]:
                        st.markdown(f"""
                        <div class="subject-tile">
                            <div class="subject-title">📁 {subj_name}</div>
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

                        if st.button(f"Start Learning {subj_name}", key=f"select_subj_{subj_id}"):
                            st.session_state.study_subject_id = subj_id
                            if "study_topic_id" in st.session_state: del st.session_state.study_topic_id
                            if "study_subtopic_id" in st.session_state: del st.session_state.study_subtopic_id
                            st.session_state.active_nav = "🧠 Learner"
                            st.session_state.sidebar_nav = "🧠 Learner"
                            st.rerun()

    with col_stats:
        st.subheader("Global Stats")
        approved_q = global_stats.get("approved", 0)
        pending_q = global_stats.get("pending", 0)
        rejected_q = global_stats.get("rejected", 0)
        total_q = global_stats.get("total", 0)

        st.metric("Approved (Study Ready)", approved_q)
        st.metric("Pending Review", pending_q)
        st.metric("Review Bin (Rejected)", rejected_q)

        if total_q > 0:
            progress = approved_q / total_q
            st.progress(progress, text=f"{int(progress*100)}% Content Verified")
