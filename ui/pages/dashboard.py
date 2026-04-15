"""
ui/pages/dashboard.py
-----------------------
Dashboard page — subject tiles with stats and quick navigation.
Moved verbatim from app.py::render_dashboard() — zero behaviour change.
"""

import logging
import time

import httpx
import streamlit as st

from ui import api_client

logger = logging.getLogger(__name__)

# Modal cold starts for heavy Python containers can take 60-120 s.
_MAX_RETRIES = 15       # 15 × 6 s = up to 90 s of auto-waiting
_RETRY_INTERVAL_S = 6


def _fetch_dashboard_data() -> tuple:
    """Returns (subjects_data, topic_counts, fc_stats, global_stats). 2 API calls total."""
    subjects = api_client.get_subjects_with_stats()  # 1 call (3 DB queries)
    global_stats = api_client.get_global_stats()     # 1 call

    subjects_data = [(s["id"], s["name"]) for s in subjects]
    topic_counts = {s["id"]: s.get("topic_count", 0) for s in subjects}
    fc_stats = {
        s["id"]: {"approved": s.get("approved", 0), "pending": s.get("pending", 0)}
        for s in subjects
    }

    return subjects_data, topic_counts, fc_stats, global_stats


def render_dashboard():
    st.header("🏠 Agentic Learning Dashboard")

    # --- API cold-start resilience -------------------------------------------
    # Modal FastAPI containers can take 60-120 s to cold-start.  Auto-retry
    # with a spinner so the user doesn't have to keep clicking Retry.
    retry_count = st.session_state.get("_api_cold_start_retries", 0)

    try:
        subjects_data, topic_counts, fc_stats, global_stats = _fetch_dashboard_data()
        # Success — clear the retry counter
        st.session_state.pop("_api_cold_start_retries", None)
    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as exc:
        logger.warning("Dashboard API unavailable (attempt %d): %s", retry_count + 1, exc)
        if retry_count < _MAX_RETRIES:
            st.session_state["_api_cold_start_retries"] = retry_count + 1
            elapsed = retry_count * _RETRY_INTERVAL_S
            st.info(
                f"API is starting up — attempt {retry_count + 1}/{_MAX_RETRIES} "
                f"({elapsed}s elapsed). Retrying in {_RETRY_INTERVAL_S}s…",
                icon="⏳",
            )
            time.sleep(_RETRY_INTERVAL_S)
            st.rerun()
        else:
            st.session_state.pop("_api_cold_start_retries", None)
            st.error(
                "API did not respond after 90 s. Check Modal deployment logs for errors.",
                icon="❌",
            )
            st.code(str(exc))
            if st.button("🔄 Try again", key="dashboard_retry_exhausted"):
                st.rerun()
        return
    except Exception as exc:
        logger.error("Dashboard data fetch failed: %s", exc)
        st.session_state.pop("_api_cold_start_retries", None)
        st.error(f"Could not load dashboard data: {exc}")
        if st.button("🔄 Retry", key="dashboard_retry_err"):
            st.rerun()
        return
    # -------------------------------------------------------------------------

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
                            st.session_state.pending_nav = "🧠 Learner"
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
