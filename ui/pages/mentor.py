"""
ui/pages/mentor.py
--------------------
Mentor Review page — HITL approval/rejection of AI-generated flashcards.
Moved verbatim from app.py::render_mentor_review() — zero behaviour change.
"""

import streamlit as st

from ui import api_client
from ui.components.flashcard_card import render_flashcard_list, render_flashcard_review_card


_ALL_QTYPES = ["active_recall", "fill_blank", "short_answer", "long_answer", "numerical", "scenario"]


def _load_subject_tree(subject_id: int) -> list:
    """Fetch topic tree for selected subject (1 API call = 2 DB queries). Cached in session state."""
    cache_key = f"mentor_tree_{subject_id}"
    return api_client.get_cached(cache_key, ttl_seconds=30, fetch_fn=lambda: _build_tree(subject_id))


def _build_tree(subject_id: int) -> list:
    topics_tree = api_client.get_topic_tree(subject_id)
    return [{"topic": t, "subtopics": t.get("subtopics", [])} for t in topics_tree]


def render_mentor_review():
    st.header("👨‍🏫 Mentor Review Workspace")

    # Load subject list once (1 call); tree loaded lazily per selected subject.
    subjects = api_client.list_active_subjects()

    if not subjects:
        st.info("No materials available for review.")
        return

    subject_names = [s["name"] for s in subjects]
    subject_by_name = {s["name"]: s for s in subjects}

    selected_name = st.selectbox(
        "Select Subject", subject_names, key="mentor_selected_subject"
    )
    selected_subj = subject_by_name[selected_name]
    subject_id = selected_subj["id"]

    # Tree: 1 API call covers all topics + subtopics + counts for the selected subject.
    topics_data = _load_subject_tree(subject_id)

    # Wrap in list so the rest of the render logic stays consistent.
    subjects_data = [{"subj": selected_subj, "topics_data": topics_data}]

    m_tabs = st.tabs(["⏳ Pending Review", "✅ Approved Items", "🗑️ Review Bin"])

    with m_tabs[0]:
        type_filter = st.multiselect(
            "Filter by Question Type (empty = show all)",
            _ALL_QTYPES,
            key="pending_type_filter",
        )

        if not subjects_data:
            st.info("No materials available for review.")

        for entry in subjects_data:
            subj = entry["subj"]
            topics_data = entry["topics_data"]

            has_pending_in_subj = any(
                s.get("pending_count", 0) > 0
                for td in topics_data
                for s in td["subtopics"]
            )

            if has_pending_in_subj:
                st.markdown(f"## 📁 Subject: {subj['name']}")
                for td in topics_data:
                    topic = td["topic"]
                    subtopics = td["subtopics"]
                    sub_ids = [s["id"] for s in subtopics]
                    has_pending_in_topic = any(s.get("pending_count", 0) > 0 for s in subtopics)

                    if has_pending_in_topic:
                        st.markdown(f"### 📘 Topic: {topic['name']}")
                        with st.expander(f"📦 Topic: {topic['name']} Actions", expanded=False):
                            t_col1, t_col2, _ = st.columns([0.25, 0.25, 0.5])
                            if t_col1.button("✅ Approve All Topic", key=f"app_topic_{topic['id']}"):
                                try:
                                    api_client.bulk_subtopic_action(sub_ids, "approve")
                                    st.session_state.pop(f"mentor_tree_{subject_id}", None)
                                    st.session_state.pop(f"mentor_tree_{subject_id}__ts", None)
                                    st.toast("Approved all pending cards for this topic.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to approve: {e}")
                            if t_col2.button("❌ Reject All Topic", key=f"rej_topic_{topic['id']}"):
                                try:
                                    api_client.bulk_subtopic_action(sub_ids, "reject")
                                    st.session_state.pop(f"mentor_tree_{subject_id}", None)
                                    st.session_state.pop(f"mentor_tree_{subject_id}__ts", None)
                                    st.toast("Rejected all pending cards for this topic.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to reject: {e}")

                            for sub in subtopics:
                                if sub.get("pending_count", 0) == 0:
                                    continue
                                # Pass single type server-side; multi-type filtered client-side.
                                server_qtype = type_filter[0] if len(type_filter) == 1 else None
                                pending_fcs = api_client.get_flashcards_by_subtopic(
                                    sub["id"], "pending", question_type=server_qtype
                                )
                                if len(type_filter) > 1:
                                    pending_fcs = [f for f in pending_fcs if f.get("question_type") in type_filter]
                                if pending_fcs:
                                    with st.container(border=True):
                                        st.markdown(f"#### 📖 {sub['name']} ({len(pending_fcs)} Pending)")
                                        b_col1, b_col2, _ = st.columns([0.2, 0.2, 0.6])
                                        if b_col1.button("✅ Approve All", key=f"app_all_{sub['id']}"):
                                            try:
                                                api_client.bulk_subtopic_action([sub["id"]], "approve")
                                                st.session_state.pop(f"mentor_tree_{subject_id}", None)
                                                st.session_state.pop(f"mentor_tree_{subject_id}__ts", None)
                                                st.toast(f"Approved all pending cards for {sub['name']}.")
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Failed to approve: {e}")
                                        if b_col2.button("❌ Reject All", key=f"rej_all_{sub['id']}"):
                                            try:
                                                api_client.bulk_subtopic_action([sub["id"]], "reject")
                                                st.session_state.pop(f"mentor_tree_{subject_id}", None)
                                                st.session_state.pop(f"mentor_tree_{subject_id}__ts", None)
                                                st.toast(f"Rejected all pending cards for {sub['name']}.")
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Failed to reject: {e}")

                                        render_flashcard_list(sub["id"], "pending")
                st.divider()

    with m_tabs[1]:
        st.subheader("Approved Knowledge")
        for entry in subjects_data:
            subj = entry["subj"]
            topics_data = entry["topics_data"]

            if topics_data:
                st.markdown(f"### 📁 Subject: {subj['name']}")
                for td in topics_data:
                    topic = td["topic"]
                    subtopics = td["subtopics"]
                    st.markdown(f"#### 📘 Topic: {topic['name']}")
                    with st.container(border=True):
                        st.markdown(f"**Show All in {topic['name']}**")
                        for sub in subtopics:
                            approved_count = sub.get("approved_count", 0)
                            if approved_count > 0:
                                with st.expander(f"📖 {sub['name']} ({approved_count})", expanded=False):
                                    render_flashcard_list(sub["id"], "approved")
                    st.divider()

    with m_tabs[2]:
        st.subheader("Review Bin (Rejected)")
        st.caption("Items here can be recreated with comments or permanently deleted.")
        rejected_fcs = api_client.get_all_rejected_flashcards()
        if not rejected_fcs:
            st.info("Review bin is empty.")
        else:
            # Batch-fetch all chunk sources in 1 call instead of 1 per card.
            chunk_ids = [fc["chunk_id"] for fc in rejected_fcs if fc.get("chunk_id")]
            sources_map = api_client.get_chunk_sources_batch(chunk_ids)
            for fc in rejected_fcs:
                src = sources_map.get(str(fc.get("chunk_id"))) if fc.get("chunk_id") else None
                render_flashcard_review_card(fc, "rejected", src=src)
