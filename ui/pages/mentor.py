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


def render_mentor_review():
    st.header("👨‍🏫 Mentor Review Workspace")

    # Pre-fetch all subjects → topics → subtopics once and share across all tabs.
    subjects = api_client.list_active_subjects()
    subjects_data = []
    for subj in subjects:
        topics = api_client.get_topics_by_subject(subj["id"])
        topics_data = []
        for topic in topics:
            subtopics = api_client.get_subtopics_by_topic(topic["id"])
            topics_data.append({"topic": topic, "subtopics": subtopics})
        subjects_data.append({"subj": subj, "topics_data": topics_data})

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
                                    st.toast("Approved all pending cards for this topic.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to approve: {e}")
                            if t_col2.button("❌ Reject All Topic", key=f"rej_topic_{topic['id']}"):
                                try:
                                    api_client.bulk_subtopic_action(sub_ids, "reject")
                                    st.toast("Rejected all pending cards for this topic.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to reject: {e}")

                            for sub in subtopics:
                                if sub.get("pending_count", 0) == 0:
                                    continue
                                pending_fcs = api_client.get_flashcards_by_subtopic(sub["id"], "pending")
                                if type_filter:
                                    pending_fcs = [f for f in pending_fcs if f.get("question_type") in type_filter]
                                if pending_fcs:
                                    with st.container(border=True):
                                        st.markdown(f"#### 📖 {sub['name']} ({len(pending_fcs)} Pending)")
                                        b_col1, b_col2, _ = st.columns([0.2, 0.2, 0.6])
                                        if b_col1.button("✅ Approve All", key=f"app_all_{sub['id']}"):
                                            try:
                                                api_client.bulk_subtopic_action([sub["id"]], "approve")
                                                st.toast(f"Approved all pending cards for {sub['name']}.")
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Failed to approve: {e}")
                                        if b_col2.button("❌ Reject All", key=f"rej_all_{sub['id']}"):
                                            try:
                                                api_client.bulk_subtopic_action([sub["id"]], "reject")
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
        for fc in rejected_fcs:
            render_flashcard_review_card(fc, "rejected")
