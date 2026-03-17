"""
ui/pages/learner.py
---------------------
Learner page — Active Recall study room for approved flashcards.
Moved verbatim from app.py::render_learner_view() — zero behaviour change.
"""

import streamlit as st

from ui import api_client
from ui.components.flashcard_card import render_flashcard_list


def render_learner_view():
    st.header("🧠 Active Recall Study Room")

    subjects = api_client.list_active_subjects()
    if not subjects:
        st.info("No subjects available.")
        return

    subject_names = [s["name"] for s in subjects]
    default_subj_idx = 0
    if "study_subject_id" in st.session_state:
        for i, s in enumerate(subjects):
            if s["id"] == st.session_state.study_subject_id:
                default_subj_idx = i
                break

    selected_subject_name = st.selectbox("Select Subject", subject_names, index=default_subj_idx)
    selected_subject = next((s for s in subjects if s["name"] == selected_subject_name), None)
    if not selected_subject:
        return

    topics = api_client.get_topics_by_subject(selected_subject["id"])

    if not topics:
        st.info("No topics available for this subject.")
        return

    topic_names = [t["name"] for t in topics]
    default_topic_idx = 0
    if "study_topic_id" in st.session_state:
        for i, t in enumerate(topics):
            if t["id"] == st.session_state.study_topic_id:
                default_topic_idx = i
                break

    selected_topic_name = st.selectbox("Select Topic to Study", topic_names, index=default_topic_idx)
    selected_topic = next((t for t in topics if t["name"] == selected_topic_name), None)
    if not selected_topic:
        return

    subtopics = api_client.get_subtopics_by_topic(selected_topic["id"])

    for sub in subtopics:
        is_expanded = False
        if "study_subtopic_id" in st.session_state and st.session_state.study_subtopic_id == sub["id"]:
            is_expanded = True

        approved_count = sub.get("approved_count", 0)
        if approved_count > 0:
            with st.expander(f"📘 {sub['name']} ({approved_count})", expanded=is_expanded):
                render_flashcard_list(sub["id"], "approved")
