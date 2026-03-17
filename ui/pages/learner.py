"""
ui/pages/learner.py
---------------------
Learner page — Active Recall study room for approved flashcards.
Moved verbatim from app.py::render_learner_view() — zero behaviour change.
"""

import streamlit as st
from sqlalchemy import func

from core.database import SessionLocal, Subject, Topic, Subtopic, Flashcard, SubjectDocumentAssociation
from ui.components.flashcard_card import render_flashcard_list


def render_learner_view():
    st.header("🧠 Active Recall Study Room")
    try:
        db = SessionLocal()

        subjects = db.query(Subject).filter(Subject.is_archived == False).all()
        if not subjects:
            st.info("No subjects available.")
            return

        subject_names = [s.name for s in subjects]
        default_subj_idx = 0
        if "study_subject_id" in st.session_state:
            for i, s in enumerate(subjects):
                if s.id == st.session_state.study_subject_id:
                    default_subj_idx = i
                    break

        selected_subject_name = st.selectbox("Select Subject", subject_names, index=default_subj_idx)
        selected_subject = db.query(Subject).filter(Subject.name == selected_subject_name).first()

        topics = db.query(Topic).join(SubjectDocumentAssociation, Topic.document_id == SubjectDocumentAssociation.document_id).\
            filter(SubjectDocumentAssociation.subject_id == selected_subject.id).all()

        if not topics:
            st.info("No topics available for this subject.")
            return

        topic_names = [t.name for t in topics]
        default_topic_idx = 0
        if "study_topic_id" in st.session_state:
            for i, t in enumerate(topics):
                if t.id == st.session_state.study_topic_id:
                    default_topic_idx = i
                    break

        selected_topic_name = st.selectbox("Select Topic to Study", topic_names, index=default_topic_idx)
        selected_topic = db.query(Topic).filter(Topic.name == selected_topic_name).first()

        subtopics = db.query(Subtopic).filter(Subtopic.topic_id == selected_topic.id).all()

        for sub in subtopics:
            is_expanded = False
            if "study_subtopic_id" in st.session_state and st.session_state.study_subtopic_id == sub.id:
                is_expanded = True

            approved_count = db.query(func.count(Flashcard.id)).filter(Flashcard.subtopic_id == sub.id, Flashcard.status == "approved").scalar()
            if approved_count > 0:
                with st.expander(f"📘 {sub.name} ({approved_count})", expanded=is_expanded):
                    render_flashcard_list(sub.id, "approved")
    finally:
        db.close()
