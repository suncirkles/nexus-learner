"""
ui/pages/mentor.py
--------------------
Mentor Review page — HITL approval/rejection of AI-generated flashcards.
Moved verbatim from app.py::render_mentor_review() — zero behaviour change.
"""

import streamlit as st
from sqlalchemy import func

from core.database import SessionLocal, Subject, Topic, Subtopic, Flashcard, SubjectDocumentAssociation
from ui.components.flashcard_card import render_flashcard_list, render_flashcard_review_card


def render_mentor_review():
    st.header("👨‍🏫 Mentor Review Workspace")
    db = SessionLocal()
    try:
        m_tabs = st.tabs(["⏳ Pending Review", "✅ Approved Items", "🗑️ Review Bin"])

        with m_tabs[0]:
            _ALL_QTYPES = ["active_recall", "fill_blank", "short_answer", "long_answer", "numerical", "scenario"]
            type_filter = st.multiselect(
                "Filter by Question Type (empty = show all)",
                _ALL_QTYPES,
                key="pending_type_filter",
            )

            subjects = db.query(Subject).filter(Subject.is_archived == False).all()
            if not subjects:
                st.info("No materials available for review.")

            for subj in subjects:
                topics = db.query(Topic).join(SubjectDocumentAssociation, Topic.document_id == SubjectDocumentAssociation.document_id).\
                    filter(SubjectDocumentAssociation.subject_id == subj.id).all()

                has_pending_in_subj = False
                for topic in topics:
                    subtopics = db.query(Subtopic).filter(Subtopic.topic_id == topic.id).all()
                    for sub in subtopics:
                        if db.query(Flashcard).filter(Flashcard.subtopic_id == sub.id, Flashcard.status == "pending").first():
                            has_pending_in_subj = True
                            break

                if has_pending_in_subj:
                    st.markdown(f"## 📁 Subject: {subj.name}")
                    for topic in topics:
                        subtopics = db.query(Subtopic).filter(Subtopic.topic_id == topic.id).all()

                        sub_ids = [s.id for s in subtopics]
                        has_pending_in_topic = db.query(Flashcard).filter(
                            Flashcard.subtopic_id.in_(sub_ids),
                            Flashcard.status == "pending"
                        ).first() is not None

                        if has_pending_in_topic:
                            st.markdown(f"### 📘 Topic: {topic.name}")
                            with st.expander(f"📦 Topic: {topic.name} Actions", expanded=False):
                                t_col1, t_col2, _ = st.columns([0.25, 0.25, 0.5])
                                if t_col1.button("✅ Approve All Topic", key=f"app_topic_{topic.id}"):
                                    db.query(Flashcard).filter(Flashcard.subtopic_id.in_(sub_ids), Flashcard.status == "pending").update({"status": "approved"}, synchronize_session=False)
                                    db.commit()
                                    st.rerun()
                                if t_col2.button("❌ Reject All Topic", key=f"rej_topic_{topic.id}"):
                                    db.query(Flashcard).filter(Flashcard.subtopic_id.in_(sub_ids), Flashcard.status == "pending").update({"status": "rejected"}, synchronize_session=False)
                                    db.commit()
                                    st.rerun()

                                for sub in subtopics:
                                    _pq = db.query(Flashcard).filter(Flashcard.subtopic_id == sub.id, Flashcard.status == "pending")
                                    if type_filter:
                                        _pq = _pq.filter(Flashcard.question_type.in_(type_filter))
                                    pending_fcs = _pq.all()
                                    if pending_fcs:
                                        with st.container(border=True):
                                            st.markdown(f"#### 📖 {sub.name} ({len(pending_fcs)} Pending)")
                                            b_col1, b_col2, _ = st.columns([0.2, 0.2, 0.6])
                                            if b_col1.button("✅ Approve All", key=f"app_all_{sub.id}"):
                                                db.query(Flashcard).filter(Flashcard.subtopic_id == sub.id, Flashcard.status == "pending").update({"status": "approved"}, synchronize_session=False)
                                                db.commit()
                                                st.rerun()
                                            if b_col2.button("❌ Reject All", key=f"rej_all_{sub.id}"):
                                                db.query(Flashcard).filter(Flashcard.subtopic_id == sub.id, Flashcard.status == "pending").update({"status": "rejected"}, synchronize_session=False)
                                                db.commit()
                                                st.rerun()

                                            render_flashcard_list(sub.id, "pending")
                    st.divider()

        with m_tabs[1]:
            st.subheader("Approved Knowledge")
            subjects = db.query(Subject).filter(Subject.is_archived == False).all()
            for subj in subjects:
                topics = db.query(Topic).join(SubjectDocumentAssociation, Topic.document_id == SubjectDocumentAssociation.document_id).\
                    filter(SubjectDocumentAssociation.subject_id == subj.id).order_by(Topic.created_at.desc()).all()

                if topics:
                    st.markdown(f"### 📁 Subject: {subj.name}")
                    for topic in topics:
                        subtopics = db.query(Subtopic).filter(Subtopic.topic_id == topic.id).all()
                        st.markdown(f"#### 📘 Topic: {topic.name}")
                        with st.container(border=True):
                            st.markdown(f"**Show All in {topic.name}**")
                            for sub in subtopics:
                                approved_count = db.query(func.count(Flashcard.id)).filter(Flashcard.subtopic_id == sub.id, Flashcard.status == "approved").scalar()
                                if approved_count > 0:
                                    with st.expander(f"📖 {sub.name} ({approved_count})", expanded=False):
                                        render_flashcard_list(sub.id, "approved")
                        st.divider()

        with m_tabs[2]:
            st.subheader("Review Bin (Rejected)")
            st.caption("Items here can be recreated with comments or permanently deleted.")
            rejected_fcs = db.query(Flashcard).filter(Flashcard.status == "rejected").order_by(Flashcard.created_at.desc()).all()
            if not rejected_fcs:
                st.info("Review bin is empty.")
            for fc in rejected_fcs:
                render_flashcard_review_card(db, fc, "rejected")

    finally:
        db.close()
