"""
app.py
------
Streamlit-based frontend for Nexus Learner. Provides five main views:
  1. Dashboard   – Subject tiles with stats and quick navigation.
  2. Study Materials – Document upload, ingestion, and background processing.
  3. Mentor Review   – HITL approval/rejection of AI-generated flashcards.
  4. Learner         – Active Recall study room for approved flashcards.
  5. System Tools    – Administrative controls (reset, edit subjects/topics).
"""

import streamlit as st
import os
import uuid
import time
from core.database import SessionLocal, Subject, Document as DBDocument, Flashcard, ContentChunk, Topic, Subtopic, Base, engine
from core.config import settings
from workflows.phase1_ingestion import phase1_graph
from sqlalchemy import delete, func
from qdrant_client import QdrantClient

# Configure Page
st.set_page_config(page_title="Nexus Learner - Agentic Learning Platform", layout="wide", page_icon="🎓")

# Initialize Qdrant Client for admin tasks
qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

# Custom CSS for Premium Look & Fixed Button Sizing
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stCard {
        background-color: #1a1c24;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 15px;
    }
    .topic-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #3b82f6;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .topic-header { color: #58a6ff; font-size: 1.6rem; font-weight: 700; margin-bottom: 5px; }
    .subtopic-header { color: #8b949e; font-size: 1.1rem; font-weight: 500; }
    .critic-score {
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        background-color: #238636;
        color: white;
    }
    /* Fixed Button Sizing for Bulk Actions */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 45px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%); }
    
    .subject-tile {
        background: linear-gradient(135deg, #232731 0%, #1a1c24 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
    }
    .subject-tile:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        border-color: #58a6ff;
    }
    .subject-title {
        color: #58a6ff;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .stat-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    .stat-label { color: #8b949e; }
    .stat-value { color: #c9d1d9; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def reset_entire_system():
    """Wipes SQLite and Qdrant Collection."""
    # 1. Clear SQLite
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    # 2. Clear Qdrant
    try:
        qdrant_client.delete_collection(collection_name=settings.QDRANT_COLLECTION_NAME)
    except Exception as e:
        st.error(f"Failed to clear Qdrant: {e}")
    
    st.success("System completely reset!")
    time.sleep(1)
    st.rerun()

def delete_topic_data(topic_id: int, document_id: str):
    """Deletes topic, subtopics, flashcards and associated vectors."""
    db = SessionLocal()
    try:
        subtopics = db.query(Subtopic).filter(Subtopic.topic_id == topic_id).all()
        subtopic_ids = [s.id for s in subtopics]
        
        # SQL Cleanup
        db.execute(delete(Flashcard).where(Flashcard.subtopic_id.in_(subtopic_ids)))
        db.execute(delete(Subtopic).where(Subtopic.topic_id == topic_id))
        db.execute(delete(Topic).where(Topic.id == topic_id))
        db.execute(delete(ContentChunk).where(ContentChunk.document_id == document_id))
        db.commit()
        
        # Qdrant Cleanup
        try:
            from qdrant_client.http import models as rest
            qdrant_client.delete(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points_selector=rest.FilterSelector(
                    filter=rest.Filter(
                        must=[
                            rest.FieldCondition(
                                key="document_id",
                                match=rest.MatchValue(value=document_id),
                            ),
                        ]
                    )
                ),
            )
        except Exception as e:
            st.warning(f"Relational data deleted, but vector cleanup failed: {e}")
            
        st.success(f"Topic {topic_id} and associated vectors reset.")
    finally:
        db.close()

@st.fragment
def render_flashcard_list(sub_id, status="approved"):
    db = SessionLocal()
    try:
        fcs = db.query(Flashcard).filter(Flashcard.subtopic_id == sub_id, Flashcard.status == status).all()
        if not fcs:
            st.info("No flashcards found.")
            return

        for fc in fcs:
            if status == "approved":
                st.markdown(f"""
                <div class='stCard'>
                    <strong>Q: {fc.question}</strong>
                    <hr style='border-top: 1px solid #30363d; margin: 10px 0;'/>
                    <details style='cursor: pointer;'>
                        <summary style='color: #58a6ff;'>Show Answer</summary>
                        <p style='margin-top: 10px;'>{fc.answer}</p>
                    </details>
                </div>
                """, unsafe_allow_html=True)
            else:
                render_flashcard_review_card(db, fc, status)
    finally:
        db.close()

# --- Tab Renderers ---

def render_dashboard():
    st.header("🏠 Agentic Learning Dashboard")
    try:
        db = SessionLocal()
        
        subjects = db.query(Subject).all()
        
        if not subjects:
            st.info("No subjects found. Head over to 'Study Materials' to define your first subject and upload documents!")
            return

        col_kb, col_stats = st.columns([0.7, 0.3])
    
        with col_kb:
            st.subheader("📚 My Subjects")
            
            from sqlalchemy import case, Integer
            # Pre-calculate topic counts for all subjects
            topic_counts_raw = db.query(Topic.subject_id, func.count(Topic.id)).group_by(Topic.subject_id).all()
            topic_counts = {r[0]: r[1] for r in topic_counts_raw}
            
            # Pre-calculate flashcard stats for all subjects using JOINs
            fc_stats_raw = db.query(
                Topic.subject_id,
                func.sum(case((Flashcard.status == 'approved', 1), else_=0)).cast(Integer),
                func.sum(case((Flashcard.status == 'pending', 1), else_=0)).cast(Integer)
            ).select_from(Topic).join(Subtopic, Topic.id == Subtopic.topic_id).join(Flashcard, Subtopic.id == Flashcard.subtopic_id).group_by(Topic.subject_id).all()
            
            fc_stats = {r[0]: {"approved": r[1] or 0, "pending": r[2] or 0} for r in fc_stats_raw}
            
            # Grid layout for subject tiles
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
                                # Clear specific topic/subtopic to let learner view handle defaults
                                if "study_topic_id" in st.session_state: del st.session_state.study_topic_id
                                if "study_subtopic_id" in st.session_state: del st.session_state.study_subtopic_id
                                st.session_state.active_nav = "🧠 Learner"
                                st.rerun()

        with col_stats:
            st.subheader("Global Stats")
            total_q = db.query(func.count(Flashcard.id)).scalar()
            approved_q = db.query(func.count(Flashcard.id)).filter(Flashcard.status == "approved").scalar()
            pending_q = db.query(func.count(Flashcard.id)).filter(Flashcard.status == "pending").scalar()
            rejected_q = db.query(func.count(Flashcard.id)).filter(Flashcard.status == "rejected").scalar()
            
            st.metric("Approved (Study Ready)", approved_q)
            st.metric("Pending Review", pending_q)
            st.metric("Review Bin (Rejected)", rejected_q)
            
            if total_q > 0:
                progress = approved_q / total_q
                st.progress(progress, text=f"{int(progress*100)}% Content Verified")
    finally:
        db.close()

def render_study_materials():
    st.header("📚 Study Material Ingestion")
    db = SessionLocal()
    try:

        # Use session state to manage the current selected subject persistently
        if "ingest_subject_id" not in st.session_state:
            st.session_state.ingest_subject_id = None

        # Step-based UI
        st.subheader("1. Subject Context")

        subjects = db.query(Subject).all()

        if not subjects:
            st.info("No subjects found. Please create your first subject below.")
            subj_mode = "Create New Subject"
        else:
            subj_mode = st.radio("What would you like to do?", ["Select Existing Subject", "Create New Subject"], horizontal=True)

        if subj_mode == "Create New Subject":
            with st.container(border=True):
                new_subj_name = st.text_input("New Subject Name (e.g. Machine Learning)")
                if st.button("✨ Create Subject"):
                    cleaned_name = new_subj_name.strip() if new_subj_name else ""
                    if not cleaned_name:
                        st.warning("Please enter a valid, non-empty name.")
                    elif len(cleaned_name) > 100:
                        st.warning("Subject name is too long (maximum 100 characters).")
                    else:
                        try:
                            new_subj = Subject(name=cleaned_name)
                            db.add(new_subj)
                            db.commit()
                            st.session_state.ingest_subject_id = new_subj.id
                            st.success(f"Subject '{new_subj_name}' created and selected!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            # Select Existing
            subject_names = [s.name for s in subjects]
            current_idx = 0
            if st.session_state.ingest_subject_id:
                for i, s in enumerate(subjects):
                    if s.id == st.session_state.ingest_subject_id:
                        current_idx = i
                        break

            selected_name = st.selectbox("Choose Subject", subject_names, index=current_idx)
            selected_subj = db.query(Subject).filter(Subject.name == selected_name).first()
            st.session_state.ingest_subject_id = selected_subj.id
            st.info(f"Adding artifacts to: **{selected_name}**")

        st.divider()

        # 2. Upload
        if st.session_state.ingest_subject_id:
            subject_id = st.session_state.ingest_subject_id
            st.subheader(f"2. Upload Artifacts to '{db.query(Subject).get(subject_id).name}'")

            with st.expander("📥 Click to Upload PDF or Image", expanded=True):
                uploaded_file = st.file_uploader("Upload File", type=["pdf", "png", "jpg", "jpeg"])
                if uploaded_file:
                    if uploaded_file.size > 50 * 1024 * 1024:
                        st.error("File exceeds the 50MB limit (50MB max).")
                        uploaded_file = None
                    else:
                        st.success(f"File selected: {uploaded_file.name}")
                if uploaded_file and st.button("🚀 Process & Generate Hierarchy"):
                    doc_id = str(uuid.uuid4())
                    safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in " ._-").strip()
                    os.makedirs("temp_uploads", exist_ok=True)
                    file_path = os.path.join("temp_uploads", f"{doc_id}_{safe_filename}")
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    progress_bar = st.progress(0, text="Initializing...")

                    state = {
                        "file_path": file_path,
                        "doc_id": doc_id,
                        "subject_id": subject_id,
                        "chunks": [],
                        "hierarchy": [],
                        "doc_summary": "",
                        "current_chunk_index": 0,
                        "generated_flashcards": [],
                        "status_message": "Starting ingestion..."
                    }

                    try:
                        # 1. Sync Phase: Ingest, Curate, and Initial Burst
                        sync_limit = 5 # Reduced for speed in demo
                        processed_count = 0

                        stream = phase1_graph.stream(state)

                        for event in stream:
                            node_name = list(event.keys())[0]
                            state.update(event[node_name])

                            if node_name == "ingest":
                                progress_bar.progress(10, text="Ingested. Analyzing structure...")
                            elif node_name == "curate":
                                progress_bar.progress(20, text="Hierarchy extracted. Starting Q&A generation...")
                            if node_name == "generate":
                                processed_count += 1
                                msg = state.get("status_message", "Generating...")
                                p = 20 + int((processed_count / min(len(state["chunks"]), sync_limit)) * 70)
                                progress_bar.progress(min(p, 90), text=msg)

                            if node_name == "increment" and processed_count >= sync_limit:
                                break

                        if processed_count < len(state["chunks"]):
                            from core.background import start_background_task
                            start_background_task(state, doc_id, filename=uploaded_file.name)
                            st.toast(f"Initial {processed_count} cards for '{uploaded_file.name}' ready! Rest processing in background.")

                        progress_bar.progress(100, text="Initial Processing Complete!")
                        st.success(f"Successfully started '{uploaded_file.name}'!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Processing Error: {e}")
                    finally:
                        if os.path.exists(file_path):
                            os.remove(file_path)
        else:
            st.warning("Please select or create a subject above before uploading artifacts.")

    finally:
        db.close()

    # Background Monitor
    from core.background import background_tasks, stop_background_task
    if background_tasks:
        st.divider()
        st.subheader("⚙️ Active Background Tasks")
        for d_id, task in list(background_tasks.items()):
            # Use filename if available, else doc_id
            display_name = task.get("filename", d_id[:8])
            
            with st.container(border=True):
                col_t, col_b = st.columns([0.85, 0.15])
                
                if task["status"] == "processing":
                    prog = task['progress']
                    total = task['total']
                    percent = (prog / total) if total > 0 else 0
                    
                    col_t.markdown(f"**⏳ Processing: {display_name}**")
                    col_t.progress(percent, text=f"{prog}/{total} chunks generated")
                    
                    if col_b.button("⏹️ Stop", key=f"stop_{d_id}"):
                        stop_background_task(d_id)
                        st.rerun()
                
                elif task["status"] == "completed":
                    col_t.success(f"✅ **Completed**: {display_name}")
                    if col_b.button("✖ Clear", key=f"clear_{d_id}"):
                        del background_tasks[d_id]
                        st.rerun()
                
                elif task["status"] == "stopped":
                    col_t.warning(f"⏹️ **Stopped**: {display_name}")
                    if col_b.button("✖ Clear", key=f"clear_{d_id}"):
                        del background_tasks[d_id]
                        st.rerun()
                
                elif task["status"] == "failed":
                    col_t.error(f"❌ **Failed**: {display_name} - {task.get('error', 'Unknown Error')}")
                    if col_b.button("✖ Clear", key=f"clear_{d_id}"):
                        del background_tasks[d_id]
                        st.rerun()

def render_mentor_review():
    st.header("👨‍🏫 Mentor Review Workspace")
    db = SessionLocal()
    try:

        m_tabs = st.tabs(["⏳ Pending Review", "✅ Approved Items", "🗑️ Review Bin"])

        with m_tabs[0]:
            subjects = db.query(Subject).all()
            if not subjects:
                st.info("No materials available for review.")

            for subj in subjects:
                topics = db.query(Topic).filter(Topic.subject_id == subj.id).all()

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

                        # Efficiently check for any pending flashcards in this topic
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
                                    pending_fcs = db.query(Flashcard).filter(Flashcard.subtopic_id == sub.id, Flashcard.status == "pending").all()
                                    if pending_fcs:
                                        with st.container(border=True):
                                            st.markdown(f"#### 📖 {sub.name} ({len(pending_fcs)} Pending)")
                                            b_col1, b_col2, _ = st.columns([0.2, 0.2, 0.6])
                                            if b_col1.button("✅ Approve All", key=f"app_all_{sub.id}"):
                                                db.query(Flashcard).filter(Flashcard.subtopic_id == sub.id, Flashcard.status == "pending").update({"status": "approved"})
                                                db.commit()
                                                st.rerun()
                                            if b_col2.button("❌ Reject All", key=f"rej_all_{sub.id}"):
                                                db.query(Flashcard).filter(Flashcard.subtopic_id == sub.id, Flashcard.status == "pending").update({"status": "rejected"})
                                                db.commit()
                                                st.rerun()

                                            render_flashcard_list(sub.id, "pending")
                    st.divider()

        with m_tabs[1]:
            st.subheader("Approved Knowledge")
            subjects = db.query(Subject).all()
            for subj in subjects:
                topics = db.query(Topic).filter(Topic.subject_id == subj.id).order_by(Topic.created_at.desc()).all()

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

def render_flashcard_review_card(db, fc, current_status):
    st.markdown(f"""
    <div class='stCard'>
        <div style='display:flex; justify-content:space-between;'>
            <strong>Question:</strong>
            <span class='critic-score'>Score: {fc.critic_score}/5</span>
        </div>
        <p>{fc.question}</p>
        <hr style='border-top: 1px solid #30363d; margin: 10px 0;'/>
        <strong>Answer:</strong>
        <p>{fc.answer}</p>
        <div style='font-size: 0.85rem; color: #8b949e;'>
            <em>Critic Feedback: {fc.critic_feedback}</em>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns([0.15, 0.15, 0.15, 0.55])
    
    if current_status == "pending":
        if cols[0].button("Approve", key=f"p_app_{fc.id}"):
            fc.status = "approved"
            db.commit()
            st.rerun()
        if cols[1].button("Reject", key=f"p_rej_{fc.id}"):
            fc.status = "rejected"
            db.commit()
            st.rerun()
    elif current_status == "approved":
        if cols[0].button("Discard", key=f"a_disc_{fc.id}"):
            fc.status = "rejected"
            db.commit()
            st.rerun()
    elif current_status == "rejected":
        if cols[0].button("Restore", key=f"r_res_{fc.id}"):
            fc.status = "pending"
            db.commit()
            st.rerun()
        if cols[1].button("Delete", key=f"r_del_{fc.id}"):
            db.delete(fc)
            db.commit()
            st.rerun()
        
        # Recreation form
        with st.expander("🔄 Recreate with Feedback", expanded=False):
            # Suggested Answer logic
            feedback_key = f"fb_text_{fc.id}"
            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = ""
            
            if st.button("💡 Get Suggested Answer", key=f"get_sug_{fc.id}"):
                from agents.socratic import SocraticAgent
                s_agent = SocraticAgent()
                with st.spinner("LLM is thinking..."):
                    suggestion = s_agent.suggest_answer(fc.question, fc.id)
                    st.session_state[feedback_key] = f"Suggested Answer: {suggestion}"
                    st.rerun()

            feedback = st.text_area("What needs to be fixed?", value=st.session_state[feedback_key], key=f"actual_fb_{fc.id}")
            if st.button("Regenerate Flashcard", key=f"fb_btn_{fc.id}"):
                if feedback:
                    from agents.socratic import SocraticAgent
                    s_agent = SocraticAgent()
                    with st.spinner("Generating better Q&A..."):
                        res = s_agent.recreate_flashcard(fc.id, feedback)
                        if res["status"] == "success":
                            # Remove from Review Bin immediately by rerunning
                            del st.session_state[feedback_key]
                            st.success("Flashcard updated and moved to Pending.")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(res.get("message", "Error recreating flashcard."))
                else:
                    st.warning("Please provide feedback first.")

def render_learner_view():
    st.header("🧠 Active Recall Study Room")
    try:
        db = SessionLocal()
        
        # 1. Subject Selection
        subjects = db.query(Subject).all()
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
    
        # 2. Topic Selection within Subject
        topics = db.query(Topic).filter(Topic.subject_id == selected_subject.id).all()
    
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
            # Determine if this subtopic should be expanded (deep link)
            is_expanded = False
            if "study_subtopic_id" in st.session_state and st.session_state.study_subtopic_id == sub.id:
                is_expanded = True
            
            approved_count = db.query(func.count(Flashcard.id)).filter(Flashcard.subtopic_id == sub.id, Flashcard.status == "approved").scalar()
            if approved_count > 0:
                with st.expander(f"📘 {sub.name} ({approved_count})", expanded=is_expanded):
                    render_flashcard_list(sub.id, "approved")
    finally:
        db.close()

def render_system_tools():
    st.header("⚙️ Administrative Controls")
    db = SessionLocal()
    try:
        from core.database import Document as DBDocument

        st.warning("These actions are destructive and cannot be undone.")

        if st.button("🚨 Global Reset: Wipe Database & Qdrant Collections"):
            reset_entire_system()

        st.divider()
        st.divider()
        st.subheader("Subject & Topic Management")

        # Subject Editing
        subjects = db.query(Subject).all()
        for subj in subjects:
            col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
            new_subj_name = col1.text_input(f"Edit Subject Name", value=subj.name, key=f"edit_subj_{subj.id}")
            if col2.button("Update Name", key=f"upd_subj_{subj.id}"):
                subj.name = new_subj_name
                db.commit()
                st.success("Subject updated!")
                st.rerun()
            if col3.button("Delete Subject", key=f"del_subj_{subj.id}"):
                # Recursive delete
                documents = db.query(DBDocument).filter(DBDocument.subject_id == subj.id).all()
                for d in documents:
                    topics = db.query(Topic).filter(Topic.document_id == d.id).all()
                    for t in topics:
                        delete_topic_data(t.id, d.id)
                    db.delete(d)
                db.delete(subj)
                db.commit()
                st.success("Subject and all its data deleted.")
                st.rerun()

            # Topic Editing within Subject
            topics = db.query(Topic).filter(Topic.subject_id == subj.id).all()

            for topic in topics:
                tcol1, tcol2, tcol3 = st.columns([0.6, 0.2, 0.2])
                new_topic_name = tcol1.text_input(f"   ↳ Edit Topic Name", value=topic.name, key=f"edit_top_{topic.id}")
                if tcol2.button("Update Topic", key=f"upd_top_{topic.id}"):
                    topic.name = new_topic_name
                    db.commit()
                    st.success("Topic updated!")
                    st.rerun()
                if tcol3.button("Delete Topic", key=f"del_top_{topic.id}"):
                    # Specific topic delete
                    doc = db.query(DBDocument).filter(DBDocument.id == topic.document_id).first()
                    delete_topic_data(topic.id, doc.id)
                    st.rerun()
            st.divider()
    finally:
        db.close()

# --- Main Entry Point ---

def main():
    st.sidebar.title("🎓 Nexus Learner")
    
    nav_options = ["🏠 Dashboard", "📚 Study Materials", "👨‍🏫 Mentor Review", "🧠 Learner", "⚙️ System Tools"]
    
    # Initialize navigation state
    if "active_nav" not in st.session_state:
        st.session_state.active_nav = nav_options[0]
        
    # Sync navigation with sidebar
    active_nav = st.sidebar.radio("Navigation", nav_options, index=nav_options.index(st.session_state.active_nav), key="sidebar_nav")
    
    # If user manually clicks sidebar, update state
    if active_nav != st.session_state.active_nav:
        st.session_state.active_nav = active_nav

    # Render Content
    if st.session_state.active_nav == "🏠 Dashboard":
        render_dashboard()
    elif st.session_state.active_nav == "📚 Study Materials":
        render_study_materials()
    elif st.session_state.active_nav == "👨‍🏫 Mentor Review":
        render_mentor_review()
    elif st.session_state.active_nav == "🧠 Learner":
        render_learner_view()
    elif st.session_state.active_nav == "⚙️ System Tools":
        render_system_tools()

if __name__ == "__main__":
    main()
