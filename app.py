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
import tempfile
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

def _get_flashcard_source_attribution(db, fc) -> str:
    """Return a short plain-text source attribution string for the Learner view."""
    try:
        chunk = db.query(ContentChunk).filter(ContentChunk.id == fc.chunk_id).first() if fc.chunk_id else None
        if chunk and chunk.source_type:
            if chunk.source_type == "web" and chunk.source_url:
                from urllib.parse import urlparse
                domain = urlparse(chunk.source_url).netloc.replace("www.", "")
                return f"🌐 {domain}"
            elif chunk.source_type == "image":
                if chunk.document_id:
                    doc = db.query(DBDocument).filter(DBDocument.id == chunk.document_id).first()
                    return f"🖼️ {doc.filename}" if doc else "🖼️ image"
            else:
                if chunk.document_id:
                    doc = db.query(DBDocument).filter(DBDocument.id == chunk.document_id).first()
                    return f"📄 {doc.filename}" if doc else "📄 document"
    except Exception:
        pass
    return ""


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
                source_attr = _get_flashcard_source_attribution(db, fc)
                source_html = (
                    f"<div style='margin-top:8px; font-size:0.78rem; color:#8b949e;'>Source: {source_attr}</div>"
                    if source_attr else ""
                )
                st.markdown(f"""
                <div class='stCard'>
                    <strong>Q: {fc.question}</strong>
                    <hr style='border-top: 1px solid #30363d; margin: 10px 0;'/>
                    <details style='cursor: pointer;'>
                        <summary style='color: #58a6ff;'>Show Answer</summary>
                        <p style='margin-top: 10px;'>{fc.answer}</p>
                        {source_html}
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
        
        subjects = db.query(Subject).filter(Subject.is_archived == False).all()
        
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

def _render_upload_tab(db, subject_id: int, subject_name: str):
    """Tab 1: existing PDF / Image upload flow — completely unchanged in behaviour."""
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
                sync_limit = 5
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


def _render_web_research_tab(subject_id: int, subject_name: str):
    """Tab 2: Web Research — topic input, safety check, and pipeline launch."""

    st.markdown("#### Topic Input")

    input_mode = st.radio(
        "How would you like to provide topics?",
        ["✏️ Type topics", "📄 Upload topic file"],
        horizontal=True,
        key="web_input_mode",
    )

    raw_topics: list = []

    if input_mode == "✏️ Type topics":
        topic_text = st.text_area(
            "Enter topics (one per line, or separated by commas)",
            height=150,
            key="web_topic_text",
            placeholder="e.g.\nBinary Search Trees\nGraph Traversal Algorithms\nDynamic Programming",
        )
        if st.button("Parse Topics", key="btn_parse_text"):
            if topic_text.strip():
                with st.spinner("Parsing topics..."):
                    from agents.topic_parser import TopicParserAgent
                    parser = TopicParserAgent()
                    raw_topics = parser.parse_topics_from_text(topic_text)
                st.session_state[f"web_parsed_topics_{subject_id}"] = raw_topics
            else:
                st.warning("Please enter some topics first.")

    else:  # Upload topic file
        topic_file = st.file_uploader(
            "Upload .txt, .pdf, or .docx",
            type=["txt", "pdf", "docx"],
            key="web_topic_file",
        )
        if topic_file and st.button("Parse Topics from File", key="btn_parse_file"):
            suffix = os.path.splitext(topic_file.name)[1].lstrip(".")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                tmp.write(topic_file.getbuffer())
                tmp_path = tmp.name
            try:
                with st.spinner("Extracting and parsing topics..."):
                    from agents.topic_parser import TopicParserAgent
                    parser = TopicParserAgent()
                    raw_topics = parser.parse_topics_from_file(tmp_path, suffix)
                st.session_state[f"web_parsed_topics_{subject_id}"] = raw_topics
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # --- Parsed topics preview with per-topic removal ---
    topics_key = f"web_parsed_topics_{subject_id}"
    if topics_key not in st.session_state:
        st.session_state[topics_key] = []

    parsed_topics: list = st.session_state[topics_key]

    if parsed_topics:
        with st.expander(f"📋 Topics identified ({len(parsed_topics)}) — click to review", expanded=True):
            to_remove = []
            for i, topic in enumerate(parsed_topics):
                col_t, col_x = st.columns([0.9, 0.1])
                col_t.markdown(f"- {topic}")
                if col_x.button("✕", key=f"rm_topic_{i}", help=f"Remove '{topic}'"):
                    to_remove.append(i)
            if to_remove:
                st.session_state[topics_key] = [
                    t for j, t in enumerate(parsed_topics) if j not in to_remove
                ]
                st.rerun()

    # --- Safety status indicator ---
    st.markdown("---")
    st.markdown("#### Safety Check")

    safety_key = f"safety_{subject_id}"
    if safety_key not in st.session_state:
        st.session_state[safety_key] = None  # None = not yet checked

    if st.button("Run Safety Check", key="btn_safety_check"):
        with st.spinner("Checking subject safety..."):
            from agents.safety import SafetyAgent
            agent = SafetyAgent()
            result = agent.check_subject_safety(subject_name)
            st.session_state[safety_key] = result

    safety_result = st.session_state.get(safety_key)
    # Require an explicit safety check — default to blocked until checked
    subject_is_safe = safety_result.is_safe if safety_result is not None else False

    if safety_result is None:
        st.info("Run the safety check above before starting research.")
    elif safety_result.is_safe:
        st.success(f"Subject '{subject_name}' passed safety check.")
    else:
        st.error(f"Subject blocked: {safety_result.reason}")

    # --- Start Web Research button ---
    st.markdown("---")
    st.markdown("#### Start Research")

    topics_ready = bool(st.session_state.get(topics_key))
    button_disabled = not topics_ready or not subject_is_safe

    if not topics_ready:
        st.info("Parse topics above before starting research.")

    if st.button(
        "🔍 Start Web Research",
        key="btn_start_web_research",
        disabled=button_disabled,
        type="primary",
    ):
        final_topics = list(st.session_state[topics_key])
        _run_web_research(subject_id=subject_id, subject_name=subject_name, topics=final_topics)


def _run_web_research(subject_id: int, subject_name: str, topics: list):
    """Execute the Phase 2 pipeline: first 2 topics sync with progress bar, rest in background."""
    import uuid as _uuid
    from workflows.phase2_web_ingestion import phase2_graph

    SYNC_TOPIC_LIMIT = 2
    sync_topics = topics[:SYNC_TOPIC_LIMIT]
    bg_topics = topics[SYNC_TOPIC_LIMIT:]
    total_topics = len(topics)

    # --- Progress UI ---
    st.markdown("#### Progress")
    progress_bar = st.progress(0, text=f"Starting — 0 / {total_topics} topics")
    status_line = st.empty()

    topics_done = [0]       # mutable counter accessible inside closure
    docs_found = [0]
    flashcards_done = [0]

    def update_status(msg: str):
        status_line.markdown(msg)

    update_status(f"Running safety check for '{subject_name}'...")

    initial_state = {
        "subject_id": subject_id,
        "subject_name": subject_name,
        "topics": sync_topics,
        "web_documents": [],
        "current_doc_index": 0,
        "doc_id": "",
        "full_text": "",
        "chunks": [],
        "hierarchy": [],
        "doc_summary": "",
        "current_chunk_index": 0,
        "generated_flashcards": [],
        "status_message": "",
        "safety_blocked": False,
        "safety_reason": "",
        "processed_urls": [],
        "status_callback": update_status,
        "stop_event": None,
    }

    final_state = initial_state.copy()

    try:
        for event in phase2_graph.stream(initial_state):
            node_name = list(event.keys())[0]
            node_data = event[node_name]
            final_state.update(node_data)

            # Update progress bar based on node milestones
            if node_name == "safety_check":
                progress_bar.progress(5, text=f"Safety check passed — researching {len(sync_topics)} topic(s)...")

            elif node_name == "research":
                web_docs = final_state.get("web_documents") or []
                docs_found[0] = len(web_docs)
                progress_bar.progress(
                    20,
                    text=f"Research complete — {docs_found[0]} page(s) found for {len(sync_topics)} topic(s)",
                )

            elif node_name == "ingest_web_document":
                doc_idx = final_state.get("current_doc_index", 0)
                total_docs = max(docs_found[0], 1)
                pct = 20 + int((doc_idx / total_docs) * 60)
                progress_bar.progress(
                    min(pct, 80),
                    text=f"Ingesting page {doc_idx + 1} / {total_docs} · Topic {min(doc_idx + 1, len(sync_topics))} / {total_topics}",
                )

            elif node_name == "generate":
                all_fc = final_state.get("generated_flashcards") or []
                flashcards_done[0] = sum(1 for f in all_fc if f.get("status") == "success")
                update_status(
                    final_state.get("status_message") or f"Generating flashcards... ({flashcards_done[0]} so far)"
                )

            elif node_name == "next_document":
                topics_done[0] = final_state.get("current_doc_index", 0)

    except Exception as exc:
        st.error(f"Web research error: {exc}")
        return

    # --- Safety blocked ---
    if final_state.get("safety_blocked"):
        progress_bar.empty()
        st.error(f"Research blocked by safety guardrail: {final_state.get('safety_reason')}")
        return

    progress_bar.progress(100, text=f"Initial {len(sync_topics)} topic(s) complete!")
    update_status(f"Done. {flashcards_done[0]} flashcard(s) generated from initial topics.")

    # --- Results summary for sync phase ---
    processed_urls = final_state.get("processed_urls") or []
    web_docs_all = final_state.get("web_documents") or []
    successful_cards = flashcards_done[0]

    with st.expander(f"Results — {len(sync_topics)} topic(s) processed", expanded=True):
        st.markdown(f"**Pages ingested:** {len(processed_urls)}")
        st.markdown(f"**Flashcards generated:** {successful_cards}")
        if processed_urls:
            st.markdown("**Sources used:**")
            for url in processed_urls:
                doc_info = next((d for d in web_docs_all if d.get("url") == url), {})
                title = doc_info.get("title", url)
                domain = doc_info.get("domain", "")
                st.markdown(f"  - [{title}]({url}) — {domain}")

    # --- Hand off remaining topics to background ---
    if bg_topics:
        from core.background import start_web_background_task
        task_id = f"web_{subject_id}_{_uuid.uuid4().hex[:8]}"
        start_web_background_task(bg_topics, subject_id, subject_name, task_id)
        st.toast(
            f"Initial {len(sync_topics)} topic(s) ready! "
            f"Remaining {len(bg_topics)} topic(s) processing in background.",
            icon="🔍",
        )
    else:
        st.success("All topics processed!")

    # Clear parsed topics for this subject after launch
    st.session_state[f"web_parsed_topics_{subject_id}"] = []
    time.sleep(1)
    st.rerun()


def render_study_materials():
    st.header("📚 Study Material Ingestion")
    db = SessionLocal()
    try:

        # Use session state to manage the current selected subject persistently
        if "ingest_subject_id" not in st.session_state:
            st.session_state.ingest_subject_id = None

        # Step-based UI
        st.subheader("1. Subject Context")

        subjects = db.query(Subject).filter(Subject.is_archived == False).all()

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

        # 2. Upload / Web Research tabs
        if st.session_state.ingest_subject_id:
            subject_id = st.session_state.ingest_subject_id
            current_subject = db.get(Subject, subject_id)
            subject_name = current_subject.name if current_subject else ""
            st.subheader(f"2. Add Content to '{subject_name}'")

            tab_upload, tab_web = st.tabs(["📄 Upload Document", "🌐 Web Research"])

            with tab_upload:
                _render_upload_tab(db, subject_id, subject_name)

            with tab_web:
                _render_web_research_tab(subject_id, subject_name)

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
            is_web = task.get("is_web", False)
            display_name = task.get("display_name") or task.get("filename") or d_id[:8]

            with st.container(border=True):
                col_t, col_b = st.columns([0.85, 0.15])

                if task["status"] == "processing":
                    if is_web:
                        curr = task.get("pages_current", 0)
                        total = task.get("pages_total", 1)
                        fc = task.get("flashcards_count", 0)
                        percent = curr / total if total > 0 else 0
                        col_t.markdown(f"**🌐 Researching: {display_name}**")
                        col_t.progress(
                            percent,
                            text=f"Page {curr} / {total} · {fc} flashcard(s) generated",
                        )
                    else:
                        prog = task.get("progress", 0)
                        total = task.get("total", 1)
                        percent = prog / total if total > 0 else 0
                        col_t.markdown(f"**⏳ Processing: {display_name}**")
                        col_t.progress(percent, text=f"{prog}/{total} chunks generated")

                    if col_b.button("⏹️ Stop", key=f"stop_{d_id}"):
                        stop_background_task(d_id)
                        st.rerun()

                elif task["status"] == "completed":
                    icon = "🌐" if is_web else "✅"
                    col_t.success(f"{icon} **Completed**: {display_name}")
                    if col_b.button("✖ Clear", key=f"clear_{d_id}"):
                        del background_tasks[d_id]
                        st.rerun()

                elif task["status"] == "stopped":
                    col_t.warning(f"⏹️ **Stopped**: {display_name}")
                    if col_b.button("✖ Clear", key=f"clear_{d_id}"):
                        del background_tasks[d_id]
                        st.rerun()

                elif task["status"] == "failed":
                    col_t.error(f"❌ **Failed**: {display_name} — {task.get('error', 'Unknown Error')}")
                    if col_b.button("✖ Clear", key=f"clear_{d_id}"):
                        del background_tasks[d_id]
                        st.rerun()

def render_mentor_review():
    st.header("👨‍🏫 Mentor Review Workspace")
    db = SessionLocal()
    try:

        m_tabs = st.tabs(["⏳ Pending Review", "✅ Approved Items", "🗑️ Review Bin"])

        with m_tabs[0]:
            subjects = db.query(Subject).filter(Subject.is_archived == False).all()
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
            subjects = db.query(Subject).filter(Subject.is_archived == False).all()
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

def _get_flashcard_source_badge(db, fc) -> str:
    """Return an HTML source badge string for a flashcard.

    Looks up the source_type and source_url from ContentChunk -> Document.
    Falls back gracefully if the chunk or document record is missing.
    """
    try:
        chunk = db.query(ContentChunk).filter(ContentChunk.id == fc.chunk_id).first() if fc.chunk_id else None
        if chunk and chunk.source_type:
            if chunk.source_type == "web" and chunk.source_url:
                from urllib.parse import urlparse
                domain = urlparse(chunk.source_url).netloc.replace("www.", "")
                return (
                    f"<a href='{chunk.source_url}' target='_blank' "
                    f"style='font-size:0.78rem; color:#58a6ff; text-decoration:none;'>"
                    f"🌐 {domain}</a>"
                )
            elif chunk.source_type == "image":
                # Get filename from Document
                if chunk.document_id:
                    doc = db.query(DBDocument).filter(DBDocument.id == chunk.document_id).first()
                    fname = doc.filename if doc else "image"
                    return f"<span style='font-size:0.78rem; color:#8b949e;'>🖼️ {fname}</span>"
            else:
                # pdf or text — get filename from Document
                if chunk.document_id:
                    doc = db.query(DBDocument).filter(DBDocument.id == chunk.document_id).first()
                    fname = doc.filename if doc else "document"
                    return f"<span style='font-size:0.78rem; color:#8b949e;'>📄 {fname}</span>"
    except Exception:
        pass
    return ""


def render_flashcard_review_card(db, fc, current_status):
    source_badge = _get_flashcard_source_badge(db, fc)
    source_html = (
        f"<div style='margin-top:6px;'>{source_badge}</div>"
        if source_badge else ""
    )
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
        {source_html}
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
    try:
        db = SessionLocal()
        from core.database import Document as DBDocument

        st.warning("These actions are destructive and cannot be undone.")

        if st.button("🚨 Global Reset: Wipe Database & Qdrant Collections"):
            reset_entire_system()

        st.divider()
        st.divider()
        
        sys_tabs = st.tabs(["🟢 Active Subjects", "📦 Archived Subjects"])
        
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

                    # Topic Editing within Subject
                    topics = db.query(Topic).filter(Topic.subject_id == subj.id).all()
                    
                    if topics:
                        st.markdown("**(Topics)**")
                    for topic in topics:
                        tcol1, tcol2, tcol3 = st.columns([0.6, 0.2, 0.2])
                        new_topic_name = tcol1.text_input(f"   ↳ Edit Topic Name", value=topic.name, key=f"edit_top_{topic.id}")
                        if tcol2.button("Update Topic", key=f"upd_top_{topic.id}"):
                            topic.name = new_topic_name
                            db.commit()
                            st.success("Topic updated!")
                            st.rerun()
                        if tcol3.button("Delete Topic", key=f"del_top_{topic.id}"):
                            doc = db.query(DBDocument).filter(DBDocument.id == topic.document_id).first()
                            delete_topic_data(topic.id, doc.id)
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
                        # Recursive delete
                        documents = db.query(DBDocument).filter(DBDocument.subject_id == subj.id).all()
                        for d in documents:
                            topics = db.query(Topic).filter(Topic.document_id == d.id).all()
                            for t in topics:
                                delete_topic_data(t.id, d.id)
                            db.delete(d)
                        db.delete(subj)
                        db.commit()
                        st.success(f"Permanently deleted '{subj.name}'.")
                        st.rerun()

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
