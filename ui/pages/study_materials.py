"""
ui/pages/study_materials.py
-----------------------------
Study Materials page — document upload, ingestion, and background processing.
Moved verbatim from app.py — zero behaviour change.
"""

import os
import uuid
import time
import logging
import streamlit as st

from ui import api_client
from ui.components.topic_input import render_topic_input_section
from ui.components.background_monitor import render_study_materials_background_monitor

logger = logging.getLogger(__name__)


def _render_upload_tab(subject_id: int, subject_name: str):
    """Refactored Tab 1: Knowledge Library Attachment and Generation."""

    st.markdown("#### 🔗 Attached Library Documents")
    attached_docs = api_client.list_attached_documents(subject_id)
    attached_doc_ids = [d["id"] for d in attached_docs]

    if not attached_docs:
        st.info("No documents attached to this subject yet.")
    else:
        # Batch: 1 call for all topics in the subject, then map by document_id.
        all_topics = api_client.get_topics_by_subject(subject_id)
        topic_count_by_doc = {}
        for t in all_topics:
            doc_id = t.get("document_id")
            if doc_id:
                topic_count_by_doc[doc_id] = topic_count_by_doc.get(doc_id, 0) + 1

        for doc in attached_docs:
            col1, col2 = st.columns([0.8, 0.2])
            topic_count = topic_count_by_doc.get(doc["id"], 0)
            col1.markdown(f"- **{doc.get('title') or doc.get('filename')}** ({topic_count} topics indexed)")
            if col2.button("🗑️ Detach", key=f"detach_{subject_id}_{doc['id']}"):
                try:
                    api_client.detach_document(subject_id, doc["id"])
                    st.rerun()
                except Exception as e:
                    st.error(f"Detach failed: {e}")

    st.divider()

    st.markdown("#### ➕ Attach from Library")
    available_docs = api_client.list_available_documents(subject_id)

    if not available_docs:
        st.caption("No new documents available in library. Index them in the '📂 Knowledge Library' tab.")
    else:
        doc_options = {f"{d.get('title') or d.get('filename')}": d["id"] for d in available_docs}
        selected_doc_name = st.selectbox("Select document to attach:", [""] + list(doc_options.keys()), key=f"lib_attach_{subject_id}")
        if selected_doc_name:
            if st.button("Link to Subject", key=f"btn_link_{subject_id}"):
                try:
                    api_client.attach_document(subject_id, doc_options[selected_doc_name])
                    st.success("Document attached!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Attach failed: {e}")

    st.divider()

    st.markdown("#### 🧠 Generate Flashcards")

    # Build topic list from all topics indexed for this subject
    topics = api_client.get_topics_by_subject(subject_id)
    topic_list = list({t["name"] for t in topics})

    if not topic_list:
        st.warning("Attach a document with indexed topics first.")
    else:
        st.caption("Select specific topics to generate cards for this subject.")
        selected_topics = st.multiselect("Select Topics:", topic_list, key=f"gen_topics_{subject_id}")

        manual_topics = st.text_input("OR Enter topics manually (comma separated):", key=f"manual_topics_{subject_id}")

        _QTYPE_LABELS = {
            "Active Recall":    "active_recall",
            "Fill in the Blank": "fill_blank",
            "Short Answer":     "short_answer",
            "Long Answer":      "long_answer",
            "Numerical":        "numerical",
            "Scenario":         "scenario",
        }
        selected_qtype_label = st.selectbox(
            "Question Type",
            list(_QTYPE_LABELS.keys()),
            key=f"qtype_{subject_id}",
            help="Determines the format of the generated flashcards.",
        )
        selected_question_type = _QTYPE_LABELS[selected_qtype_label]

        if st.button("🔥 Process and Generate Cards", key=f"btn_gen_{subject_id}"):
            final_target_topics = selected_topics
            if manual_topics:
                final_target_topics += [t.strip() for t in manual_topics.split(",") if t.strip()]

            if not final_target_topics:
                st.warning("Please select or enter at least one topic.")
            elif not attached_docs:
                st.warning("No documents attached to this subject.")
            else:
                from core.background import start_background_task
                for doc in attached_docs:
                    gen_id = str(uuid.uuid4())
                    state = {
                        "mode": "GENERATION",
                        "doc_id": doc["id"],
                        "subject_id": subject_id,
                        "target_topics": final_target_topics,
                        "question_type": selected_question_type,
                        "chunks": [],
                        "current_chunk_index": 0,
                        "generated_flashcards": [],
                        "status_message": "Matching topics and identifying chunks...",
                    }
                    label = doc.get("title") or doc.get("filename")
                    start_background_task(
                        state, gen_id,
                        filename=f"Gen ({label}): {', '.join(final_target_topics[:2])}...",
                    )
                st.toast(f"Flashcard generation started for {len(attached_docs)} document(s).")
                st.rerun()


def _render_web_research_tab(subject_id: int, subject_name: str):
    """Tab 2: Web Research — topic input, safety check, and pipeline launch."""

    parsed_topics = render_topic_input_section(subject_id, suffix="web")

    st.markdown("---")
    st.markdown("#### Safety Check")

    safety_key = f"safety_{subject_id}"
    if safety_key not in st.session_state:
        st.session_state[safety_key] = None

    if st.button("Run Safety Check", key=f"btn_safety_check_{subject_id}"):
        with st.spinner("Checking subject safety..."):
            from agents.safety import SafetyAgent
            agent = SafetyAgent()
            result = agent.check_subject_safety(subject_name)
            st.session_state[safety_key] = result

    safety_result = st.session_state.get(safety_key)
    subject_is_safe = safety_result.is_safe if safety_result is not None else False

    if safety_result is None:
        st.info("Run the safety check above before starting research.")
    elif safety_result.is_safe:
        st.success(f"Subject '{subject_name}' passed safety check.")
    else:
        st.error(f"Subject blocked: {safety_result.reason}")

    st.markdown("---")
    st.markdown("#### Start Research")

    topics_ready = bool(parsed_topics)
    button_disabled = not topics_ready or not subject_is_safe

    if not topics_ready:
        st.info("Parse topics above before starting research.")

    if st.button(
        "🔍 Start Web Research",
        key=f"btn_start_web_research_{subject_id}",
        disabled=button_disabled,
        type="primary",
    ):
        final_topics = list(parsed_topics)
        _run_web_research(subject_id=subject_id, subject_name=subject_name, topics=final_topics)


def _run_web_research(subject_id: int, subject_name: str, topics: list):
    """Execute the Phase 2 pipeline: first 2 topics sync with progress bar, rest in background."""
    import uuid as _uuid
    from workflows.phase2_web_ingestion import phase2_graph

    SYNC_TOPIC_LIMIT = 2
    sync_topics = topics[:SYNC_TOPIC_LIMIT]
    bg_topics = topics[SYNC_TOPIC_LIMIT:]
    total_topics = len(topics)

    st.markdown("#### Progress")
    progress_bar = st.progress(0, text=f"Starting — 0 / {total_topics} topics")
    status_line = st.empty()

    topics_done = [0]
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

    if final_state.get("safety_blocked"):
        progress_bar.empty()
        st.error(f"Research blocked by safety guardrail: {final_state.get('safety_reason')}")
        return

    progress_bar.progress(100, text=f"Initial {len(sync_topics)} topic(s) complete!")
    update_status(f"Done. {flashcards_done[0]} flashcard(s) generated from initial topics.")

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

    st.session_state[f"parsed_topics_{subject_id}"] = []
    time.sleep(1)
    st.rerun()


def render_study_materials():
    st.header("📚 Study Material Ingestion")

    if "ingest_subject_id" not in st.session_state:
        st.session_state.ingest_subject_id = None

    st.subheader("1. Subject Context")

    subjects = api_client.list_active_subjects()

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
                        new_subj = api_client.create_subject(cleaned_name)
                        st.session_state.ingest_subject_id = new_subj["id"]
                        st.success(f"Subject '{cleaned_name}' created and selected!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        subject_names = [s["name"] for s in subjects]
        current_idx = 0
        if st.session_state.ingest_subject_id:
            for i, s in enumerate(subjects):
                if s["id"] == st.session_state.ingest_subject_id:
                    current_idx = i
                    break

        selected_name = st.selectbox("Choose Subject", subject_names, index=current_idx)
        selected_subj = next((s for s in subjects if s["name"] == selected_name), None)
        if selected_subj:
            st.session_state.ingest_subject_id = selected_subj["id"]
            st.info(f"Adding artifacts to: **{selected_name}**")

    st.divider()

    if st.session_state.ingest_subject_id:
        subject_id = st.session_state.ingest_subject_id
        current_subject = next((s for s in subjects if s["id"] == subject_id), None)
        subject_name = current_subject["name"] if current_subject else ""
        st.subheader(f"2. Add Content to '{subject_name}'")

        tab_upload, tab_web = st.tabs(["📄 Upload Document", "🌐 Web Research"])

        with tab_upload:
            _render_upload_tab(subject_id, subject_name)

        with tab_web:
            _render_web_research_tab(subject_id, subject_name)

    else:
        st.warning("Please select or create a subject above before uploading artifacts.")

    render_study_materials_background_monitor()
