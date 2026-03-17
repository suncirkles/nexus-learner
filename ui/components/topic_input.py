"""
ui/components/topic_input.py
------------------------------
Reusable topic-input UI component (type or upload).
Moved verbatim from app.py — zero behaviour change.
"""

import os
import tempfile
import streamlit as st


def render_topic_input_section(subject_id: int, suffix: str):
    """Reusable UI component for topic input (type or upload)."""
    st.markdown("#### Topic Input")

    topics_key = f"parsed_topics_{subject_id}"
    if topics_key not in st.session_state:
        st.session_state[topics_key] = []

    input_mode = st.radio(
        "How would you like to provide topics?",
        ["✏️ Type topics", "📄 Upload topic file"],
        horizontal=True,
        key=f"input_mode_{subject_id}_{suffix}",
    )

    if input_mode == "✏️ Type topics":
        topic_text = st.text_area(
            "Enter topics (one per line, or separated by commas)",
            height=150,
            key=f"topic_text_{subject_id}_{suffix}",
            placeholder="e.g.\nBinary Search Trees\nGraph Traversal Algorithms\nDynamic Programming",
        )
        if st.button("Parse Topics", key=f"btn_parse_text_{subject_id}_{suffix}"):
            if topic_text.strip():
                with st.spinner("Parsing topics..."):
                    from agents.topic_parser import TopicParserAgent
                    parser = TopicParserAgent()
                    st.session_state[topics_key] = parser.parse_topics_from_text(topic_text)
                st.rerun()
            else:
                st.warning("Please enter some topics first.")

    else:
        topic_file = st.file_uploader(
            "Upload .txt, .pdf, or .docx",
            type=["txt", "pdf", "docx"],
            key=f"topic_file_{subject_id}_{suffix}",
        )
        if topic_file and st.button("Parse Topics from File", key=f"btn_parse_file_{subject_id}_{suffix}"):
            suffix_ext = os.path.splitext(topic_file.name)[1].lstrip(".")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix_ext}") as tmp:
                tmp.write(topic_file.getbuffer())
                tmp_path = tmp.name
            try:
                with st.spinner("Extracting and parsing topics..."):
                    from agents.topic_parser import TopicParserAgent
                    parser = TopicParserAgent()
                    st.session_state[topics_key] = parser.parse_topics_from_file(tmp_path, suffix_ext)
                st.rerun()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    parsed_topics = st.session_state[topics_key]
    if parsed_topics:
        with st.expander(f"📋 Topics identified ({len(parsed_topics)})", expanded=True):
            to_remove = []
            for i, topic in enumerate(parsed_topics):
                col_t, col_x = st.columns([0.9, 0.1])
                col_t.markdown(f"- {topic}")
                if col_x.button("✕", key=f"rm_topic_{subject_id}_{i}_{suffix}"):
                    to_remove.append(i)
            if to_remove:
                st.session_state[topics_key] = [t for j, t in enumerate(parsed_topics) if j not in to_remove]
                st.rerun()
    return st.session_state[topics_key]
