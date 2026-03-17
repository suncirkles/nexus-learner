"""
ui/components/flashcard_card.py
---------------------------------
Shared flashcard rendering components used by mentor.py and learner.py.
Moved verbatim from app.py — zero behaviour change.
"""

import os
import json
import logging
import streamlit as st

from core.database import SessionLocal, Flashcard, ContentChunk, Document as DBDocument
from core.database import get_session

logger = logging.getLogger(__name__)


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


def _get_flashcard_source_badge(db, fc) -> str:
    """Return an HTML source badge string for a flashcard."""
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
                if chunk.document_id:
                    doc = db.query(DBDocument).filter(DBDocument.id == chunk.document_id).first()
                    fname = doc.filename if doc else "image"
                    return f"<span style='font-size:0.78rem; color:#8b949e;'>🖼️ {fname}</span>"
            else:
                if chunk.document_id:
                    doc = db.query(DBDocument).filter(DBDocument.id == chunk.document_id).first()
                    fname = doc.filename if doc else "document"
                    return f"<span style='font-size:0.78rem; color:#8b949e;'>📄 {fname}</span>"
    except Exception:
        pass
    return ""


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


def render_flashcard_review_card(db, fc, current_status):
    # C1/C2: Capture all display fields as primitives immediately.
    fc_id = fc.id
    fc_question = fc.question
    fc_answer = fc.answer
    fc_critic_score = fc.critic_score or 0
    fc_critic_feedback = fc.critic_feedback or ""
    fc_question_type = getattr(fc, "question_type", None) or "active_recall"
    fc_chunk_id = getattr(fc, "chunk_id", None)
    fc_rubric_json = getattr(fc, "rubric", None)
    fc_rubric_scores_json = getattr(fc, "critic_rubric_scores", None)
    fc_complexity = getattr(fc, "complexity_level", None) or "(unset)"

    _QTYPE_DISPLAY = {
        "active_recall": "Active Recall",
        "fill_blank": "Fill in the Blank",
        "short_answer": "Short Answer",
        "long_answer": "Long Answer",
        "numerical": "Numerical",
        "scenario": "Scenario",
    }
    qtype_label = _QTYPE_DISPLAY.get(fc_question_type, fc_question_type)

    score_label = (
        f"⚠️ {fc_critic_score}/4 low quality"
        if fc_critic_score and fc_critic_score < 2
        else f"{fc_critic_score}/4"
    )

    source_badge = _get_flashcard_source_badge(db, fc)
    source_html = (
        f"<div style='margin-top:6px;'>{source_badge}</div>"
        if source_badge else ""
    )

    st.markdown(f"""
    <div class='stCard'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <strong>Question:</strong>
            <div style='display:flex; gap:8px; align-items:center;'>
                <span style='font-size:0.75rem; background:#1f3a5c; color:#58a6ff;
                             padding:2px 8px; border-radius:12px; font-weight:600;'>
                    {qtype_label}
                </span>
                <span class='critic-score'>Score: {score_label}</span>
            </div>
        </div>
        <p>{fc_question}</p>
        <hr style='border-top: 1px solid #30363d; margin: 10px 0;'/>
        <strong>Answer:</strong>
        <p>{fc_answer}</p>
        <div style='font-size: 0.85rem; color: #8b949e;'>
            <em>Critic Feedback: {fc_critic_feedback}</em>
        </div>
        {source_html}
    </div>
    """, unsafe_allow_html=True)

    if fc_rubric_scores_json:
        try:
            rs = json.loads(fc_rubric_scores_json)
            score_cols = st.columns(4)
            for col, (label, key) in zip(score_cols, [
                ("Accuracy", "accuracy"), ("Logic", "logic"),
                ("Grounding", "grounding"), ("Clarity", "clarity"),
            ]):
                val = rs.get(key, "—")
                col.metric(label, f"{val}/4")
        except Exception:
            pass

    if fc_chunk_id:
        chunk = db.query(ContentChunk).filter(ContentChunk.id == fc_chunk_id).first()
        if chunk:
            import html as _html
            page_num = getattr(chunk, "page_number", None)
            doc_id_for_img = chunk.document_id
            img_path = (
                os.path.join("page_cache", f"{doc_id_for_img}_p{page_num:04d}.png")
                if page_num is not None and doc_id_for_img
                else None
            )
            has_image = img_path and os.path.exists(img_path)

            if has_image:
                import base64 as _b64
                with open(img_path, "rb") as _f:
                    _img_b64 = _b64.b64encode(_f.read()).decode()
                st.markdown(
                    f"<details><summary style='cursor:pointer; color:#58a6ff;'>📄 Source Page</summary>"
                    f"<img src='data:image/png;base64,{_img_b64}' "
                    f"style='width:100%; margin-top:8px; border-radius:4px;'/>"
                    f"<div style='font-size:0.75rem; color:#8b949e; margin-top:4px;'>"
                    f"Page {page_num + 1}</div></details>",
                    unsafe_allow_html=True,
                )
            else:
                snippet = chunk.text[:1500] + ("…" if len(chunk.text) > 1500 else "")
                st.markdown(
                    f"<details><summary style='cursor:pointer; color:#58a6ff;'>📄 Source Snippet</summary>"
                    f"<pre style='white-space:pre-wrap; font-size:0.8rem; margin-top:8px;'>"
                    f"{_html.escape(snippet)}</pre></details>",
                    unsafe_allow_html=True,
                )

    if fc_rubric_json:
        try:
            import html as _html
            rubric_items = json.loads(fc_rubric_json)
            rows = "".join(
                f"<p style='margin:4px 0;'><strong>{_html.escape(item.get('criterion','?'))}</strong>"
                f" — {_html.escape(item.get('description',''))}</p>"
                for item in rubric_items
            )
            st.markdown(
                f"<details><summary style='cursor:pointer; color:#58a6ff;'>📋 Grading Rubric</summary>"
                f"<div style='margin-top:8px;'>{rows}</div></details>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    cols = st.columns([0.15, 0.15, 0.15, 0.55])

    if current_status == "pending":
        complexity_key = f"complexity_{fc_id}"
        complexity_options = ["(unset)", "simple", "medium", "complex"]
        default_idx = complexity_options.index(fc_complexity) if fc_complexity in complexity_options else 0
        selected_complexity = st.selectbox(
            "Complexity",
            complexity_options,
            index=default_idx,
            key=complexity_key,
        )

        if cols[0].button("Approve", key=f"p_app_{fc_id}"):
            with get_session() as _db:
                _fc = _db.query(Flashcard).filter(Flashcard.id == fc_id).first()
                if _fc:
                    _fc.status = "approved"
                    _fc.complexity_level = None if selected_complexity == "(unset)" else selected_complexity
                    _db.commit()
            st.rerun()
        if cols[1].button("Reject", key=f"p_rej_{fc_id}"):
            with get_session() as _db:
                _fc = _db.query(Flashcard).filter(Flashcard.id == fc_id).first()
                if _fc:
                    _fc.status = "rejected"
                    _db.commit()
            st.rerun()
    elif current_status == "approved":
        if cols[0].button("Discard", key=f"a_disc_{fc_id}"):
            with get_session() as _db:
                _fc = _db.query(Flashcard).filter(Flashcard.id == fc_id).first()
                if _fc:
                    _fc.status = "rejected"
                    _db.commit()
            st.rerun()
    elif current_status == "rejected":
        if cols[0].button("Restore", key=f"r_res_{fc_id}"):
            with get_session() as _db:
                _fc = _db.query(Flashcard).filter(Flashcard.id == fc_id).first()
                if _fc:
                    _fc.status = "pending"
                    _db.commit()
            st.rerun()
        if cols[1].button("Delete", key=f"r_del_{fc_id}"):
            with get_session() as _db:
                _fc = _db.query(Flashcard).filter(Flashcard.id == fc_id).first()
                if _fc:
                    _db.delete(_fc)
                    _db.commit()
            st.rerun()

        with st.expander("🔄 Recreate with Feedback", expanded=False):
            feedback_key = f"fb_text_{fc_id}"
            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = ""

            if st.button("💡 Get Suggested Answer", key=f"get_sug_{fc_id}"):
                from agents.socratic import SocraticAgent
                s_agent = SocraticAgent()
                with st.spinner("LLM is thinking..."):
                    suggestion = s_agent.suggest_answer(fc_question, fc_id)
                    st.session_state[feedback_key] = f"Suggested Answer: {suggestion}"
                    st.rerun()

            feedback = st.text_area("What needs to be fixed?", value=st.session_state[feedback_key], key=f"actual_fb_{fc_id}")
            if st.button("Regenerate Flashcard", key=f"fb_btn_{fc_id}"):
                if feedback:
                    from agents.socratic import SocraticAgent
                    import time
                    s_agent = SocraticAgent()
                    with st.spinner("Generating better Q&A..."):
                        res = s_agent.recreate_flashcard(fc_id, feedback)
                        if res["status"] == "success":
                            del st.session_state[feedback_key]
                            st.success("Flashcard updated and moved to Pending.")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(res.get("message", "Error recreating flashcard."))
                else:
                    st.warning("Please provide feedback first.")
