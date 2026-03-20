"""
ui/components/flashcard_card.py
---------------------------------
Shared flashcard rendering components used by mentor.py and learner.py.
Moved verbatim from app.py — zero behaviour change.
"""

import html as _html
import os
import json
import logging
from urllib.parse import urlparse
import streamlit as st

from ui import api_client

logger = logging.getLogger(__name__)

_QTYPE_DISPLAY = {
    "active_recall": "Active Recall",
    "fill_blank": "Fill in the Blank",
    "short_answer": "Short Answer",
    "long_answer": "Long Answer",
    "numerical": "Numerical",
    "scenario": "Scenario",
}


def _resolve_source(chunk_id: int | None) -> dict | None:
    """Fetch chunk source once — shared by badge, attribution, and snippet helpers."""
    if not chunk_id:
        return None
    return api_client.get_chunk_source(chunk_id)


def _format_source_attribution(src: dict | None) -> str:
    """Return a short plain-text source attribution string for the Learner view."""
    if not src:
        return ""
    source_type = src.get("source_type")
    if source_type == "web" and src.get("source_url"):
        domain = urlparse(src["source_url"]).netloc.replace("www.", "")
        return f"🌐 {domain}"
    elif source_type == "image":
        return f"🖼️ {src.get('filename') or 'image'}"
    else:
        return f"📄 {src.get('filename') or 'document'}"


def _format_source_badge(src: dict | None) -> str:
    """Return an HTML source badge string."""
    if not src:
        return ""
    source_type = src.get("source_type")
    if source_type == "web" and src.get("source_url"):
        domain = urlparse(src["source_url"]).netloc.replace("www.", "")
        return (
            f"<a href='{src['source_url']}' target='_blank' "
            f"style='font-size:0.78rem; color:#58a6ff; text-decoration:none;'>"
            f"🌐 {domain}</a>"
        )
    elif source_type == "image":
        return f"<span style='font-size:0.78rem; color:#8b949e;'>🖼️ {src.get('filename') or 'image'}</span>"
    else:
        return f"<span style='font-size:0.78rem; color:#8b949e;'>📄 {src.get('filename') or 'document'}</span>"


_PAGE_SIZE = 20


def render_flashcard_list(sub_id, status="approved"):
    page_key = f"fc_page_{sub_id}_{status}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    page = st.session_state[page_key]

    cache_key = f"fc_cache_{sub_id}_{status}_{page}"
    if cache_key not in st.session_state:
        fcs = api_client.get_flashcards_by_subtopic(
            sub_id, status, skip=page * _PAGE_SIZE, limit=_PAGE_SIZE
        )
        st.session_state[cache_key] = fcs
    else:
        fcs = st.session_state[cache_key]

    if not fcs and page == 0:
        st.info("No flashcards found.")
        return

    # Batch fetch all chunk sources in one call.
    chunk_ids = [fc["chunk_id"] for fc in fcs if fc.get("chunk_id")]
    sources_map: dict = {}
    if chunk_ids:
        sources_map = api_client.get_chunk_sources_batch(chunk_ids)

    for fc in fcs:
        if status == "approved":
            src = sources_map.get(str(fc.get("chunk_id"))) if fc.get("chunk_id") else None
            source_attr = _format_source_attribution(src)
            source_html = (
                f"<div style='margin-top:8px; font-size:0.78rem; color:#8b949e;'>Source: {source_attr}</div>"
                if source_attr else ""
            )
            st.markdown(f"""
            <div class='stCard'>
                <strong>Q: {fc['question']}</strong>
                <hr style='border-top: 1px solid #30363d; margin: 10px 0;'/>
                <details style='cursor: pointer;'>
                    <summary style='color: #58a6ff;'>Show Answer</summary>
                    <p style='margin-top: 10px;'>{fc['answer']}</p>
                    {source_html}
                </details>
            </div>
            """, unsafe_allow_html=True)
        else:
            _render_review_card_with_cache(fc, status, cache_key, sources_map)

    # Pagination controls
    prev_col, _, next_col = st.columns([0.15, 0.7, 0.15])
    has_next = len(fcs) == _PAGE_SIZE
    if page > 0:
        if prev_col.button("← Prev", key=f"fc_prev_{sub_id}_{status}"):
            st.session_state[page_key] = page - 1
            # Don't clear cache for previous page — it may already be cached.
    if has_next:
        if next_col.button("Next →", key=f"fc_next_{sub_id}_{status}"):
            st.session_state[page_key] = page + 1


def _invalidate_fc_cache(sub_id, status):
    """Remove all page caches for a subtopic+status so they re-fetch on next render."""
    keys_to_clear = [k for k in st.session_state if k.startswith(f"fc_cache_{sub_id}_{status}_")]
    for k in keys_to_clear:
        del st.session_state[k]


def _render_review_card_with_cache(fc: dict, current_status: str, cache_key: str, sources_map: dict):
    """Wrapper around render_flashcard_review_card that uses pre-fetched sources and handles
    optimistic removal from the session-state cache after status changes."""
    fc_id = fc["id"]
    sub_id = fc.get("subtopic_id")

    # Override _resolve_source via pre-fetched map for this card.
    chunk_id = fc.get("chunk_id")
    src = sources_map.get(str(chunk_id)) if chunk_id else None

    _render_review_card_inner(fc, current_status, src, cache_key, sub_id)


def _render_review_card_inner(fc: dict, current_status: str, src, cache_key: str, sub_id):
    """Like render_flashcard_review_card but accepts pre-fetched src and does optimistic cache mutations."""
    fc_id = fc["id"]
    fc_question = fc["question"]
    fc_answer = fc["answer"]
    fc_critic_score = fc.get("critic_score") or 0
    fc_critic_feedback = fc.get("critic_feedback") or ""
    fc_question_type = fc.get("question_type") or "active_recall"
    fc_chunk_id = fc.get("chunk_id")
    fc_rubric_json = fc.get("rubric")
    fc_rubric_scores_json = fc.get("critic_rubric_scores")
    fc_complexity = fc.get("complexity_level") or "(unset)"

    qtype_label = _QTYPE_DISPLAY.get(fc_question_type, fc_question_type)
    score_label = (
        f"⚠️ {fc_critic_score}/4 low quality"
        if fc_critic_score and fc_critic_score < 2
        else f"{fc_critic_score}/4"
    )

    source_badge = _format_source_badge(src)
    source_html = f"<div style='margin-top:6px;'>{source_badge}</div>" if source_badge else ""

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

    if src:
        page_num = src.get("page_number")
        doc_id_for_img = src.get("document_id")
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
        elif src.get("text"):
            snippet = src["text"][:1500] + ("…" if len(src["text"]) > 1500 else "")
            st.markdown(
                f"<details><summary style='cursor:pointer; color:#58a6ff;'>📄 Source Snippet</summary>"
                f"<pre style='white-space:pre-wrap; font-size:0.8rem; margin-top:8px;'>"
                f"{_html.escape(snippet)}</pre></details>",
                unsafe_allow_html=True,
            )

    if fc_rubric_json:
        try:
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
            api_client.update_flashcard_status(
                fc_id, "approved",
                complexity_level=None if selected_complexity == "(unset)" else selected_complexity,
            )
            # Optimistic: remove from current cache, invalidate approved cache.
            if cache_key in st.session_state:
                st.session_state[cache_key] = [c for c in st.session_state[cache_key] if c["id"] != fc_id]
            if sub_id:
                _invalidate_fc_cache(sub_id, "approved")
            st.rerun()
        if cols[1].button("Reject", key=f"p_rej_{fc_id}"):
            api_client.update_flashcard_status(fc_id, "rejected")
            if cache_key in st.session_state:
                st.session_state[cache_key] = [c for c in st.session_state[cache_key] if c["id"] != fc_id]
            if sub_id:
                _invalidate_fc_cache(sub_id, "rejected")
            st.rerun()
    elif current_status == "approved":
        if cols[0].button("Discard", key=f"a_disc_{fc_id}"):
            api_client.update_flashcard_status(fc_id, "rejected")
            if cache_key in st.session_state:
                st.session_state[cache_key] = [c for c in st.session_state[cache_key] if c["id"] != fc_id]
            if sub_id:
                _invalidate_fc_cache(sub_id, "rejected")
            st.rerun()
    elif current_status == "rejected":
        if cols[0].button("Restore", key=f"r_res_{fc_id}"):
            api_client.update_flashcard_status(fc_id, "pending")
            if cache_key in st.session_state:
                st.session_state[cache_key] = [c for c in st.session_state[cache_key] if c["id"] != fc_id]
            if sub_id:
                _invalidate_fc_cache(sub_id, "pending")
            st.rerun()
        if cols[1].button("Delete", key=f"r_del_{fc_id}"):
            api_client.delete_flashcard(fc_id)
            # Optimistic: remove from cache — no list re-fetch needed.
            if cache_key in st.session_state:
                st.session_state[cache_key] = [c for c in st.session_state[cache_key] if c["id"] != fc_id]
            st.rerun()

        with st.expander("🔄 Recreate with Feedback", expanded=False):
            feedback_key = f"fb_text_{fc_id}"
            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = ""

            if st.button("💡 Get Suggested Answer", key=f"get_sug_{fc_id}"):
                from agents.socratic import SocraticAgent
                s_agent = SocraticAgent()
                try:
                    with st.spinner("LLM is thinking..."):
                        suggestion = s_agent.suggest_answer(fc_question, fc_id)
                    suggestion_text = f"Suggested Answer: {suggestion}"
                    st.session_state[feedback_key] = suggestion_text
                    st.session_state[f"actual_fb_{fc_id}"] = suggestion_text
                except Exception as _e:
                    st.error(f"Could not get suggestion: {_e}")

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
                            if cache_key in st.session_state:
                                st.session_state[cache_key] = [c for c in st.session_state[cache_key] if c["id"] != fc_id]
                            st.success("Flashcard updated and moved to Pending.")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(res.get("message", "Error recreating flashcard."))
                else:
                    st.warning("Please provide feedback first.")


def render_flashcard_review_card(fc: dict, current_status: str, src=None):
    # C1/C2: Capture all display fields as primitives immediately.
    fc_id = fc["id"]
    fc_question = fc["question"]
    fc_answer = fc["answer"]
    fc_critic_score = fc.get("critic_score") or 0
    fc_critic_feedback = fc.get("critic_feedback") or ""
    fc_question_type = fc.get("question_type") or "active_recall"
    fc_chunk_id = fc.get("chunk_id")
    fc_rubric_json = fc.get("rubric")
    fc_rubric_scores_json = fc.get("critic_rubric_scores")
    fc_complexity = fc.get("complexity_level") or "(unset)"

    qtype_label = _QTYPE_DISPLAY.get(fc_question_type, fc_question_type)

    score_label = (
        f"⚠️ {fc_critic_score}/4 low quality"
        if fc_critic_score and fc_critic_score < 2
        else f"{fc_critic_score}/4"
    )

    # Use pre-fetched source if provided (batch path); fall back to per-card fetch.
    if src is None:
        src = _resolve_source(fc_chunk_id)
    source_badge = _format_source_badge(src)
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

    if src:
        page_num = src.get("page_number")
        doc_id_for_img = src.get("document_id")
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
        elif src.get("text"):
            snippet = src["text"][:1500] + ("…" if len(src["text"]) > 1500 else "")
            st.markdown(
                f"<details><summary style='cursor:pointer; color:#58a6ff;'>📄 Source Snippet</summary>"
                f"<pre style='white-space:pre-wrap; font-size:0.8rem; margin-top:8px;'>"
                f"{_html.escape(snippet)}</pre></details>",
                unsafe_allow_html=True,
            )

    if fc_rubric_json:
        try:
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
            api_client.update_flashcard_status(
                fc_id, "approved",
                complexity_level=None if selected_complexity == "(unset)" else selected_complexity,
            )
            st.rerun()
        if cols[1].button("Reject", key=f"p_rej_{fc_id}"):
            api_client.update_flashcard_status(fc_id, "rejected")
            st.rerun()
    elif current_status == "approved":
        if cols[0].button("Discard", key=f"a_disc_{fc_id}"):
            api_client.update_flashcard_status(fc_id, "rejected")
            st.rerun()
    elif current_status == "rejected":
        if cols[0].button("Restore", key=f"r_res_{fc_id}"):
            api_client.update_flashcard_status(fc_id, "pending")
            st.rerun()
        if cols[1].button("Delete", key=f"r_del_{fc_id}"):
            api_client.delete_flashcard(fc_id)
            st.rerun()

        with st.expander("🔄 Recreate with Feedback", expanded=False):
            feedback_key = f"fb_text_{fc_id}"
            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = ""

            if st.button("💡 Get Suggested Answer", key=f"get_sug_{fc_id}"):
                from agents.socratic import SocraticAgent
                s_agent = SocraticAgent()
                try:
                    with st.spinner("LLM is thinking..."):
                        suggestion = s_agent.suggest_answer(fc_question, fc_id)
                    suggestion_text = f"Suggested Answer: {suggestion}"
                    st.session_state[feedback_key] = suggestion_text
                    # Must update the text area's own widget key so Streamlit
                    # reflects the new value (it ignores `value=` when key already exists).
                    st.session_state[f"actual_fb_{fc_id}"] = suggestion_text
                except Exception as _e:
                    st.error(f"Could not get suggestion: {_e}")

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
