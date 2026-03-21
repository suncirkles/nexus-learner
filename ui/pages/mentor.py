"""
ui/pages/mentor.py
--------------------
Mentor Review — progressive, lazy card loading.

Loading model
-------------
1. Subject list  — 1 call on every render (lightweight list endpoint).
2. Topic tree    — 1 call per selected subject, cached 30 s in session state.
3. Cards         — fetched on demand when a subtopic is selected; first page
                   shows a spinner, subsequent pages show spinner via
                   render_flashcard_list.

Navigation
----------
Uses st.radio instead of st.tabs so only the active section's code executes
on each rerun.  st.tabs renders all tab content on every script run, which
causes local render work (widget-tree construction) that manifests as a
spinner with no backend activity.
"""

import streamlit as st

from ui import api_client
from ui.components.flashcard_card import render_flashcard_list, render_flashcard_review_card


_ALL_QTYPES = [
    "active_recall", "fill_blank", "short_answer",
    "long_answer", "numerical", "scenario",
]
_PAGE_SIZE = 20


# ---------------------------------------------------------------------------
# Tree cache helpers
# ---------------------------------------------------------------------------

def _load_subject_tree(subject_id: int) -> list:
    """Return topic tree for subject from 30-second session-state cache."""
    cache_key = f"mentor_tree_{subject_id}"
    return api_client.get_cached(
        cache_key, ttl_seconds=30,
        fetch_fn=lambda: [
            {"topic": t, "subtopics": t.get("subtopics", [])}
            for t in api_client.get_topic_tree(subject_id)
        ],
    )


def _invalidate_tree(subject_id: int) -> None:
    """Clear tree + topic/subtopic selectbox state so they reload cleanly."""
    st.session_state.pop(f"mentor_tree_{subject_id}", None)
    st.session_state.pop(f"mentor_tree_{subject_id}__ts", None)
    # Clear topic/subtopic selection — the previously selected item may no
    # longer be present after a bulk approve/reject.
    for key in ("mentor_pending_topic", "mentor_pending_subtopic",
                "mentor_approved_topic", "mentor_approved_subtopic"):
        st.session_state.pop(key, None)


def _invalidate_fc_cache(sub_id: int, status: str) -> None:
    """Remove all page caches for a subtopic+status."""
    keys = [k for k in st.session_state
            if k.startswith(f"fc_cache_{sub_id}_{status}_")]
    for k in keys:
        del st.session_state[k]


# ---------------------------------------------------------------------------
# Pending Review tab
# ---------------------------------------------------------------------------

def _render_pending_tab(subject_id: int) -> None:
    # Question-type filter — placed above topic selector so it's always visible.
    type_filter: list = st.multiselect(
        "Filter by Question Type (empty = show all)",
        _ALL_QTYPES,
        key="mentor_pending_type_filter",
    )

    topics_data = _load_subject_tree(subject_id)

    pending_topics = [
        td for td in topics_data
        if any(s.get("pending_count", 0) > 0 for s in td["subtopics"])
    ]

    if not pending_topics:
        st.success("All caught up — no pending cards to review.")
        return

    # --- Topic selector ---
    topic_options = {td["topic"]["name"]: td for td in pending_topics}
    selected_topic_name = st.selectbox(
        "Topic", list(topic_options.keys()), key="mentor_pending_topic",
    )
    td = topic_options[selected_topic_name]
    topic = td["topic"]
    all_sub_ids = [s["id"] for s in td["subtopics"]]

    # Topic-level bulk actions
    t_col1, t_col2, _ = st.columns([0.22, 0.22, 0.56])
    if t_col1.button("Approve All Topic", key=f"app_topic_{topic['id']}"):
        with st.spinner("Approving all cards in topic..."):
            api_client.bulk_subtopic_action(all_sub_ids, "approve")
        for sid in all_sub_ids:
            _invalidate_fc_cache(sid, "pending")
        _invalidate_tree(subject_id)
        st.toast("Approved all pending cards for this topic.")
        st.rerun()
    if t_col2.button("Reject All Topic", key=f"rej_topic_{topic['id']}"):
        with st.spinner("Rejecting all cards in topic..."):
            api_client.bulk_subtopic_action(all_sub_ids, "reject")
        for sid in all_sub_ids:
            _invalidate_fc_cache(sid, "pending")
        _invalidate_tree(subject_id)
        st.toast("Rejected all pending cards for this topic.")
        st.rerun()

    st.divider()

    # --- Subtopic selector ---
    pending_subs = [s for s in td["subtopics"] if s.get("pending_count", 0) > 0]
    if not pending_subs:
        st.info("No pending subtopics in this topic.")
        return

    sub_options = {
        f"{s['name']}  ({s.get('pending_count', 0)} pending)": s
        for s in pending_subs
    }
    selected_sub_label = st.selectbox(
        "Subtopic", list(sub_options.keys()), key="mentor_pending_subtopic",
    )
    sub = sub_options[selected_sub_label]

    # Subtopic bulk actions
    b_col1, b_col2, _ = st.columns([0.18, 0.18, 0.64])
    if b_col1.button("Approve All", key=f"app_all_{sub['id']}"):
        with st.spinner("Approving..."):
            api_client.bulk_subtopic_action([sub["id"]], "approve")
        _invalidate_fc_cache(sub["id"], "pending")
        _invalidate_tree(subject_id)
        st.toast(f"Approved all pending cards for {sub['name']}.")
        st.rerun()
    if b_col2.button("Reject All", key=f"rej_all_{sub['id']}"):
        with st.spinner("Rejecting..."):
            api_client.bulk_subtopic_action([sub["id"]], "reject")
        _invalidate_fc_cache(sub["id"], "pending")
        _invalidate_tree(subject_id)
        st.toast(f"Rejected all pending cards for {sub['name']}.")
        st.rerun()

    # Cards — render_flashcard_list shows its own spinner on uncached pages.
    render_flashcard_list(
        sub["id"], "pending",
        question_types=type_filter if type_filter else None,
    )


# ---------------------------------------------------------------------------
# Approved tab
# ---------------------------------------------------------------------------

def _render_approved_tab(subject_id: int) -> None:
    topics_data = _load_subject_tree(subject_id)

    approved_topics = [
        td for td in topics_data
        if any(s.get("approved_count", 0) > 0 for s in td["subtopics"])
    ]

    if not approved_topics:
        st.info("No approved cards yet for this subject.")
        return

    # --- Topic selector ---
    topic_options = {td["topic"]["name"]: td for td in approved_topics}
    selected_topic_name = st.selectbox(
        "Topic", list(topic_options.keys()), key="mentor_approved_topic",
    )
    td = topic_options[selected_topic_name]

    # --- Subtopic selector ---
    approved_subs = [s for s in td["subtopics"] if s.get("approved_count", 0) > 0]
    if not approved_subs:
        st.info("No approved subtopics in this topic.")
        return

    sub_options = {
        f"{s['name']}  ({s.get('approved_count', 0)} approved)": s
        for s in approved_subs
    }
    selected_sub_label = st.selectbox(
        "Subtopic", list(sub_options.keys()), key="mentor_approved_subtopic",
    )
    sub = sub_options[selected_sub_label]

    # Cards — spinner handled inside render_flashcard_list.
    render_flashcard_list(sub["id"], "approved")


# ---------------------------------------------------------------------------
# Review Bin (Rejected) tab
# ---------------------------------------------------------------------------

def _render_rejected_tab() -> None:
    st.caption("Items here can be recreated with feedback or permanently deleted.")
    with st.spinner("Loading rejected cards..."):
        rejected_fcs = api_client.get_all_rejected_flashcards()

    if not rejected_fcs:
        st.info("Review bin is empty.")
        return

    chunk_ids = [fc["chunk_id"] for fc in rejected_fcs if fc.get("chunk_id")]
    sources_map = api_client.get_chunk_sources_batch(chunk_ids) if chunk_ids else {}

    st.caption(f"{len(rejected_fcs)} card(s) in review bin.")
    for fc in rejected_fcs:
        src = sources_map.get(str(fc.get("chunk_id"))) if fc.get("chunk_id") else None
        render_flashcard_review_card(fc, "rejected", src=src)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_mentor_review():
    st.header("Mentor Review Workspace")

    subjects = api_client.list_active_subjects()
    if not subjects:
        st.info("No materials available for review.")
        return

    subject_by_name = {s["name"]: s for s in subjects}

    selected_name = st.selectbox(
        "Subject", list(subject_by_name.keys()), key="mentor_selected_subject",
    )
    selected_subj = subject_by_name[selected_name]
    subject_id = selected_subj["id"]

    # Reset per-subject navigation state when the subject changes.
    if st.session_state.get("_mentor_last_subject") != subject_id:
        for key in ("mentor_pending_topic", "mentor_pending_subtopic",
                    "mentor_approved_topic", "mentor_approved_subtopic"):
            st.session_state.pop(key, None)
        st.session_state["_mentor_last_subject"] = subject_id

    # Session-state radio instead of st.tabs — only the selected section's
    # code runs on each rerun, preventing spurious widget-tree work.
    section = st.radio(
        "Section",
        ["Pending Review", "Approved", "Review Bin"],
        horizontal=True,
        key="mentor_tab",
        label_visibility="collapsed",
    )
    st.divider()

    if section == "Pending Review":
        _render_pending_tab(subject_id)
    elif section == "Approved":
        _render_approved_tab(subject_id)
    else:
        _render_rejected_tab()
