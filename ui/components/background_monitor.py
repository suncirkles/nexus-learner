"""
ui/components/background_monitor.py
--------------------------------------
Background task monitor fragment for the sidebar and study materials page.
Moved verbatim from app.py — zero behaviour change.
"""

import streamlit as st


@st.fragment(run_every=5)
def _sidebar_monitor():
    """Module-level fragment — stable identity lets run_every fire reliably."""
    from core.background import background_tasks, stop_background_task, _lock as _bg_lock

    with _bg_lock:
        active_tasks = {
            tid: task
            for tid, task in background_tasks.items()
            if task.get("status") in ["processing", "failed"]
        }
    if not active_tasks:
        return

    st.divider()
    st.subheader("⏳ Background Processes")
    for tid, tinfo in active_tasks.items():
        st.markdown(f"**{tinfo.get('filename', 'Task')}**")

        if tinfo["status"] == "failed":
            st.error(f"Failed: {tinfo.get('error', 'Unknown Error')}")
        elif tinfo.get("is_web"):
            current = tinfo.get("pages_current", 0)
            total = tinfo.get("pages_total", 1)
            st.progress(
                min(max(current / total if total > 0 else 0, 0.0), 1.0),
                text=f"Topics: {current}/{total}",
            )
        elif tinfo.get("mode") == "GENERATION":
            cc = tinfo.get("current_chunk_index", 0)
            tc = tinfo.get("total_chunks", 0)
            fc = tinfo.get("flashcards_count", 0)
            percent = min(max(cc / tc if tc > 0 else 0, 0.0), 1.0)
            st.progress(
                percent,
                text=f"Chunk {cc}/{tc} · {fc} card(s) · {tinfo.get('status_message', 'Processing...')}",
            )
        else:
            tp = tinfo.get("total_pages", 1)
            cp = tinfo.get("current_page", 0)
            cc = tinfo.get("current_chunk_index", 0)
            tc = tinfo.get("chunks_in_page", 1)
            chunk_progress = cc / tc if tc > 0 else 0
            global_progress = (cp + chunk_progress) / tp if tp > 0 else 0
            st.progress(
                min(max(global_progress, 0.0), 1.0),
                text=f"Page {cp+1}/{tp}: {tinfo.get('status_message', 'Processing...')}",
            )

        if st.button("⏹️ Remove/Stop", key=f"stop_task_{tid}"):
            stop_background_task(tid)
            if tinfo["status"] == "failed":
                from core.background import background_tasks as bt
                bt.pop(tid, None)


def render_sidebar_background_monitor():
    """Sidebar fragment showing active background tasks.

    The run_every fragment is only mounted when there are active/failed tasks.
    When idle this renders nothing — no fragment, no WebSocket polling, no errors.
    """
    from core.background import background_tasks, _lock as _bg_lock
    with _bg_lock:
        has_active = any(
            t.get("status") in ("processing", "failed")
            for t in background_tasks.values()
        )
    if has_active:
        _sidebar_monitor()


@st.fragment(run_every=5)
def _study_materials_monitor():
    """Module-level fragment — stable identity lets run_every fire reliably."""
    from core.background import background_tasks, stop_background_task, clear_background_task, _lock as _bg_lock

    with _bg_lock:
        tasks_snapshot = dict(background_tasks)
    if not tasks_snapshot:
        return

    st.divider()
    st.subheader("⚙️ Active Background Tasks")
    for d_id, task in tasks_snapshot.items():
        is_web = task.get("is_web", False)
        display_name = task.get("display_name") or task.get("filename") or d_id[:8]

        with st.container(border=True):
            col_t, col_b = st.columns([0.85, 0.15])

            if task["status"] == "processing":
                if is_web:
                    curr = task.get("pages_current", 0)
                    total = task.get("pages_total", 1)
                    fc = task.get("flashcards_count", 0)
                    percent = min(max(curr / total if total > 0 else 0, 0.0), 1.0)
                    col_t.markdown(f"**🌐 Researching: {display_name}**")
                    col_t.progress(percent, text=f"Page {curr} / {total} · {fc} flashcard(s) generated")
                elif task.get("mode") == "GENERATION":
                    cc = task.get("current_chunk_index", 0)
                    tc = task.get("total_chunks", 0)
                    fc = task.get("flashcards_count", 0)
                    percent = min(max(cc / tc if tc > 0 else 0, 0.0), 1.0)
                    col_t.markdown(f"**🧠 Generating: {display_name}**")
                    col_t.progress(percent, text=f"Chunk {cc} / {tc} · {fc} card(s) · {task.get('status_message', 'Processing...')}")
                else:
                    tp = task.get("total_pages", 1)
                    cp = task.get("current_page", 0)
                    cc = task.get("current_chunk_index", 0)
                    tc = task.get("chunks_in_page", 1)
                    chunk_progress = cc / tc if tc > 0 else 0
                    percent = min(max((cp + chunk_progress) / tp if tp > 0 else 0, 0.0), 1.0)
                    col_t.markdown(f"**⏳ Indexing: {display_name}**")
                    col_t.progress(percent, text=f"Page {cp+1}/{tp}: {task.get('status_message', 'Processing...')}")

                if col_b.button("⏹️ Stop", key=f"stop_{d_id}"):
                    stop_background_task(d_id)

            elif task["status"] == "completed":
                if is_web:
                    icon = "🌐"
                    col_t.success(f"{icon} **Completed**: {display_name}")
                elif task.get("mode") == "GENERATION":
                    fc = task.get("flashcards_count", 0)
                    col_t.success(f"✅ **Completed**: {display_name} — {fc} card(s) generated")
                else:
                    col_t.success(f"✅ **Completed**: {display_name}")
                if col_b.button("✖ Clear", key=f"clear_{d_id}"):
                    clear_background_task(d_id)

            elif task["status"] == "stopped":
                col_t.warning(f"⏹️ **Stopped**: {display_name}")
                if col_b.button("✖ Clear", key=f"clear_{d_id}"):
                    clear_background_task(d_id)

            elif task["status"] == "failed":
                col_t.error(f"❌ **Failed**: {display_name} — {task.get('error', 'Unknown Error')}")
                if col_b.button("✖ Clear", key=f"clear_{d_id}"):
                    clear_background_task(d_id)


def render_study_materials_background_monitor():
    """Inline background task monitor shown at the bottom of Study Materials page.

    The run_every fragment is only mounted when tasks exist.
    When idle this renders nothing — no fragment, no WebSocket polling, no errors.
    """
    from core.background import background_tasks, _lock as _bg_lock
    with _bg_lock:
        has_any = bool(background_tasks)
    if has_any:
        _study_materials_monitor()
