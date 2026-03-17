"""
ui/components/background_monitor.py
--------------------------------------
Background task monitor fragment for the sidebar and study materials page.
Moved verbatim from app.py — zero behaviour change.
"""

import streamlit as st


def render_sidebar_background_monitor():
    """Sidebar fragment showing active background tasks."""
    from core.background import background_tasks, stop_background_task, _lock as _bg_lock

    @st.fragment
    def _monitor():
        with _bg_lock:
            active_tasks = {
                tid: task
                for tid, task in background_tasks.items()
                if task.get("status") in ["processing", "failed"]
            }
        if active_tasks:
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
                        del bt[tid]
                    st.rerun()

    _monitor()


def render_study_materials_background_monitor():
    """Inline background task monitor shown at the bottom of Study Materials page."""
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
                else:
                    tp = task.get("total_pages", 1)
                    cp = task.get("current_page", 0)
                    cc = task.get("current_chunk_index", 0)
                    tc = task.get("chunks_in_page", 1)
                    chunk_progress = cc / tc if tc > 0 else 0
                    percent = min(max((cp + chunk_progress) / tp if tp > 0 else 0, 0.0), 1.0)
                    col_t.markdown(f"**⏳ Processing: {display_name}**")
                    col_t.progress(percent, text=f"Page {cp+1}/{tp}: {task.get('status_message', 'Processing...')}")

                if col_b.button("⏹️ Stop", key=f"stop_{d_id}"):
                    stop_background_task(d_id)
                    st.rerun()

            elif task["status"] == "completed":
                icon = "🌐" if is_web else "✅"
                col_t.success(f"{icon} **Completed**: {display_name}")
                if col_b.button("✖ Clear", key=f"clear_{d_id}"):
                    clear_background_task(d_id)
                    st.rerun()

            elif task["status"] == "stopped":
                col_t.warning(f"⏹️ **Stopped**: {display_name}")
                if col_b.button("✖ Clear", key=f"clear_{d_id}"):
                    clear_background_task(d_id)
                    st.rerun()

            elif task["status"] == "failed":
                col_t.error(f"❌ **Failed**: {display_name} — {task.get('error', 'Unknown Error')}")
                if col_b.button("✖ Clear", key=f"clear_{d_id}"):
                    clear_background_task(d_id)
                    st.rerun()
