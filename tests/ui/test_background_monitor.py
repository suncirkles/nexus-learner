"""
tests/ui/test_background_monitor.py
--------------------------------------
UI regression tests for the background task monitor components.

Covers two categories:

1. STRUCTURAL — pure Python, no Streamlit runtime needed.
   Regression: fragments defined as *nested* functions get a new function object
   on every Streamlit rerun, so Streamlit cannot maintain their identity and
   `run_every` never fires.  The fix: fragments must be module-level.

2. RENDER (AppTest) — headless Streamlit tests that inject fake task state and
   assert the correct widgets are rendered (progress bars, success/error messages).

Run with:
    PYTHONPATH=. pytest tests/ui/test_background_monitor.py -v
"""

import sys
import threading
import importlib
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf_task(status="processing", current_page=2, total_pages=5,
                   current_chunk_index=3, chunks_in_page=10):
    return {
        "status": status,
        "filename": "test_doc.pdf",
        "is_web": False,
        "total_pages": total_pages,
        "current_page": current_page,
        "current_chunk_index": current_chunk_index,
        "chunks_in_page": chunks_in_page,
        "status_message": "Processing chunk...",
    }


def _make_web_task(status="processing", pages_current=2, pages_total=5, flashcards_count=4):
    return {
        "status": status,
        "display_name": "Web Research — PySpark",
        "filename": "Web Research — PySpark",
        "is_web": True,
        "pages_current": pages_current,
        "pages_total": pages_total,
        "flashcards_count": flashcards_count,
    }


def _make_mock_bg(tasks: dict):
    """Return a fake core.background module with the given tasks dict."""
    mock = MagicMock()
    mock.background_tasks = tasks
    mock._lock = threading.Lock()
    mock.stop_background_task = MagicMock()
    mock.clear_background_task = MagicMock()
    return mock


def _run_monitor_fragment(component: str, tasks: dict):
    """
    Inject a fake core.background into sys.modules (shared with AppTest since it
    runs in-process), then run a self-contained app() via AppTest.from_function.

    The app() functions must NOT reference any closure variables — AppTest
    serialises the function source to a temp file, so only names that are
    imported *inside* the function body are available.

    Calling bm.render_*() through the module reference means Python looks up
    _sidebar_monitor / _study_materials_monitor in background_monitor's own
    globals (where they ARE defined), avoiding the NameError.
    """
    from streamlit.testing.v1 import AppTest

    # Set mock before at.run() — sys.modules is shared in-process
    sys.modules["core.background"] = _make_mock_bg(tasks)

    if component == "sidebar":
        def app():
            from ui.components import background_monitor as bm
            bm.render_sidebar_background_monitor()
    else:
        def app():
            from ui.components import background_monitor as bm
            bm.render_study_materials_background_monitor()

    at = AppTest.from_function(app)
    at.run()
    return at


# ---------------------------------------------------------------------------
# 1. Structural tests
# ---------------------------------------------------------------------------

class TestFragmentIsModuleLevel:
    """
    Regression guard: if either monitor is moved back inside a wrapper function,
    Streamlit loses fragment identity on rerun and run_every stops firing.
    """

    def test_sidebar_monitor_exists_at_module_level(self):
        import ui.components.background_monitor as m
        assert hasattr(m, "_sidebar_monitor"), (
            "_sidebar_monitor must be a module-level name. "
            "Nested @st.fragment functions get a new object on every rerun — "
            "Streamlit cannot track them and run_every never fires."
        )

    def test_study_materials_monitor_exists_at_module_level(self):
        import ui.components.background_monitor as m
        assert hasattr(m, "_study_materials_monitor"), (
            "_study_materials_monitor must be module-level for run_every to work."
        )

    def test_sidebar_monitor_is_callable(self):
        import ui.components.background_monitor as m
        assert callable(m._sidebar_monitor)

    def test_study_materials_monitor_is_callable(self):
        import ui.components.background_monitor as m
        assert callable(m._study_materials_monitor)

    def test_render_sidebar_background_monitor_is_callable(self):
        import ui.components.background_monitor as m
        assert callable(m.render_sidebar_background_monitor)

    def test_render_study_materials_background_monitor_is_callable(self):
        import ui.components.background_monitor as m
        assert callable(m.render_study_materials_background_monitor)


# ---------------------------------------------------------------------------
# 2. AppTest render tests — sidebar monitor
# ---------------------------------------------------------------------------

class TestSidebarMonitorRender:
    # Note: st.progress renders as UnknownElement in AppTest (not queryable by value).
    # We test via button presence (each active task gets a Stop/Remove button)
    # and markdown content.

    def test_no_tasks_renders_no_buttons(self):
        # Returns early when active_tasks is empty — nothing rendered
        at = _run_monitor_fragment("sidebar", {})
        assert not at.exception
        assert len(at.button) == 0

    def test_pdf_task_renders_stop_button(self):
        at = _run_monitor_fragment("sidebar", {"t1": _make_pdf_task()})
        assert not at.exception
        assert len(at.button) == 1
        assert "Stop" in at.button[0].label

    def test_pdf_task_renders_filename_in_markdown(self):
        at = _run_monitor_fragment("sidebar", {"t1": _make_pdf_task()})
        assert not at.exception
        assert any("test_doc.pdf" in m.value for m in at.markdown)

    def test_web_task_renders_stop_button(self):
        at = _run_monitor_fragment("sidebar", {"t1": _make_web_task()})
        assert not at.exception
        assert len(at.button) == 1

    def test_web_task_renders_display_name_in_markdown(self):
        at = _run_monitor_fragment("sidebar", {"t1": _make_web_task()})
        assert not at.exception
        assert any("Web Research" in m.value for m in at.markdown)

    def test_failed_task_renders_error(self):
        task = {"status": "failed", "is_web": False,
                "filename": "bad.pdf", "error": "LLM timeout"}
        at = _run_monitor_fragment("sidebar", {"t1": task})
        assert not at.exception
        assert len(at.error) >= 1
        assert any("LLM timeout" in e.value for e in at.error)

    def test_multiple_tasks_render_multiple_stop_buttons(self):
        at = _run_monitor_fragment(
            "sidebar", {"t1": _make_pdf_task(), "t2": _make_web_task()}
        )
        assert not at.exception
        assert len(at.button) == 2


# ---------------------------------------------------------------------------
# 3. AppTest render tests — study materials monitor
# ---------------------------------------------------------------------------

class TestStudyMaterialsMonitorRender:
    # st.progress is an UnknownElement in AppTest — we assert via buttons and
    # status messages which are the user-visible regression signals.

    def test_no_tasks_renders_no_buttons(self):
        at = _run_monitor_fragment("study_materials", {})
        assert not at.exception
        assert len(at.button) == 0

    def test_processing_pdf_task_renders_stop_button(self):
        at = _run_monitor_fragment("study_materials", {"t1": _make_pdf_task()})
        assert not at.exception
        assert len(at.button) == 1
        assert "Stop" in at.button[0].label

    def test_processing_pdf_task_renders_filename_in_markdown(self):
        at = _run_monitor_fragment("study_materials", {"t1": _make_pdf_task()})
        assert not at.exception
        assert any("test_doc.pdf" in m.value for m in at.markdown)

    def test_completed_task_renders_success_message(self):
        task = {"status": "completed", "filename": "spark.pdf",
                "is_web": False, "display_name": "PySpark Notes"}
        at = _run_monitor_fragment("study_materials", {"t1": task})
        assert not at.exception
        assert len(at.success) >= 1
        assert any("Completed" in s.value for s in at.success)

    def test_stopped_task_renders_warning(self):
        task = {"status": "stopped", "filename": "spark.pdf",
                "is_web": False, "display_name": "PySpark Notes"}
        at = _run_monitor_fragment("study_materials", {"t1": task})
        assert not at.exception
        assert len(at.warning) >= 1

    def test_failed_task_renders_error(self):
        task = {"status": "failed", "filename": "spark.pdf",
                "is_web": False, "display_name": "PySpark Notes",
                "error": "Connection refused"}
        at = _run_monitor_fragment("study_materials", {"t1": task})
        assert not at.exception
        assert len(at.error) >= 1
        assert any("Connection refused" in e.value for e in at.error)

    def test_web_task_renders_stop_button(self):
        at = _run_monitor_fragment("study_materials", {"t1": _make_web_task()})
        assert not at.exception
        assert len(at.button) == 1

    def test_web_task_renders_display_name_in_markdown(self):
        at = _run_monitor_fragment("study_materials", {"t1": _make_web_task()})
        assert not at.exception
        assert any("Web Research" in m.value for m in at.markdown)
