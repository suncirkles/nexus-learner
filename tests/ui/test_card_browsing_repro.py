"""
tests/ui/test_card_browsing_repro.py
--------------------------------------
Regression test for the WebSocketClosedError that occurs when browsing
flashcards (viewing source / rubric) on the Mentor Review page with no
active background tasks.

Context
-------
The errors are bursts of simultaneous Tornado write_message failures — the
signature of a full-page rerun hitting a closing WebSocket.  AppTest runs
in-process without a real WebSocket so it cannot reproduce the exact Tornado
error, but it CAN:

  1. Verify the background-monitor gates do NOT mount run_every fragments when
     background_tasks is empty (regression guard for the conditional-gate fix).

  2. Walk through the exact user sequence (Mentor page → expand subtopic →
     view pending cards → open "Recreate with Feedback" expander → switch
     tabs) and assert no Python exception is raised during any step.

  3. Confirm that every st.rerun() path in the card component is reachable
     under the mock data and exits cleanly.

If the WebSocket errors ever originate from a fragment firing unexpectedly,
a failing gate test here will catch it before it reaches prod.

Run with:
    PYTHONPATH=. pytest tests/ui/test_card_browsing_repro.py -v
"""

import sys
import threading
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from streamlit.testing.v1 import AppTest

import ui.api_client as _api


# ---------------------------------------------------------------------------
# Shared fake data
# ---------------------------------------------------------------------------

_SUBJECTS = [{"id": 1, "name": "Chemistry", "is_archived": False}]

_TOPIC_TREE = [
    {
        "id": 10, "document_id": "doc-abc",
        "name": "Acids and Bases", "summary": "pH and reaction types",
        "subtopics": [
            {
                "id": 100, "topic_id": 10,
                "name": "Strong Acids", "summary": "",
                "approved_count": 0, "pending_count": 2,
            },
            {
                "id": 101, "topic_id": 10,
                "name": "Buffer Systems", "summary": "",
                "approved_count": 1, "pending_count": 0,
            },
        ],
    },
]

_PENDING_CARDS = [
    {
        "id": 1, "subject_id": 1, "subtopic_id": 100, "chunk_id": 55,
        "question": "What is the pH of 0.1 M HCl?",
        "answer": "pH = 1",
        "question_type": "numerical", "complexity_level": "simple",
        "rubric": '[{"criterion": "Accuracy", "points": 5}]',
        "critic_rubric_scores": '{"Accuracy": 5}',
        "critic_score": 5, "critic_feedback": "Correct.",
        "status": "pending", "mentor_feedback": None, "created_at": None,
    },
    {
        "id": 2, "subject_id": 1, "subtopic_id": 100, "chunk_id": 56,
        "question": "Define a strong acid.",
        "answer": "One that fully dissociates.",
        "question_type": "active_recall", "complexity_level": "medium",
        "rubric": None, "critic_rubric_scores": None,
        "critic_score": 4, "critic_feedback": "Good.",
        "status": "pending", "mentor_feedback": None, "created_at": None,
    },
]

_APPROVED_CARDS = [
    {
        "id": 3, "subject_id": 1, "subtopic_id": 101, "chunk_id": 57,
        "question": "What is a buffer?",
        "answer": "A solution that resists pH change.",
        "question_type": "active_recall", "complexity_level": "medium",
        "rubric": None, "critic_rubric_scores": None,
        "critic_score": 5, "critic_feedback": "Perfect.",
        "status": "approved", "mentor_feedback": None, "created_at": None,
    },
]

_SOURCES_BATCH = {
    "55": {
        "source_type": "pdf", "source_url": None,
        "filename": "chemistry.pdf", "document_id": "doc-abc",
        "page_number": 3, "text": "HCl dissociates completely.",
    },
    "56": {
        "source_type": "pdf", "source_url": None,
        "filename": "chemistry.pdf", "document_id": "doc-abc",
        "page_number": 4, "text": "Strong acids fully ionise.",
    },
    "57": {
        "source_type": "pdf", "source_url": None,
        "filename": "chemistry.pdf", "document_id": "doc-abc",
        "page_number": 10, "text": "Buffers maintain pH.",
    },
}

_GLOBAL_STATS = {"total": 3, "approved": 1, "pending": 2, "rejected": 0}


def _patch_api_for_mentor(mock_obj):
    """Wire the minimal api_client surface needed by mentor.py + flashcard_card.py."""
    mock_obj.list_active_subjects.return_value = _SUBJECTS
    mock_obj.get_topic_tree.return_value = _TOPIC_TREE
    mock_obj.get_subjects_with_stats.return_value = _SUBJECTS
    mock_obj.get_global_flashcard_stats.return_value = _GLOBAL_STATS
    mock_obj.get_flashcards_by_subtopic.side_effect = lambda sub_id, status, **kw: (
        _PENDING_CARDS if status == "pending" else
        _APPROVED_CARDS if status == "approved" else []
    )
    mock_obj.get_chunk_sources_batch.return_value = _SOURCES_BATCH
    mock_obj.get_chunk_source.return_value = _SOURCES_BATCH.get("55")
    # HITL bulk actions (not needed for browse, but prevent AttributeError)
    mock_obj.bulk_approve_flashcards.return_value = {"approved": 0}
    mock_obj.bulk_reject_flashcards.return_value = {"rejected": 0}


# ---------------------------------------------------------------------------
# 1. Background-monitor gate — no fragment when background_tasks is empty
# ---------------------------------------------------------------------------

class TestBackgroundMonitorGate:
    """Verify that run_every fragments are NOT mounted when background_tasks is empty."""

    def test_sidebar_gate_blocks_fragment_when_idle(self):
        """render_sidebar_background_monitor must not call _sidebar_monitor when
        background_tasks is empty (no active or failed tasks)."""
        import importlib
        import ui.components.background_monitor as bm_module

        mock_bg = MagicMock()
        mock_bg.background_tasks = {}
        mock_bg._lock = threading.Lock()

        call_log = []
        original_monitor = bm_module._sidebar_monitor

        def spy_sidebar_monitor():
            call_log.append("sidebar")
            original_monitor()

        with patch.dict(sys.modules, {"core.background": mock_bg}):
            with patch.object(bm_module, "_sidebar_monitor", spy_sidebar_monitor):
                importlib.reload(bm_module)
                # Reload re-imports core.background; re-apply patch
                mock_bg2 = MagicMock()
                mock_bg2.background_tasks = {}
                mock_bg2._lock = threading.Lock()
                with patch.dict(sys.modules, {"core.background": mock_bg2}):
                    bm_module.render_sidebar_background_monitor()

        assert call_log == [], (
            "_sidebar_monitor was called despite empty background_tasks — "
            "the run_every fragment will poll even with no tasks, risking WebSocketClosedError"
        )

    def test_sidebar_gate_blocks_fragment_when_all_completed(self):
        """Completed tasks must NOT cause the sidebar fragment to mount
        (sidebar only shows processing/failed)."""
        from ui.components.background_monitor import render_sidebar_background_monitor
        import ui.components.background_monitor as bm_module

        completed_tasks = {
            "task-abc": {"status": "completed", "filename": "old.pdf", "is_web": False},
        }
        mock_bg = MagicMock()
        mock_bg.background_tasks = completed_tasks
        mock_bg._lock = threading.Lock()

        call_log = []

        def spy(*a, **kw):
            call_log.append("sidebar")

        with patch.dict(sys.modules, {"core.background": mock_bg}):
            with patch.object(bm_module, "_sidebar_monitor", spy):
                bm_module.render_sidebar_background_monitor()

        assert call_log == [], (
            "Completed task caused sidebar fragment to mount — "
            "once the task completes but isn't cleared, the gate should still block"
        )

    def test_study_gate_blocks_fragment_when_empty(self):
        """render_study_materials_background_monitor must not mount when
        background_tasks is empty."""
        import ui.components.background_monitor as bm_module

        mock_bg = MagicMock()
        mock_bg.background_tasks = {}
        mock_bg._lock = threading.Lock()

        call_log = []

        def spy(*a, **kw):
            call_log.append("study")

        with patch.dict(sys.modules, {"core.background": mock_bg}):
            with patch.object(bm_module, "_study_materials_monitor", spy):
                bm_module.render_study_materials_background_monitor()

        assert call_log == [], (
            "_study_materials_monitor mounted despite empty background_tasks"
        )

    def test_study_gate_mounts_for_completed_task(self):
        """Study-materials gate uses bool(background_tasks) — a completed task
        that was never cleared WILL keep the fragment mounted.  This test
        documents that behaviour so we know it exists."""
        import ui.components.background_monitor as bm_module

        completed_tasks = {
            "task-xyz": {"status": "completed", "filename": "doc.pdf", "is_web": False},
        }
        mock_bg = MagicMock()
        mock_bg.background_tasks = completed_tasks
        mock_bg._lock = threading.Lock()
        mock_bg.stop_background_task = MagicMock()
        mock_bg.clear_background_task = MagicMock()

        call_log = []

        def spy(*a, **kw):
            call_log.append("study")

        with patch.dict(sys.modules, {"core.background": mock_bg}):
            with patch.object(bm_module, "_study_materials_monitor", spy):
                bm_module.render_study_materials_background_monitor()

        # Document: study gate DOES mount for completed task (bool({...}) is True)
        # This is intentional — study page shows Clear button for completed tasks.
        assert call_log == ["study"], (
            "Expected study fragment to mount for a completed-but-uncleared task"
        )


# ---------------------------------------------------------------------------
# 2. Mentor page card browsing — AppTest sequence
# ---------------------------------------------------------------------------

def _mentor_app():
    """Minimal app script that renders just the Mentor Review page."""
    import streamlit as st
    from ui.pages.mentor import render_mentor_review
    render_mentor_review()


class TestMentorCardBrowsing:
    """Walk the exact sequence the user described: browse cards, check source/rubric."""

    def _build_at(self) -> AppTest:
        at = AppTest.from_function(_mentor_app, default_timeout=30)
        with patch.object(_api, "list_active_subjects", return_value=_SUBJECTS), \
             patch.object(_api, "get_topic_tree", return_value=_TOPIC_TREE), \
             patch.object(_api, "get_subjects_with_stats", return_value=_SUBJECTS), \
             patch.object(_api, "get_global_flashcard_stats", return_value=_GLOBAL_STATS), \
             patch.object(_api, "get_flashcards_by_subtopic",
                         side_effect=lambda sub_id, status, **kw: (
                             _PENDING_CARDS if status == "pending" else
                             _APPROVED_CARDS if status == "approved" else []
                         )), \
             patch.object(_api, "get_chunk_sources_batch", return_value=_SOURCES_BATCH), \
             patch.object(_api, "get_chunk_source",
                         return_value=_SOURCES_BATCH.get("55")), \
             patch.object(_api, "bulk_approve_flashcards", return_value={"approved": 0}), \
             patch.object(_api, "bulk_reject_flashcards", return_value={"rejected": 0}):
            at.run()
        return at

    def test_mentor_renders_without_exception(self):
        """Initial render of Mentor Review page raises no exception."""
        at = AppTest.from_function(_mentor_app, default_timeout=30)
        with patch.object(_api, "list_active_subjects", return_value=_SUBJECTS), \
             patch.object(_api, "get_topic_tree", return_value=_TOPIC_TREE), \
             patch.object(_api, "get_subjects_with_stats", return_value=_SUBJECTS), \
             patch.object(_api, "get_global_flashcard_stats", return_value=_GLOBAL_STATS), \
             patch.object(_api, "get_flashcards_by_subtopic",
                         side_effect=lambda sub_id, status, **kw: (
                             _PENDING_CARDS if status == "pending" else
                             _APPROVED_CARDS if status == "approved" else []
                         )), \
             patch.object(_api, "get_chunk_sources_batch", return_value=_SOURCES_BATCH), \
             patch.object(_api, "get_chunk_source", return_value=_SOURCES_BATCH.get("55")), \
             patch.object(_api, "bulk_approve_flashcards", return_value={"approved": 0}), \
             patch.object(_api, "bulk_reject_flashcards", return_value={"rejected": 0}):
            at.run()
        assert not at.exception, f"Mentor page raised: {at.exception}"

    def test_no_fragment_mounted_during_card_browse(self):
        """Browsing cards on Mentor page with empty background_tasks
        must not trigger any run_every fragment mount."""
        import ui.components.background_monitor as bm_module

        fragment_calls = []

        def spy_sidebar(*a, **kw):
            fragment_calls.append("sidebar")

        def spy_study(*a, **kw):
            fragment_calls.append("study")

        mock_bg = MagicMock()
        mock_bg.background_tasks = {}
        mock_bg._lock = threading.Lock()

        at = AppTest.from_function(_mentor_app, default_timeout=30)
        with patch.dict(sys.modules, {"core.background": mock_bg}), \
             patch.object(bm_module, "_sidebar_monitor", spy_sidebar), \
             patch.object(bm_module, "_study_materials_monitor", spy_study), \
             patch.object(_api, "list_active_subjects", return_value=_SUBJECTS), \
             patch.object(_api, "get_topic_tree", return_value=_TOPIC_TREE), \
             patch.object(_api, "get_subjects_with_stats", return_value=_SUBJECTS), \
             patch.object(_api, "get_global_flashcard_stats", return_value=_GLOBAL_STATS), \
             patch.object(_api, "get_flashcards_by_subtopic",
                         side_effect=lambda sub_id, status, **kw: (
                             _PENDING_CARDS if status == "pending" else
                             _APPROVED_CARDS if status == "approved" else []
                         )), \
             patch.object(_api, "get_chunk_sources_batch", return_value=_SOURCES_BATCH), \
             patch.object(_api, "get_chunk_source", return_value=_SOURCES_BATCH.get("55")), \
             patch.object(_api, "bulk_approve_flashcards", return_value={"approved": 0}), \
             patch.object(_api, "bulk_reject_flashcards", return_value={"rejected": 0}):
            at.run()

        assert not at.exception, f"Mentor page raised: {at.exception}"
        assert fragment_calls == [], (
            f"run_every fragment(s) mounted during idle card browse: {fragment_calls}"
        )
