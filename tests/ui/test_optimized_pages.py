"""
tests/ui/test_optimized_pages.py
----------------------------------
UI integration tests for the chatty-API-optimised pages (feat/optimize-chatty-apis).

Strategy: AppTest.from_function runs in-process, so patch.object on the already-loaded
ui.api_client module is visible to all page code without any sys.modules trickery.

Coverage:
  1. Dashboard — renders using get_subjects_with_stats() (1 call, not 1+2N)
  2. Mentor    — renders subject selectbox + topic tree from get_topic_tree()
  3. Flashcard list — batches chunk sources via get_chunk_sources_batch()
                      instead of one-per-card get_chunk_source() calls

Run with:
    PYTHONPATH=. pytest tests/ui/test_optimized_pages.py -v
"""

import sys
from unittest.mock import MagicMock, call, patch

import pytest
from streamlit.testing.v1 import AppTest

import ui.api_client as _api  # imported once; patch.object modifies in-place


# ---------------------------------------------------------------------------
# Shared fake data
# ---------------------------------------------------------------------------

_SUBJECTS_WITH_STATS = [
    {
        "id": 1, "name": "Physics", "is_archived": False,
        "topic_count": 3, "approved": 5, "pending": 2, "rejected": 0,
    },
    {
        "id": 2, "name": "Chemistry", "is_archived": False,
        "topic_count": 1, "approved": 2, "pending": 1, "rejected": 0,
    },
]

_GLOBAL_STATS = {"total": 10, "approved": 7, "pending": 3, "rejected": 0}

_ACTIVE_SUBJECTS = [
    {"id": 1, "name": "Physics", "is_archived": False},
    {"id": 2, "name": "Chemistry", "is_archived": False},
]

_TOPIC_TREE = [
    {
        "id": 10, "document_id": "doc-a",
        "name": "Mechanics", "summary": "Force and motion",
        "subtopics": [
            {
                "id": 100, "topic_id": 10,
                "name": "Newton's Laws", "summary": "",
                "approved_count": 2, "pending_count": 1,
            },
        ],
    },
]

_FLASHCARDS_PENDING = [
    {
        "id": 1, "subject_id": 1, "subtopic_id": 100, "chunk_id": 55,
        "question": "State Newton's 2nd law.", "answer": "F = ma",
        "question_type": "active_recall", "complexity_level": "medium",
        "rubric": None, "critic_rubric_scores": None,
        "critic_score": 4, "critic_feedback": "Well grounded.",
        "status": "pending", "mentor_feedback": None, "created_at": None,
    },
]

_SOURCES_BATCH = {
    "55": {
        "source_type": "pdf", "source_url": None,
        "filename": "physics.pdf", "document_id": "doc-a",
        "page_number": 1, "text": "Newton stated that F = ma.",
    }
}


# ---------------------------------------------------------------------------
# 1. Dashboard — 1 batch call instead of 1 + 2N
# ---------------------------------------------------------------------------

class TestDashboardBatchLoad:
    def test_dashboard_renders_subject_names_from_with_stats(self):
        """Dashboard shows subject tiles using get_subjects_with_stats (1 API call)."""
        from ui.pages.dashboard import _get_dashboard_data
        _get_dashboard_data.clear()  # evict any cached result from a prior test run

        with (
            patch.object(_api, "get_subjects_with_stats", return_value=_SUBJECTS_WITH_STATS) as mock_stats,
            patch.object(_api, "get_global_stats", return_value=_GLOBAL_STATS),
        ):
            def app():
                from ui.pages.dashboard import render_dashboard
                render_dashboard()

            at = AppTest.from_function(app)
            at.run()

        assert not at.exception, f"Dashboard raised: {at.exception}"

        # Both subject names should appear in rendered markdown
        all_md = " ".join(m.value for m in at.markdown)
        assert "Physics" in all_md
        assert "Chemistry" in all_md

        # The new batch endpoint was called exactly once
        mock_stats.assert_called_once()

    def test_dashboard_does_not_call_legacy_per_subject_stats(self):
        """get_flashcard_stats (per-subject) must NOT be called — it's replaced by the batch call."""
        from ui.pages.dashboard import _get_dashboard_data
        _get_dashboard_data.clear()

        with (
            patch.object(_api, "get_subjects_with_stats", return_value=_SUBJECTS_WITH_STATS),
            patch.object(_api, "get_global_stats", return_value=_GLOBAL_STATS),
            patch.object(_api, "get_flashcard_stats") as legacy,
        ):
            def app():
                from ui.pages.dashboard import render_dashboard
                render_dashboard()

            AppTest.from_function(app).run()

        legacy.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Mentor — lazy subject selectbox + topic tree (1 call, not 1+S+S×T)
# ---------------------------------------------------------------------------

class TestMentorLazyLoad:
    def test_mentor_renders_subject_selectbox(self):
        """Mentor page shows a subject selectbox (lazy selection replaces pre-load loop)."""
        with (
            patch.object(_api, "list_active_subjects", return_value=_ACTIVE_SUBJECTS),
            patch.object(_api, "get_topic_tree", return_value=_TOPIC_TREE),
            patch.object(_api, "get_all_rejected_flashcards", return_value=[]),
            patch.object(_api, "get_flashcards_by_subtopic", return_value=[]),
        ):
            def app():
                from ui.pages.mentor import render_mentor_review
                render_mentor_review()

            at = AppTest.from_function(app)
            at.run()

        assert not at.exception, f"Mentor raised: {at.exception}"
        # Subject selectbox must be present
        assert len(at.selectbox) >= 1
        assert at.selectbox[0].options == ["Physics", "Chemistry"]

    def test_review_bin_uses_batch_source_call(self):
        """Review Bin makes 1 get_chunk_sources_batch call, not 1 get_chunk_source per card."""
        rejected_fcs = [
            {
                "id": 9, "subject_id": 1, "subtopic_id": 100, "chunk_id": 55,
                "question": "Q?", "answer": "A.", "question_type": "active_recall",
                "complexity_level": None, "rubric": None, "critic_rubric_scores": None,
                "critic_score": 2, "critic_feedback": "Needs work.", "status": "rejected",
                "mentor_feedback": None, "created_at": None,
            },
            {
                "id": 10, "subject_id": 1, "subtopic_id": 100, "chunk_id": 56,
                "question": "Q2?", "answer": "A2.", "question_type": "short_answer",
                "complexity_level": None, "rubric": None, "critic_rubric_scores": None,
                "critic_score": 1, "critic_feedback": "Poor.", "status": "rejected",
                "mentor_feedback": None, "created_at": None,
            },
        ]
        with (
            patch.object(_api, "list_active_subjects", return_value=_ACTIVE_SUBJECTS),
            patch.object(_api, "get_topic_tree", return_value=_TOPIC_TREE),
            patch.object(_api, "get_flashcards_by_subtopic", return_value=[]),
            patch.object(_api, "get_all_rejected_flashcards", return_value=rejected_fcs),
            patch.object(_api, "get_chunk_sources_batch", return_value={
                "55": {"source_type": "pdf", "source_url": None, "filename": "physics.pdf",
                       "document_id": "doc-a", "page_number": None, "text": None},
                "56": {"source_type": "pdf", "source_url": None, "filename": "physics.pdf",
                       "document_id": "doc-a", "page_number": None, "text": None},
            }) as mock_batch,
            patch.object(_api, "get_chunk_source") as mock_single,
        ):
            def app():
                from ui.pages.mentor import render_mentor_review
                render_mentor_review()

            at = AppTest.from_function(app)
            at.run()

        assert not at.exception, f"Mentor review bin raised: {at.exception}"
        # Batch call made with both chunk IDs
        mock_batch.assert_called_once_with([55, 56])
        # Per-card individual call must NOT fire
        mock_single.assert_not_called()

    def test_mentor_calls_tree_not_individual_subtopic_fetches(self):
        """get_topic_tree (1 call) replaces get_topics_by_subject + N×get_subtopics_by_topic."""
        with (
            patch.object(_api, "list_active_subjects", return_value=_ACTIVE_SUBJECTS),
            patch.object(_api, "get_topic_tree", return_value=_TOPIC_TREE) as mock_tree,
            patch.object(_api, "get_all_rejected_flashcards", return_value=[]),
            patch.object(_api, "get_flashcards_by_subtopic", return_value=[]),
            patch.object(_api, "get_subtopics_by_topic") as legacy_sub,
            patch.object(_api, "get_topics_by_subject") as legacy_topics,
        ):
            def app():
                from ui.pages.mentor import render_mentor_review
                render_mentor_review()

            AppTest.from_function(app).run()

        # New tree endpoint called; legacy per-topic subtopic fetch must NOT be called
        mock_tree.assert_called_once_with(1)  # subject_id=1 (first subject, default selection)
        legacy_sub.assert_not_called()
        legacy_topics.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Flashcard list — batch chunk sources (1 call, not 1-per-card)
# ---------------------------------------------------------------------------

class TestFlashcardListBatchSources:
    def test_render_flashcard_list_uses_batch_source_call(self):
        """render_flashcard_list makes 1 get_chunk_sources_batch call, not N get_chunk_source calls."""
        with (
            patch.object(_api, "get_flashcards_by_subtopic", return_value=_FLASHCARDS_PENDING) as mock_fc,
            patch.object(_api, "get_chunk_sources_batch", return_value=_SOURCES_BATCH) as mock_batch,
            patch.object(_api, "get_chunk_source") as mock_single,
        ):
            def app():
                from ui.components.flashcard_card import render_flashcard_list
                render_flashcard_list(100, "pending")

            at = AppTest.from_function(app)
            at.run()

        assert not at.exception, f"Flashcard list raised: {at.exception}"

        # Batch call made with the correct chunk_ids
        mock_batch.assert_called_once_with([55])

        # Per-card individual source call must NOT be made
        mock_single.assert_not_called()

    def test_render_flashcard_list_shows_pagination_buttons(self):
        """Prev/Next pagination buttons are rendered when a full page of cards is returned."""
        # Return exactly _PAGE_SIZE cards so "Next →" is shown
        from ui.components.flashcard_card import _PAGE_SIZE
        full_page = [
            {
                "id": i, "subject_id": 1, "subtopic_id": 100, "chunk_id": None,
                "question": f"Q{i}", "answer": f"A{i}",
                "question_type": "active_recall", "complexity_level": None,
                "rubric": None, "critic_rubric_scores": None, "critic_score": 3,
                "critic_feedback": "", "status": "pending",
                "mentor_feedback": None, "created_at": None,
            }
            for i in range(_PAGE_SIZE)
        ]

        with (
            patch.object(_api, "get_flashcards_by_subtopic", return_value=full_page),
            patch.object(_api, "get_chunk_sources_batch", return_value={}),
        ):
            def app():
                from ui.components.flashcard_card import render_flashcard_list
                render_flashcard_list(100, "pending")

            at = AppTest.from_function(app)
            at.run()

        assert not at.exception, f"Flashcard list raised: {at.exception}"

        button_labels = [b.label for b in at.button]
        assert any("Next" in lbl for lbl in button_labels), (
            f"Expected a 'Next →' pagination button but got: {button_labels}"
        )

    def test_render_flashcard_list_passes_skip_and_limit(self):
        """get_flashcards_by_subtopic is called with skip=0 and limit=PAGE_SIZE (not unbounded)."""
        from ui.components.flashcard_card import _PAGE_SIZE

        with (
            patch.object(_api, "get_flashcards_by_subtopic", return_value=[]) as mock_fc,
            patch.object(_api, "get_chunk_sources_batch", return_value={}),
        ):
            def app():
                from ui.components.flashcard_card import render_flashcard_list
                render_flashcard_list(100, "approved")

            AppTest.from_function(app).run()

        mock_fc.assert_called_once_with(100, "approved", skip=0, limit=_PAGE_SIZE)
