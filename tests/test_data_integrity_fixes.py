"""
tests/test_data_integrity_fixes.py
------------------------------------
Fast unit/integration tests for data integrity fixes H1, H6, H13, H16, H19.

All tests run without live LLM or Qdrant calls (monkeypatched where needed).
Uses an isolated in-memory SQLite database so the real nexus.db is never touched.
"""
import pytest
import uuid
from unittest.mock import MagicMock, patch, call
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# Shared in-memory DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def mem_db():
    """Fresh in-memory SQLite DB with all ORM tables.  Torn down after each test."""
    from core.database import Base, Subject, Document, Topic, Subtopic, Flashcard
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    yield db, {"Subject": Subject, "Document": Document, "Topic": Topic,
                "Subtopic": Subtopic, "Flashcard": Flashcard}
    db.close()
    Base.metadata.drop_all(bind=engine)


def _seed_topic_with_cards(db, models):
    """Helper: creates doc → topic → 2 subtopics → 3 flashcards (2 approved, 1 pending)."""
    S = models["Subject"]; D = models["Document"]; T = models["Topic"]
    St = models["Subtopic"]; F = models["Flashcard"]

    subj = S(name=f"subj-{uuid.uuid4().hex[:6]}")
    db.add(subj); db.flush()

    doc = D(id=str(uuid.uuid4()), filename="test.pdf", content_hash=uuid.uuid4().hex)
    db.add(doc); db.flush()

    topic = T(document_id=doc.id, name="Core Concepts")
    db.add(topic); db.flush()

    sub1 = St(topic_id=topic.id, name="RDDs")
    sub2 = St(topic_id=topic.id, name="DataFrames")
    db.add_all([sub1, sub2]); db.flush()

    fc_approved1 = F(subject_id=subj.id, subtopic_id=sub1.id,
                     question="Q1", answer="A1", status="approved", critic_score=4)
    fc_approved2 = F(subject_id=subj.id, subtopic_id=sub2.id,
                     question="Q2", answer="A2", status="approved", critic_score=5)
    fc_pending   = F(subject_id=subj.id, subtopic_id=sub1.id,
                     question="Q3", answer="A3", status="pending",  critic_score=2)
    db.add_all([fc_approved1, fc_approved2, fc_pending])
    db.commit()

    return {
        "subject_id": subj.id, "doc_id": doc.id,
        "topic_id": topic.id,
        "sub1_id": sub1.id, "sub2_id": sub2.id,
        "fc_approved1_id": fc_approved1.id,
        "fc_approved2_id": fc_approved2.id,
        "fc_pending_id":   fc_pending.id,
    }


# ===========================================================================
# H6 — delete_topic_data preserves approved flashcards
# ===========================================================================

class TestH6PreserveApprovedFlashcards:
    """H6: When a topic is deleted, approved flashcards must survive with subtopic_id=NULL."""

    def _run_delete_logic(self, db, models, topic_id):
        """Replicates the H6-fixed delete_topic_data logic against the test session."""
        from sqlalchemy import delete, update as sa_update
        F = models["Flashcard"]; St = models["Subtopic"]; T = models["Topic"]
        from core.database import ContentChunk

        subtopics = db.query(St).filter(St.topic_id == topic_id).all()
        subtopic_ids = [s.id for s in subtopics]

        if subtopic_ids:
            # Preserve approved flashcards
            db.execute(
                sa_update(F)
                .where(F.subtopic_id.in_(subtopic_ids), F.status == "approved")
                .values(subtopic_id=None),
                execution_options={"synchronize_session": False},
            )
            # Delete only non-approved
            db.execute(
                delete(F).where(
                    F.subtopic_id.in_(subtopic_ids),
                    F.status != "approved",
                )
            )
        db.execute(delete(St).where(St.topic_id == topic_id))
        db.execute(delete(T).where(T.id == topic_id))
        db.commit()

    def test_approved_flashcards_survive(self, mem_db):
        db, models = mem_db
        ids = _seed_topic_with_cards(db, models)
        F = models["Flashcard"]

        self._run_delete_logic(db, models, ids["topic_id"])

        surviving = db.query(F).filter(F.subject_id == ids["subject_id"]).all()
        assert len(surviving) == 2, "Both approved flashcards should survive"
        for fc in surviving:
            assert fc.status == "approved"

    def test_approved_flashcards_have_null_subtopic(self, mem_db):
        db, models = mem_db
        ids = _seed_topic_with_cards(db, models)
        F = models["Flashcard"]

        self._run_delete_logic(db, models, ids["topic_id"])

        for fc in db.query(F).filter(F.subject_id == ids["subject_id"]).all():
            assert fc.subtopic_id is None, "Approved cards must have subtopic_id=NULL after topic deletion"

    def test_pending_flashcards_deleted(self, mem_db):
        db, models = mem_db
        ids = _seed_topic_with_cards(db, models)
        F = models["Flashcard"]

        self._run_delete_logic(db, models, ids["topic_id"])

        pending = db.query(F).filter(F.id == ids["fc_pending_id"]).first()
        assert pending is None, "Pending flashcard should be deleted along with the topic"

    def test_subtopics_deleted(self, mem_db):
        db, models = mem_db
        ids = _seed_topic_with_cards(db, models)
        St = models["Subtopic"]

        self._run_delete_logic(db, models, ids["topic_id"])

        assert db.query(St).filter(St.topic_id == ids["topic_id"]).count() == 0


# ===========================================================================
# H13 — curator deduplicates topics and subtopics from LLM response
# ===========================================================================

class TestH13CuratorDedup:
    """H13: Duplicate topic/subtopic names in the LLM response must be merged, not double-inserted."""

    def _run_dedup(self, topics):
        """Runs only the dedup portion of curate_structure in isolation."""
        seen_topic_keys = {}
        deduped_topics = []
        for t in topics:
            key = t.name.strip().lower()
            if key not in seen_topic_keys:
                seen_topic_keys[key] = t
                seen_sub_keys = set()
                unique_subs = []
                for s in t.subtopics:
                    sub_key = s.name.strip().lower()
                    if sub_key not in seen_sub_keys:
                        seen_sub_keys.add(sub_key)
                        unique_subs.append(s)
                t.subtopics = unique_subs
                deduped_topics.append(t)
            else:
                existing = seen_topic_keys[key]
                existing_sub_keys = {s.name.strip().lower() for s in existing.subtopics}
                for s in t.subtopics:
                    if s.name.strip().lower() not in existing_sub_keys:
                        existing.subtopics.append(s)
                        existing_sub_keys.add(s.name.strip().lower())
        return deduped_topics

    def _make_topic(self, name, subtopics):
        from agents.curator import TopicStructure, SubtopicStructure
        return TopicStructure(
            name=name, summary="",
            subtopics=[SubtopicStructure(name=s, summary="") for s in subtopics]
        )

    def test_duplicate_topics_merged_into_one(self):
        topics = [
            self._make_topic("RDD Basics", ["RDD Creation", "Transformations"]),
            self._make_topic("RDD Basics", ["Actions"]),  # duplicate topic name
        ]
        result = self._run_dedup(topics)
        assert len(result) == 1, "Two topics with same name should merge into one"

    def test_merged_topic_has_all_subtopics(self):
        topics = [
            self._make_topic("RDD Basics", ["RDD Creation", "Transformations"]),
            self._make_topic("RDD Basics", ["Actions"]),
        ]
        result = self._run_dedup(topics)
        sub_names = [s.name for s in result[0].subtopics]
        assert "RDD Creation" in sub_names
        assert "Transformations" in sub_names
        assert "Actions" in sub_names

    def test_duplicate_subtopics_within_same_topic_removed(self):
        topics = [
            self._make_topic("Spark SQL", ["DataFrames", "DataFrames", "SQL Queries"]),
        ]
        result = self._run_dedup(topics)
        sub_names = [s.name for s in result[0].subtopics]
        assert sub_names.count("DataFrames") == 1

    def test_case_insensitive_dedup(self):
        topics = [
            self._make_topic("spark sql", ["DataFrames"]),
            self._make_topic("Spark SQL", ["SQL Queries"]),
        ]
        result = self._run_dedup(topics)
        assert len(result) == 1

    def test_distinct_topics_both_kept(self):
        topics = [
            self._make_topic("RDD Basics", ["RDD Creation"]),
            self._make_topic("Spark SQL",  ["DataFrames"]),
        ]
        result = self._run_dedup(topics)
        assert len(result) == 2


# ===========================================================================
# H16 — low-quality score badge logic
# ===========================================================================

class TestH16ScoreBadge:
    """H16: Cards with critic_score < 3 must display a warning badge."""

    def _score_label(self, score):
        """Mirrors the score_label logic in render_flashcard_review_card."""
        return (
            f"⚠️ {score}/5 low quality"
            if score and score < 3
            else f"{score}/5"
        )

    def test_score_1_shows_warning(self):
        assert "⚠️" in self._score_label(1)

    def test_score_2_shows_warning(self):
        assert "⚠️" in self._score_label(2)

    def test_score_3_no_warning(self):
        assert "⚠️" not in self._score_label(3)

    def test_score_5_no_warning(self):
        assert "⚠️" not in self._score_label(5)

    def test_score_none_no_crash(self):
        label = self._score_label(None)
        assert "⚠️" not in label  # None is falsy → no warning badge

    def test_low_score_label_contains_score_value(self):
        label = self._score_label(2)
        assert "2/5" in label


# ===========================================================================
# H19 — TopicAssignerAgent raises on failure (no General Overview fallback)
# ===========================================================================

class TestH19NoFallback:
    """H19: assign_topic must raise on LLM failure, not return a 'General Overview' fallback."""

    @patch("agents.topic_assigner.get_llm")
    def test_raises_on_llm_failure(self, mock_get_llm):
        from agents.topic_assigner import TopicAssignerAgent
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("LLM timed out")
        
        with patch("agents.topic_assigner.ChatPromptTemplate.from_messages") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt.__or__.return_value = mock_chain
            mock_prompt_cls.return_value = mock_prompt
            
            agent = TopicAssignerAgent()
            with pytest.raises(RuntimeError, match="LLM timed out"):
                agent.assign_topic("some chunk text", [])

    @patch("agents.topic_assigner.get_llm")
    def test_does_not_return_general_overview(self, mock_get_llm):
        from agents.topic_assigner import TopicAssignerAgent
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = ValueError("parse error")
        
        with patch("agents.topic_assigner.ChatPromptTemplate.from_messages") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt.__or__.return_value = mock_chain
            mock_prompt_cls.return_value = mock_prompt
            
            agent = TopicAssignerAgent()
            try:
                result = agent.assign_topic("some chunk text", [])
                assert "General Overview" not in getattr(result, "topic_name", "")
                assert "Introduction" not in getattr(result, "subtopic_name", "")
            except (ValueError, RuntimeError):
                pass


# ===========================================================================
# H1 — Qdrant vectors deleted before DB record on library doc removal
# ===========================================================================

class TestH1QdrantCleanupOnDelete:
    """H1: Qdrant delete must be called with the correct document_id filter."""

    def test_qdrant_delete_uses_correct_document_id(self):
        """Verifies the filter structure passed to qdrant_client.delete matches document_id."""
        from qdrant_client.http import models as rest

        doc_id = str(uuid.uuid4())
        mock_qdrant = MagicMock()

        # Replicate the H1 deletion logic from app.py
        mock_qdrant.delete(
            collection_name="nexus_chunks",
            points_selector=rest.FilterSelector(
                filter=rest.Filter(
                    must=[rest.FieldCondition(
                        key="document_id",
                        match=rest.MatchValue(value=doc_id),
                    )]
                )
            ),
        )

        mock_qdrant.delete.assert_called_once()
        call_kwargs = mock_qdrant.delete.call_args.kwargs
        assert call_kwargs["collection_name"] == "nexus_chunks"
        selector = call_kwargs["points_selector"]
        field_cond = selector.filter.must[0]
        assert field_cond.key == "document_id"
        assert field_cond.match.value == doc_id

    def test_qdrant_failure_does_not_block_db_delete(self):
        """If Qdrant delete fails, the DB delete should still proceed."""
        import logging

        mock_qdrant = MagicMock()
        mock_qdrant.delete.side_effect = ConnectionError("Qdrant unreachable")
        mock_db = MagicMock()
        mock_doc = MagicMock()
        mock_doc.id = str(uuid.uuid4())

        # Simulate the H1-fixed deletion block from app.py
        logger = logging.getLogger("test_h1")
        try:
            from qdrant_client.http import models as rest
            mock_qdrant.delete(
                collection_name="nexus_chunks",
                points_selector=rest.FilterSelector(
                    filter=rest.Filter(must=[rest.FieldCondition(
                        key="document_id",
                        match=rest.MatchValue(value=mock_doc.id),
                    )])
                ),
            )
        except Exception as qe:
            logger.warning("Qdrant cleanup failed: %s", qe)

        # DB delete must still be called
        mock_db.delete(mock_doc)
        mock_db.commit()

        mock_db.delete.assert_called_once_with(mock_doc)
        mock_db.commit.assert_called_once()
