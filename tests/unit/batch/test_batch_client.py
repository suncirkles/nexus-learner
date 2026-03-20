"""
tests/unit/batch/test_batch_client.py
--------------------------------------
Unit tests for core/batch_client.py — BatchClient with mocked Anthropic SDK
and an in-memory SQLite database.

Chunk content is sourced from:
    D:\\cse\\Probability\\descriptive statistics.pdf

If that file is not present the tests fall back to synthetic statistics text
and still exercise every code path.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database import (
    Base,
    BatchJob,
    BatchRequest,
    ContentChunk,
    Document as DBDocument,
    Flashcard,
    Subject,
    SubjectDocumentAssociation,
    Subtopic,
    Topic,
)

# ---------------------------------------------------------------------------
# Real PDF path
# ---------------------------------------------------------------------------

_PDF = Path(r"D:\cse\Probability\descriptive statistics.pdf")


# ---------------------------------------------------------------------------
# DB fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture
def sf(engine):
    """Per-test session factory bound to the in-memory engine."""
    return sessionmaker(bind=engine, autobegin=True)


@pytest.fixture
def db(sf):
    with sf() as s:
        yield s


# ---------------------------------------------------------------------------
# Domain fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def subject(db):
    subj = Subject(name=f"Probability_{uuid.uuid4().hex[:6]}")
    db.add(subj)
    db.commit()
    db.refresh(subj)
    return subj


@pytest.fixture
def doc(db):
    d = DBDocument(
        id=str(uuid.uuid4()),
        filename="descriptive statistics.pdf",
        content_hash=uuid.uuid4().hex,
        source_type="pdf",
    )
    db.add(d)
    db.commit()
    db.refresh(d)
    return d


@pytest.fixture
def topic(db, doc):
    t = Topic(
        document_id=doc.id,
        name="Descriptive Statistics",
        summary="Measures of central tendency and dispersion",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def subtopic(db, topic):
    s = Subtopic(
        topic_id=topic.id,
        name="Measures of Central Tendency",
        summary="Mean, median, mode definitions and examples",
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


# ---------------------------------------------------------------------------
# PDF text extraction (real PDF preferred; synthetic fallback)
# ---------------------------------------------------------------------------

def _extract_pdf_pages(max_pages: int = 3) -> List[str]:
    """Extract text from the real PDF, or return synthetic statistics content."""
    if _PDF.exists():
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(_PDF))
            pages = []
            for i in range(min(max_pages, len(doc))):
                text = doc[i].get_text().strip()
                if text:
                    pages.append(text[:3000])
            doc.close()
            if pages:
                return pages
        except Exception:
            pass

    # Synthetic fallback — representative descriptive statistics content
    return [
        (
            "Measures of Central Tendency\n\n"
            "The mean is the arithmetic average of all data points. "
            "For a dataset {2, 4, 6, 8, 10}: mean = (2+4+6+8+10)/5 = 6.\n\n"
            "The median is the middle value when sorted. "
            "For an odd-count dataset, median = value at position (n+1)/2.\n\n"
            "The mode is the most frequently occurring value."
        ),
        (
            "Measures of Dispersion\n\n"
            "Variance: σ² = Σ(xi − μ)² / N.\n"
            "Standard deviation σ = √(Variance).\n"
            "Range = max − min.\n"
            "Interquartile range (IQR) = Q3 − Q1."
        ),
        (
            "Frequency Distributions\n\n"
            "A frequency distribution groups data into class intervals. "
            "Relative frequency = frequency / total observations.\n\n"
            "Cumulative frequency is the running total of frequencies up to and "
            "including each class."
        ),
    ]


@pytest.fixture
def pdf_chunks(db, doc, subtopic) -> List[ContentChunk]:
    """Insert real (or synthetic) PDF page text as ContentChunk rows."""
    texts = _extract_pdf_pages()
    chunks = []
    for i, text in enumerate(texts):
        c = ContentChunk(
            document_id=doc.id,
            text=text,
            subtopic_id=subtopic.id,
            page_number=i,
            source_type="pdf",
        )
        db.add(c)
    db.commit()
    for c in db.query(ContentChunk).filter(ContentChunk.document_id == doc.id).all():
        chunks.append(c)
    return chunks


# ---------------------------------------------------------------------------
# BatchClient factory
# ---------------------------------------------------------------------------

def _make_client(sf, mock_anthropic=None, mock_critic=None):
    """Return a BatchClient with all external I/O replaced by mocks."""
    from core.batch_client import BatchClient
    from repositories.sql.flashcard_repo import FlashcardRepo

    anthropic_mock = mock_anthropic or MagicMock()
    with (
        patch("core.batch_client.SessionLocal", sf),
        patch("repositories.sql.flashcard_repo.SessionLocal", sf),
    ):
        client = BatchClient(_client=anthropic_mock)
        client._fc_repo = FlashcardRepo()
        if mock_critic is not None:
            client._critic = mock_critic
        else:
            client._critic = MagicMock()
            client._critic.evaluate_flashcard.return_value = MagicMock(
                aggregate_score=4,
                rubric_scores_json='{"accuracy":4,"logic":4,"grounding":4,"clarity":4}',
                feedback="Well grounded.",
                suggested_complexity="medium",
                should_reject=False,
                reject_reason="",
                error=None,
            )
    return client, anthropic_mock


# ---------------------------------------------------------------------------
# Tests: build_requests
# ---------------------------------------------------------------------------

class TestBuildRequests:
    def test_creates_one_request_per_chunk_per_qtype(self, sf, doc, subject, pdf_chunks):
        n_chunks = len(pdf_chunks)
        qtypes = ["active_recall", "fill_blank"]
        job_id = str(uuid.uuid4())

        with patch("core.batch_client.SessionLocal", sf):
            from core.batch_client import BatchClient
            client = BatchClient(_client=MagicMock())

        with patch("core.batch_client.SessionLocal", sf):
            reqs = client.build_requests(job_id, [doc.id], subject.id, qtypes)

        assert len(reqs) == n_chunks * len(qtypes)

    def test_custom_id_format(self, sf, doc, subject, pdf_chunks):
        job_id = str(uuid.uuid4())
        with patch("core.batch_client.SessionLocal", sf):
            from core.batch_client import BatchClient
            client = BatchClient(_client=MagicMock())
            reqs = client.build_requests(job_id, [doc.id], subject.id, ["active_recall"])

        for req in reqs:
            parts = req["custom_id"].split(":")
            assert len(parts) == 3
            assert parts[0] == job_id
            assert parts[2] == "active_recall"

    def test_request_contains_model_and_tools(self, sf, doc, subject, pdf_chunks):
        job_id = str(uuid.uuid4())
        with patch("core.batch_client.SessionLocal", sf):
            from core.batch_client import BatchClient
            client = BatchClient(_client=MagicMock())
            reqs = client.build_requests(job_id, [doc.id], subject.id, ["active_recall"])

        params = reqs[0]["params"]
        assert "model" in params
        assert params["tools"][0]["name"] == "flashcard_output"
        assert params["tool_choice"]["type"] == "tool"

    def test_chunk_text_present_in_user_message(self, sf, doc, subject, pdf_chunks):
        """Source text from the PDF appears in the user message sent to the API."""
        job_id = str(uuid.uuid4())
        with patch("core.batch_client.SessionLocal", sf):
            from core.batch_client import BatchClient
            client = BatchClient(_client=MagicMock())
            reqs = client.build_requests(job_id, [doc.id], subject.id, ["active_recall"])

        # Every request should carry the chunk's text
        chunk_texts = {c.text[:40] for c in pdf_chunks}
        for req in reqs:
            user_msg = req["params"]["messages"][0]["content"]
            found = any(snippet in user_msg for snippet in chunk_texts)
            assert found, "Chunk text not found in batch request message"

    def test_numerical_prompt_includes_examples(self, sf, doc, subject, pdf_chunks):
        job_id = str(uuid.uuid4())
        with patch("core.batch_client.SessionLocal", sf):
            from core.batch_client import BatchClient
            client = BatchClient(_client=MagicMock())
            reqs = client.build_requests(job_id, [doc.id], subject.id, ["numerical"])

        system = reqs[0]["params"]["system"]
        assert "NOVEL" in system or "IoT" in system  # from _get_novel_numerical_examples

    def test_skips_chunks_with_existing_approved_cards(self, sf, db, doc, subject, subtopic, pdf_chunks):
        # Create an approved card for the subtopic
        fc = Flashcard(
            subject_id=subject.id,
            subtopic_id=subtopic.id,
            question="Q?",
            answer="A.",
            status="approved",
        )
        db.add(fc)
        db.commit()

        job_id = str(uuid.uuid4())
        with patch("core.batch_client.SessionLocal", sf):
            from core.batch_client import BatchClient
            client = BatchClient(_client=MagicMock())
            reqs = client.build_requests(job_id, [doc.id], subject.id, ["active_recall"])

        assert reqs == [], "Should produce no requests when all subtopics already have cards"

        # Cleanup
        db.delete(fc)
        db.commit()

    def test_empty_doc_ids_produces_no_requests(self, sf, subject):
        job_id = str(uuid.uuid4())
        with patch("core.batch_client.SessionLocal", sf):
            from core.batch_client import BatchClient
            client = BatchClient(_client=MagicMock())
            reqs = client.build_requests(job_id, [], subject.id, ["active_recall"])

        assert reqs == []

    def test_defaults_to_active_recall_when_qtypes_empty(self, sf, doc, subject, pdf_chunks):
        job_id = str(uuid.uuid4())
        with patch("core.batch_client.SessionLocal", sf):
            from core.batch_client import BatchClient
            client = BatchClient(_client=MagicMock())
            reqs = client.build_requests(job_id, [doc.id], subject.id, [])

        assert all(r["custom_id"].endswith(":active_recall") for r in reqs)


# ---------------------------------------------------------------------------
# Tests: submit
# ---------------------------------------------------------------------------

class TestSubmit:
    def _job(self, db, subject_id: int) -> str:
        job_id = str(uuid.uuid4())
        db.add(
            BatchJob(
                id=job_id,
                subject_id=subject_id,
                status="indexing",
                doc_ids="[]",
                question_types='["active_recall"]',
                request_count=0,
            )
        )
        db.commit()
        return job_id

    def _fake_requests(self, job_id: str, n: int = 2) -> list[dict]:
        return [
            {
                "custom_id": f"{job_id}:{i}:active_recall",
                "params": {"model": "claude-sonnet-4-6", "max_tokens": 1024, "messages": []},
            }
            for i in range(n)
        ]

    def test_calls_anthropic_batches_create(self, sf, db, subject):
        mock_ant = MagicMock()
        mock_ant.beta.messages.batches.create.return_value = MagicMock(id="batch-abc123")
        with patch("core.batch_client.SessionLocal", sf):
            from core.batch_client import BatchClient
            client = BatchClient(_client=mock_ant)
        job_id = self._job(db, subject.id)
        reqs = self._fake_requests(job_id)

        with patch("core.batch_client.SessionLocal", sf):
            result = client.submit(job_id, reqs)

        mock_ant.beta.messages.batches.create.assert_called_once()
        assert result == "batch-abc123"

    def test_stores_anthropic_batch_id_in_db(self, sf, db, subject):
        mock_ant = MagicMock()
        mock_ant.beta.messages.batches.create.return_value = MagicMock(id="batch-xyz")
        with patch("core.batch_client.SessionLocal", sf):
            from core.batch_client import BatchClient
            client = BatchClient(_client=mock_ant)
        job_id = self._job(db, subject.id)

        with patch("core.batch_client.SessionLocal", sf):
            client.submit(job_id, self._fake_requests(job_id, n=1))

        job = db.get(BatchJob, job_id)
        db.refresh(job)
        assert job.anthropic_batch_id == "batch-xyz"
        assert job.status == "submitted"

    def test_creates_batch_request_rows(self, sf, db, subject):
        mock_ant = MagicMock()
        mock_ant.beta.messages.batches.create.return_value = MagicMock(id="batch-rows")
        with patch("core.batch_client.SessionLocal", sf):
            from core.batch_client import BatchClient
            client = BatchClient(_client=mock_ant)
        job_id = self._job(db, subject.id)
        reqs = self._fake_requests(job_id, n=3)

        with patch("core.batch_client.SessionLocal", sf):
            client.submit(job_id, reqs)

        br_count = (
            db.query(BatchRequest).filter(BatchRequest.job_id == job_id).count()
        )
        assert br_count == 3

    def test_raises_without_api_key(self):
        with patch("core.batch_client.settings") as mock_settings:
            mock_settings.ANTHROPIC_API_KEY = ""
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                from core.batch_client import BatchClient
                BatchClient()


# ---------------------------------------------------------------------------
# Tests: collect
# ---------------------------------------------------------------------------

def _make_batch_status(processing_status: str, succeeded: int = 0, errored: int = 0, processing: int = 0):
    mock = MagicMock()
    mock.processing_status = processing_status
    mock.request_counts.succeeded = succeeded
    mock.request_counts.errored = errored
    mock.request_counts.processing = processing
    return mock


def _make_succeeded_result(custom_id: str, flashcards: list[dict]):
    """Build a mock Anthropic batch result object for a succeeded request."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "flashcard_output"
    tool_block.input = {"flashcards": flashcards}

    result_obj = MagicMock()
    result_obj.custom_id = custom_id
    result_obj.result.type = "succeeded"
    result_obj.result.message.content = [tool_block]
    return result_obj


class TestCollect:
    def _setup_job(self, db, subject, anthropic_batch_id: str = "batch-test") -> str:
        job_id = str(uuid.uuid4())
        db.add(
            BatchJob(
                id=job_id,
                subject_id=subject.id,
                status="submitted",
                doc_ids="[]",
                question_types='["active_recall"]',
                request_count=2,
                anthropic_batch_id=anthropic_batch_id,
                submitted_at=datetime.now(timezone.utc),
            )
        )
        db.commit()
        return job_id

    def test_returns_in_progress_when_batch_not_ended(self, sf, db, subject):
        mock_ant = MagicMock()
        mock_ant.beta.messages.batches.retrieve.return_value = _make_batch_status(
            "in_progress", succeeded=1, processing=1
        )
        with patch("core.batch_client.SessionLocal", sf):
            from core.batch_client import BatchClient
            client = BatchClient(_client=mock_ant)
        job_id = self._setup_job(db, subject)

        with patch("core.batch_client.SessionLocal", sf):
            result = client.collect(job_id)

        assert result.status == "in_progress"
        assert result.completed_requests == 1

    def test_returns_completed_for_already_done_job(self, sf, db, subject):
        job_id = str(uuid.uuid4())
        db.add(
            BatchJob(
                id=job_id,
                subject_id=subject.id,
                status="completed",
                doc_ids="[]",
                question_types='["active_recall"]',
                request_count=2,
                completed_count=3,
                anthropic_batch_id="batch-done",
            )
        )
        db.commit()

        mock_ant = MagicMock()
        with patch("core.batch_client.SessionLocal", sf):
            from core.batch_client import BatchClient
            client = BatchClient(_client=mock_ant)
            result = client.collect(job_id)

        assert result.status == "completed"
        assert result.completed_requests == 3
        mock_ant.beta.messages.batches.retrieve.assert_not_called()

    def test_creates_flashcards_when_batch_ends(self, sf, db, subject, doc, subtopic):
        mock_ant = MagicMock()
        mock_ant.beta.messages.batches.retrieve.return_value = _make_batch_status(
            "ended", succeeded=1
        )

        # Insert a chunk to reference in the custom_id
        chunk = ContentChunk(
            document_id=doc.id, text="Statistics text.", subtopic_id=subtopic.id, source_type="pdf"
        )
        db.add(chunk)
        db.commit()
        db.refresh(chunk)

        job_id = self._setup_job(db, subject)
        custom_id = f"{job_id}:{chunk.id}:active_recall"

        # Insert a pending BatchRequest
        db.add(BatchRequest(id=custom_id, job_id=job_id, chunk_id=chunk.id, question_type="active_recall"))
        db.commit()

        flashcard_payload = [
            {
                "question": "What is the mean?",
                "answer": "The arithmetic average.",
                "question_type": "active_recall",
                "rubric": [
                    {"criterion": "Accuracy", "description": "Correct definition."},
                    {"criterion": "Completeness", "description": "All aspects covered."},
                    {"criterion": "Clarity", "description": "Unambiguous phrasing."},
                ],
                "suggested_complexity": "simple",
            }
        ]
        mock_ant.beta.messages.batches.results.return_value = [
            _make_succeeded_result(custom_id, flashcard_payload)
        ]

        mock_critic = MagicMock()
        mock_critic.evaluate_flashcard.return_value = MagicMock(
            aggregate_score=4,
            rubric_scores_json='{"accuracy":4,"logic":4,"grounding":4,"clarity":4}',
            feedback="Good.",
            suggested_complexity="simple",
            should_reject=False,
            reject_reason="",
            error=None,
        )

        with (
            patch("core.batch_client.SessionLocal", sf),
            patch("repositories.sql.flashcard_repo.SessionLocal", sf),
        ):
            from core.batch_client import BatchClient
            from repositories.sql.flashcard_repo import FlashcardRepo
            client = BatchClient(_client=mock_ant)
            client._fc_repo = FlashcardRepo()
            client._critic = mock_critic
            result = client.collect(job_id)

        assert result.status == "completed"
        assert result.flashcards_created == 1
        assert result.flashcards_rejected == 0

        # Flashcard persisted in DB
        cards = db.query(Flashcard).filter(Flashcard.subject_id == subject.id).all()
        assert len(cards) >= 1
        assert cards[-1].question == "What is the mean?"

    def test_errored_request_does_not_create_flashcard(self, sf, db, subject, doc, subtopic):
        mock_ant = MagicMock()
        mock_ant.beta.messages.batches.retrieve.return_value = _make_batch_status(
            "ended", errored=1
        )
        chunk = ContentChunk(
            document_id=doc.id, text="Stats text.", subtopic_id=subtopic.id, source_type="pdf"
        )
        db.add(chunk)
        db.commit()
        db.refresh(chunk)

        job_id = self._setup_job(db, subject)
        custom_id = f"{job_id}:{chunk.id}:active_recall"
        db.add(BatchRequest(id=custom_id, job_id=job_id, chunk_id=chunk.id, question_type="active_recall"))
        db.commit()

        errored_result = MagicMock()
        errored_result.custom_id = custom_id
        errored_result.result.type = "errored"
        errored_result.result.error = "rate_limit"
        mock_ant.beta.messages.batches.results.return_value = [errored_result]

        with (
            patch("core.batch_client.SessionLocal", sf),
            patch("repositories.sql.flashcard_repo.SessionLocal", sf),
        ):
            from core.batch_client import BatchClient
            from repositories.sql.flashcard_repo import FlashcardRepo
            client = BatchClient(_client=mock_ant)
            client._fc_repo = FlashcardRepo()
            result = client.collect(job_id)

        assert result.flashcards_created == 0

    def test_auto_reject_low_grounding(self, sf, db, subject, doc, subtopic):
        mock_ant = MagicMock()
        mock_ant.beta.messages.batches.retrieve.return_value = _make_batch_status(
            "ended", succeeded=1
        )
        chunk = ContentChunk(
            document_id=doc.id, text="Stats.", subtopic_id=subtopic.id, source_type="pdf"
        )
        db.add(chunk)
        db.commit()
        db.refresh(chunk)

        job_id = self._setup_job(db, subject)
        custom_id = f"{job_id}:{chunk.id}:active_recall"
        db.add(BatchRequest(id=custom_id, job_id=job_id, chunk_id=chunk.id, question_type="active_recall"))
        db.commit()

        flashcard_payload = [
            {
                "question": "Ungrounded Q?",
                "answer": "A.",
                "question_type": "active_recall",
                "rubric": [
                    {"criterion": "C1", "description": "d1"},
                    {"criterion": "C2", "description": "d2"},
                    {"criterion": "C3", "description": "d3"},
                ],
                "suggested_complexity": "simple",
            }
        ]
        mock_ant.beta.messages.batches.results.return_value = [
            _make_succeeded_result(custom_id, flashcard_payload)
        ]

        mock_critic = MagicMock()
        mock_critic.evaluate_flashcard.return_value = MagicMock(
            aggregate_score=1,
            rubric_scores_json='{"accuracy":1,"logic":1,"grounding":1,"clarity":1}',
            feedback="Not grounded.",
            suggested_complexity="simple",
            should_reject=True,
            reject_reason="grounding_score=1/4",
            error=None,
        )

        with (
            patch("core.batch_client.SessionLocal", sf),
            patch("repositories.sql.flashcard_repo.SessionLocal", sf),
        ):
            from core.batch_client import BatchClient
            from repositories.sql.flashcard_repo import FlashcardRepo
            client = BatchClient(_client=mock_ant)
            client._fc_repo = FlashcardRepo()
            client._critic = mock_critic
            result = client.collect(job_id)

        assert result.flashcards_created == 1
        assert result.flashcards_rejected == 1


# ---------------------------------------------------------------------------
# Tests: PDF content (uses real file when available)
# ---------------------------------------------------------------------------

class TestPDFContent:
    def test_pdf_yields_non_empty_text(self):
        texts = _extract_pdf_pages()
        assert len(texts) >= 1
        assert all(len(t) > 50 for t in texts), "Expected meaningful text from PDF pages"

    def test_pdf_text_contains_statistics_vocabulary(self):
        texts = _extract_pdf_pages()
        combined = " ".join(texts).lower()
        keywords = ["mean", "median", "variance", "frequency", "data"]
        matched = [k for k in keywords if k in combined]
        assert len(matched) >= 3, f"Expected statistics vocabulary; found only: {matched}"

    def test_batch_requests_built_from_real_pdf_chunks(self, sf, doc, subject, pdf_chunks):
        """End-to-end: real PDF text → build_requests → valid Anthropic request dicts."""
        job_id = str(uuid.uuid4())
        with patch("core.batch_client.SessionLocal", sf):
            from core.batch_client import BatchClient
            client = BatchClient(_client=MagicMock())
            reqs = client.build_requests(job_id, [doc.id], subject.id, ["active_recall"])

        assert len(reqs) > 0
        for req in reqs:
            assert req["params"]["model"]
            msg_content = req["params"]["messages"][0]["content"]
            # At least some statistics text should be present
            assert len(msg_content) > 100
