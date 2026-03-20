"""
tests/unit/batch/test_batch_ingest_cli.py
------------------------------------------
Unit tests for scripts/batch_ingest.py CLI commands.

Uses click.testing.CliRunner so no real Anthropic calls, no real
PDF indexing, and no real filesystem access are made. All I/O is
patched; the SQLite DB is an in-memory instance shared across each test.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database import Base, BatchJob, Subject


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
    return sessionmaker(bind=engine, autobegin=True)


@pytest.fixture
def db(sf):
    with sf() as s:
        yield s


@pytest.fixture
def subject(db):
    subj = Subject(name=f"Stats_{uuid.uuid4().hex[:6]}")
    db.add(subj)
    db.commit()
    db.refresh(subj)
    return subj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _runner():
    return CliRunner()


def _fake_folder(tmp_path: Path, n_pdfs: int = 1) -> Path:
    folder = tmp_path / "pdfs"
    folder.mkdir()
    for i in range(n_pdfs):
        (folder / f"doc{i}.pdf").write_bytes(b"%PDF-1.4 fake")
    return folder


# ---------------------------------------------------------------------------
# submit command
# ---------------------------------------------------------------------------

class TestSubmitCommand:
    def test_requires_folder(self, sf, db):
        from scripts.batch_ingest import cli
        result = _runner().invoke(cli, ["submit", "--subject", "Physics"])
        assert result.exit_code != 0
        assert "folder" in result.output.lower()

    def test_requires_subject(self, sf, tmp_path):
        folder = _fake_folder(tmp_path)
        from scripts.batch_ingest import cli
        result = _runner().invoke(cli, ["submit", "--folder", str(folder)])
        assert result.exit_code != 0
        assert "subject" in result.output.lower()

    def test_fails_on_nonexistent_folder(self, sf, db, subject):
        from scripts.batch_ingest import cli
        with patch("scripts.batch_ingest.SessionLocal", sf):
            result = _runner().invoke(
                cli,
                ["submit", "--folder", "/no/such/path", "--subject", subject.name],
            )
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or result.exit_code == 1

    def test_resolves_existing_subject_by_name(self, sf, db, subject, tmp_path):
        folder = _fake_folder(tmp_path)
        mock_client = MagicMock()
        mock_client.index_pdf_if_needed.return_value = str(uuid.uuid4())
        mock_client.build_requests.return_value = []  # no requests → exits early

        from scripts.batch_ingest import cli
        with (
            patch("scripts.batch_ingest.SessionLocal", sf),
            patch("scripts.batch_ingest.BatchClient", return_value=mock_client),
        ):
            result = _runner().invoke(
                cli,
                ["submit", "--folder", str(folder), "--subject", subject.name.lower()],
            )

        assert result.exit_code == 0

    def test_creates_subject_with_create_flag(self, sf, db, tmp_path):
        folder = _fake_folder(tmp_path)
        new_name = f"NewSubject_{uuid.uuid4().hex[:6]}"
        mock_client = MagicMock()
        mock_client.index_pdf_if_needed.return_value = str(uuid.uuid4())
        mock_client.build_requests.return_value = []

        from scripts.batch_ingest import cli
        with (
            patch("scripts.batch_ingest.SessionLocal", sf),
            patch("scripts.batch_ingest.BatchClient", return_value=mock_client),
        ):
            result = _runner().invoke(
                cli,
                ["submit", "--folder", str(folder), "--subject", new_name, "--create-subject"],
            )

        assert "Created subject" in result.output or result.exit_code == 0
        created = db.query(Subject).filter(Subject.name == new_name).first()
        assert created is not None

    def test_fails_on_unknown_subject_without_create_flag(self, sf, db, tmp_path):
        folder = _fake_folder(tmp_path)
        from scripts.batch_ingest import cli
        with patch("scripts.batch_ingest.SessionLocal", sf):
            result = _runner().invoke(
                cli,
                ["submit", "--folder", str(folder), "--subject", "NoSuchSubject_xyz"],
            )
        assert result.exit_code != 0

    def test_submit_prints_job_summary(self, sf, db, subject, tmp_path):
        folder = _fake_folder(tmp_path)
        mock_client = MagicMock()
        mock_client.index_pdf_if_needed.return_value = str(uuid.uuid4())
        fake_reqs = [{"custom_id": f"{uuid.uuid4()}:1:active_recall", "params": {}}]
        mock_client.build_requests.return_value = fake_reqs
        mock_client.submit.return_value = "batch-test-abc"

        from scripts.batch_ingest import cli
        with (
            patch("scripts.batch_ingest.SessionLocal", sf),
            patch("scripts.batch_ingest.BatchClient", return_value=mock_client),
        ):
            result = _runner().invoke(
                cli,
                ["submit", "--folder", str(folder), "--subject", subject.name],
            )

        assert "batch-test-abc" in result.output
        assert "Job ID" in result.output
        assert result.exit_code == 0

    def test_accepts_yaml_config_file(self, sf, db, subject, tmp_path):
        folder = _fake_folder(tmp_path)
        config = tmp_path / "batch_config.yaml"
        config.write_text(
            f"folder: {folder}\nsubject: {subject.name}\nquestion_types:\n  - fill_blank\n"
        )
        mock_client = MagicMock()
        mock_client.index_pdf_if_needed.return_value = str(uuid.uuid4())
        mock_client.build_requests.return_value = []

        from scripts.batch_ingest import cli
        with (
            patch("scripts.batch_ingest.SessionLocal", sf),
            patch("scripts.batch_ingest.BatchClient", return_value=mock_client),
        ):
            result = _runner().invoke(cli, ["submit", "--config", str(config)])

        assert result.exit_code == 0
        # build_requests was called with fill_blank question type
        call_args = mock_client.build_requests.call_args
        assert "fill_blank" in call_args[0][3] or "fill_blank" in call_args[1].get("question_types", [])

    def test_exits_zero_when_no_new_chunks(self, sf, db, subject, tmp_path):
        folder = _fake_folder(tmp_path)
        mock_client = MagicMock()
        mock_client.index_pdf_if_needed.return_value = str(uuid.uuid4())
        mock_client.build_requests.return_value = []

        from scripts.batch_ingest import cli
        with (
            patch("scripts.batch_ingest.SessionLocal", sf),
            patch("scripts.batch_ingest.BatchClient", return_value=mock_client),
        ):
            result = _runner().invoke(
                cli,
                ["submit", "--folder", str(folder), "--subject", subject.name],
            )

        assert result.exit_code == 0
        assert "Nothing to submit" in result.output


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

class TestStatusCommand:
    def _insert_job(self, db, subject_id: int, status: str = "submitted") -> str:
        job_id = str(uuid.uuid4())
        db.add(
            BatchJob(
                id=job_id,
                subject_id=subject_id,
                status=status,
                doc_ids="[]",
                question_types='["active_recall"]',
                request_count=5,
                anthropic_batch_id="batch-xyz",
            )
        )
        db.commit()
        return job_id

    def test_no_active_jobs_message(self, sf):
        # Use a fresh isolated DB so no leftover jobs from other tests bleed in
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        fresh_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=fresh_engine)
        fresh_sf = sessionmaker(bind=fresh_engine, autobegin=True)

        from scripts.batch_ingest import cli
        with patch("scripts.batch_ingest.SessionLocal", fresh_sf):
            result = _runner().invoke(cli, ["status"])
        assert "No active" in result.output

    def test_shows_in_progress_status(self, sf, db, subject):
        job_id = self._insert_job(db, subject.id)
        mock_client = MagicMock()
        from core.batch_client import CollectResult
        mock_client.collect.return_value = CollectResult(
            status="in_progress", completed_requests=2, total_requests=5, eta_seconds=600
        )

        from scripts.batch_ingest import cli
        with (
            patch("scripts.batch_ingest.SessionLocal", sf),
            patch("scripts.batch_ingest.BatchClient", return_value=mock_client),
        ):
            result = _runner().invoke(cli, ["status"])

        assert "IN PROGRESS" in result.output or "in_progress" in result.output.lower()

    def test_shows_completed_status(self, sf, db, subject):
        job_id = self._insert_job(db, subject.id)
        mock_client = MagicMock()
        from core.batch_client import CollectResult
        mock_client.collect.return_value = CollectResult(
            status="completed",
            completed_requests=5,
            total_requests=5,
            flashcards_created=12,
            flashcards_rejected=1,
        )

        from scripts.batch_ingest import cli
        with (
            patch("scripts.batch_ingest.SessionLocal", sf),
            patch("scripts.batch_ingest.BatchClient", return_value=mock_client),
        ):
            result = _runner().invoke(cli, ["status"])

        assert "COMPLETE" in result.output
        assert "12" in result.output   # cards created

    def test_status_specific_job_id(self, sf, db, subject):
        job_id = self._insert_job(db, subject.id)
        mock_client = MagicMock()
        from core.batch_client import CollectResult
        mock_client.collect.return_value = CollectResult(status="in_progress", total_requests=5)

        from scripts.batch_ingest import cli
        with (
            patch("scripts.batch_ingest.SessionLocal", sf),
            patch("scripts.batch_ingest.BatchClient", return_value=mock_client),
        ):
            result = _runner().invoke(cli, ["status", "--job-id", job_id])

        assert result.exit_code == 0
        mock_client.collect.assert_called_once_with(job_id)

    def test_eta_displayed_in_minutes(self, sf, db, subject):
        self._insert_job(db, subject.id)
        mock_client = MagicMock()
        from core.batch_client import CollectResult
        mock_client.collect.return_value = CollectResult(
            status="in_progress", total_requests=10, completed_requests=2, eta_seconds=1800
        )

        from scripts.batch_ingest import cli
        with (
            patch("scripts.batch_ingest.SessionLocal", sf),
            patch("scripts.batch_ingest.BatchClient", return_value=mock_client),
        ):
            result = _runner().invoke(cli, ["status"])

        assert "30m" in result.output  # 1800s = 30 minutes


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------

class TestListCommand:
    def test_empty_list_message(self, sf, db):
        from scripts.batch_ingest import cli
        with patch("scripts.batch_ingest.SessionLocal", sf):
            result = _runner().invoke(cli, ["list"])
        # Either empty message or just header — no crash
        assert result.exit_code == 0

    def test_shows_all_jobs(self, sf, db, subject):
        for _ in range(3):
            db.add(
                BatchJob(
                    id=str(uuid.uuid4()),
                    subject_id=subject.id,
                    status="completed",
                    doc_ids="[]",
                    question_types='["active_recall"]',
                    request_count=5,
                    completed_count=8,
                )
            )
        db.commit()

        from scripts.batch_ingest import cli
        with patch("scripts.batch_ingest.SessionLocal", sf):
            result = _runner().invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "completed" in result.output
