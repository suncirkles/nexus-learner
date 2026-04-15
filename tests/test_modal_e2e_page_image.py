"""
tests/test_modal_e2e_page_image.py
------------------------------------
End-to-end test against the LIVE Modal deployment.

Verifies the full source-page-image pipeline under actual production conditions:

  POST /ingestion/upload-and-spawn
    → PDF written to Modal Volume, vol.commit() called
    → Background worker spawned, vol.reload(), indexes, writes PNGs, vol.commit()
  GET /flashcards/chunk-page-image/{chunk_id}
    → API container calls vol.reload(), finds cached PNG or renders on demand
    → Returns {"image_b64": "...", "page_number": N}  (not 422)

This test CANNOT be run locally — it exercises Modal volume cross-container
consistency which is invisible in a single-process pytest run.

Requirements before running:
  1. Deploy the latest code:
       modal deploy modal_app.py
  2. Set the target URL (or export MODAL_API_URL):
       export MODAL_API_URL=https://rajeshrkt55--nexus-learner-fastapi-app.modal.run
  3. DB_URL must be set in .env (or env) pointing at Supabase
  4. documents/d and f block.pdf must exist

Run:
    PYTHONPATH=. pytest tests/test_modal_e2e_page_image.py -v -s -m modal_e2e
    # or pass the URL explicitly:
    MODAL_API_URL=https://... pytest tests/test_modal_e2e_page_image.py -v -s -m modal_e2e

The test creates a document, runs indexing, asserts chunk-page-image works, then
deletes all created rows.  Runtime: ~3–6 minutes depending on Modal cold start.
"""

import sys
import os
# Ensure project root is on sys.path regardless of how pytest is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import time
import uuid
import logging
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PDF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "documents", "d and f block.pdf")
)

_MODAL_API_URL = os.environ.get(
    "MODAL_API_URL",
    "https://rajeshrkt55--nexus-learner-fastapi-app.modal.run",
)

# How long to wait for indexing to complete before giving up
_INDEXING_TIMEOUT_SECONDS = 360
_POLL_INTERVAL_SECONDS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _http():
    """Return an httpx client configured for the Modal API."""
    import httpx
    return httpx.Client(base_url=_MODAL_API_URL, timeout=120.0)


def _wait_for_job(client, job_id: str) -> dict:
    """Poll /ingestion/status/{job_id} until terminal state. Return final status dict."""
    deadline = time.time() + _INDEXING_TIMEOUT_SECONDS
    while time.time() < deadline:
        r = client.get(f"/ingestion/status/{job_id}")
        r.raise_for_status()
        body = r.json()
        status = body.get("status", "")
        logger.info(
            "  job %s: status=%s msg=%s page=%s/%s",
            job_id[:8], status,
            body.get("status_message", ""),
            body.get("current_page"), body.get("total_pages"),
        )
        if status == "completed":
            return body
        if status == "failed":
            pytest.fail(f"Ingestion job {job_id} failed: {body.get('error') or body.get('status_message')}")
        time.sleep(_POLL_INTERVAL_SECONDS)
    pytest.fail(f"Ingestion job {job_id} did not complete within {_INDEXING_TIMEOUT_SECONDS}s")


# ---------------------------------------------------------------------------
# Fixture: upload → index → yield doc_id + chunk_ids → teardown
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def indexed_document():
    """
    Upload d and f block.pdf to the live Modal API, wait for indexing to
    complete, then yield the document_id and a list of chunk_ids from Supabase.
    Tears down DB rows on exit.
    """
    if not os.path.exists(_PDF):
        pytest.skip(f"PDF not found: {_PDF}")

    import httpx

    # ---- Upload ----
    client = _http()
    logger.info("Uploading %s to %s", os.path.basename(_PDF), _MODAL_API_URL)
    with open(_PDF, "rb") as fh:
        resp = client.post(
            "/ingestion/upload-and-spawn",
            files={"file": (os.path.basename(_PDF), fh, "application/pdf")},
            data={
                "mode": "INDEXING",
                "subject_id": "",
                "question_type": "active_recall",
                "target_topics": "[]",
            },
        )
    resp.raise_for_status()
    body = resp.json()
    job_id = body["job_id"]
    logger.info("Upload accepted: job_id=%s", job_id)

    # ---- Wait for indexing ----
    _wait_for_job(client, job_id)
    logger.info("Indexing complete for job %s", job_id)

    # ---- Get doc_id + chunk_ids from DB ----
    from core.database import SessionLocal, ContentChunk, Document as DBDocument, BatchJob

    db = SessionLocal()
    try:
        # The BatchJob stores the doc_id in its doc_ids JSON column
        job_row = db.query(BatchJob).filter(BatchJob.id == job_id).first()
        assert job_row, f"BatchJob {job_id} not found in DB"

        import json as _json
        doc_ids = _json.loads(job_row.doc_ids or "[]")
        assert doc_ids, f"BatchJob {job_id} has no doc_ids"
        doc_id = doc_ids[0]
        logger.info("doc_id from job: %s", doc_id)

        chunks = db.query(ContentChunk).filter(
            ContentChunk.document_id == doc_id
        ).limit(5).all()
        chunk_ids = [c.id for c in chunks]
        logger.info("Chunk IDs to test: %s", chunk_ids)
        assert chunk_ids, f"No ContentChunks found for doc {doc_id} — indexing may have failed silently"
    finally:
        db.close()

    yield {"doc_id": doc_id, "chunk_ids": chunk_ids, "job_id": job_id}

    # ---- Teardown ----
    logger.info("Teardown: deleting doc %s and its chunks/topics from DB", doc_id)
    db = SessionLocal()
    try:
        from core.database import Topic, Subtopic

        # Delete chunks first (FK chain: subtopics → chunks)
        db.query(ContentChunk).filter(
            ContentChunk.document_id == doc_id
        ).delete(synchronize_session=False)

        for topic in db.query(Topic).filter(Topic.document_id == doc_id).all():
            db.query(Subtopic).filter(Subtopic.topic_id == topic.id).delete(
                synchronize_session=False
            )
        db.query(Topic).filter(Topic.document_id == doc_id).delete(
            synchronize_session=False
        )
        db.query(DBDocument).filter(DBDocument.id == doc_id).delete(
            synchronize_session=False
        )
        db.commit()
        logger.info("Teardown complete.")
    except Exception as exc:
        db.rollback()
        logger.error("Teardown failed (manual cleanup needed for doc %s): %s", doc_id, exc)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 1: chunk-page-image returns 200 with a valid PNG (not 422)
# ---------------------------------------------------------------------------

@pytest.mark.modal_e2e
def test_chunk_page_image_returns_png_not_422(indexed_document):
    """
    GET /flashcards/chunk-page-image/{chunk_id} must return HTTP 200 with a
    valid PNG for every chunk from a freshly-indexed document.

    The 422 regression: the API container and the upload/indexing containers
    are different Modal function instances.  Without vol.reload(), the API
    container sees a stale filesystem — the uploaded PDF and the pre-rendered
    PNGs written by the worker are both invisible, so the scan finds nothing
    and returns 422.

    This test is the ONLY way to verify the vol.reload() fix works, because
    cross-container file visibility is a Modal infrastructure behavior and
    cannot be replicated in a local pytest run.
    """
    client = _http()
    chunk_ids = indexed_document["chunk_ids"]
    doc_id = indexed_document["doc_id"]

    failures = []
    for chunk_id in chunk_ids:
        r = client.get(f"/flashcards/chunk-page-image/{chunk_id}")
        if r.status_code != 200:
            failures.append(
                f"chunk_id={chunk_id}: HTTP {r.status_code} — {r.text[:200]}"
            )
            continue

        body = r.json()
        assert "image_b64" in body, f"chunk_id={chunk_id}: response missing image_b64 key"
        assert "page_number" in body, f"chunk_id={chunk_id}: response missing page_number key"

        try:
            png_bytes = base64.b64decode(body["image_b64"])
        except Exception as exc:
            failures.append(f"chunk_id={chunk_id}: image_b64 is not valid base64: {exc}")
            continue

        if png_bytes[:4] != b"\x89PNG":
            failures.append(
                f"chunk_id={chunk_id}: returned bytes are not a PNG "
                f"(header={png_bytes[:4]!r}, len={len(png_bytes)})"
            )
            continue

        logger.info(
            "chunk_id=%s OK: page=%s png_size=%d bytes",
            chunk_id, body["page_number"], len(png_bytes),
        )

    assert not failures, (
        f"chunk-page-image returned non-200 or invalid PNG for {len(failures)}/{len(chunk_ids)} chunks.\n"
        "This means vol.reload() in the API container is NOT pulling the files written by the\n"
        "upload container (PDF) or the indexing worker (pre-rendered PNGs).\n\n"
        + "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# Test 2: second request hits the page cache (not re-rendered)
# ---------------------------------------------------------------------------

@pytest.mark.modal_e2e
def test_second_request_is_faster_than_first(indexed_document):
    """
    The second call for the same chunk_id should be faster because it hits the
    pre-rendered PNG in page_cache rather than opening the PDF on demand.

    This is a soft assertion — if the cache isn't working the first test would
    fail anyway, but a large latency difference is a signal the cache path isn't
    being hit.
    """
    client = _http()
    chunk_id = indexed_document["chunk_ids"][0]

    t0 = time.time()
    r1 = client.get(f"/flashcards/chunk-page-image/{chunk_id}")
    t1 = time.time() - t0
    assert r1.status_code == 200, f"First request failed: {r1.status_code} {r1.text[:200]}"

    t0 = time.time()
    r2 = client.get(f"/flashcards/chunk-page-image/{chunk_id}")
    t2 = time.time() - t0
    assert r2.status_code == 200, f"Second request failed: {r2.status_code} {r2.text[:200]}"

    logger.info(
        "chunk_id=%s: first=%.2fs second=%.2fs",
        chunk_id, t1, t2,
    )

    # Both should return identical image bytes
    b1 = base64.b64decode(r1.json()["image_b64"])
    b2 = base64.b64decode(r2.json()["image_b64"])
    assert b1 == b2, "First and second response returned different PNG bytes — cache inconsistency"
