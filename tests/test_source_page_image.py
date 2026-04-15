"""
tests/test_source_page_image.py
--------------------------------
Verifies the full source-page-image pipeline:

  upload_and_spawn
    → file renamed to {canonical_doc_id}_{filename}
  GET /flashcards/chunk-page-image/{chunk_id}
    → cache hit returns PNG
    → cache miss renders from upload dir and returns PNG

Run with:
    PYTHONPATH=. pytest tests/test_source_page_image.py -v
"""
import base64
import io
import os
import uuid
from unittest.mock import MagicMock, PropertyMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_real_pdf() -> bytes:
    """Return bytes of a minimal valid single-page PDF using PyMuPDF."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "d and f block elements — test content")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


ROUTER_UUID    = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
CANONICAL_UUID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


# ---------------------------------------------------------------------------
# Test 1 — upload_and_spawn renames file to canonical doc_id prefix
# ---------------------------------------------------------------------------

def test_upload_renames_file_to_canonical_doc_id(tmp_path):
    """
    After upload_and_spawn the PDF on disk must be at
      {canonical_doc_id}_{filename}
    not
      {router_uuid}_{filename}

    Without this fix get_chunk_page_image scans for {canonical_uuid}_* and finds
    nothing — 422, falls back to text snippet instead of the page image.
    """
    # Pre-import before uuid is patched: agents.ingestion triggers a deep import
    # chain (langchain → transformers) that runs `from uuid import uuid4` at module
    # level and then calls `uuid4().hex`.  If the patch context is already active
    # when that import fires, `uuid4()` returns a plain string (from our uuid_seq)
    # and `.hex` raises AttributeError.  Force the import now so every `from uuid
    # import uuid4` binding is set to the real function before we patch.
    import agents.ingestion  # noqa: F401

    from fastapi.testclient import TestClient
    from api.app import app
    from core.config import Settings

    pdf_bytes  = _make_real_pdf()
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    # The first call to uuid.uuid4() inside upload_and_spawn generates doc_id —
    # we inject ROUTER_UUID here so we can verify the file gets renamed away from
    # that prefix.  Subsequent calls (job_id, any internal helpers) get fresh real
    # UUIDs pre-generated *before* the patch so they aren't affected by the mock.
    _real_uuids = [str(uuid.uuid4()) for _ in range(8)]
    uuid_seq = iter([ROUTER_UUID] + _real_uuids)

    mock_db      = MagicMock()
    mock_subject = MagicMock()
    mock_subject.id = 1
    mock_db.query.return_value.filter.return_value.first.return_value = mock_subject

    mock_agent = MagicMock()
    mock_agent.create_document_record.return_value = CANONICAL_UUID

    with (
        patch("api.routers.ingestion.uuid.uuid4", side_effect=uuid_seq),
        # abs_upload_dir is a @property — patch it on the class
        patch.object(Settings, "abs_upload_dir",
                     new_callable=PropertyMock, return_value=str(upload_dir)),
        patch("api.routers.ingestion._commit_volume"),
        patch("api.routers.ingestion._spawn_worker", return_value="thread"),
        # _IngestionAgent is a lazy local import — patch at source
        patch("agents.ingestion.IngestionAgent", return_value=mock_agent),
        patch("api.dependencies.get_db", return_value=iter([mock_db])),
    ):
        client = TestClient(app)
        resp   = client.post(
            "/ingestion/upload-and-spawn",
            data={
                "mode": "INDEXING",
                "subject_id": "1",
                "question_type": "active_recall",
                "target_topics": "[]",
            },
            files={"file": ("d and f block.pdf", pdf_bytes, "application/pdf")},
        )

    assert resp.status_code == 202, f"Upload failed: {resp.text}"

    files_on_disk = [f.name for f in upload_dir.iterdir()]
    assert files_on_disk, "No files written to upload dir"

    canonical_files = [n for n in files_on_disk if n.startswith(f"{CANONICAL_UUID}_")]
    router_files    = [n for n in files_on_disk if n.startswith(f"{ROUTER_UUID}_")]

    assert canonical_files, (
        f"BUG: file not renamed to canonical prefix '{CANONICAL_UUID}_*'. "
        f"Files present: {files_on_disk}. "
        "get_chunk_page_image scans for canonical prefix and returns 422."
    )
    assert not router_files, (
        f"BUG: original router-uuid file still on disk: {router_files}. "
        "Rename did not remove the old file."
    )


# ---------------------------------------------------------------------------
# Test 2 — get_chunk_page_image renders from upload dir (cache miss path)
# ---------------------------------------------------------------------------

def test_chunk_page_image_renders_from_upload_dir(tmp_path):
    """
    When no cached PNG exists, the endpoint must find the PDF by scanning
    the upload dir for {canonical_doc_id}_* and render the page on-demand.
    The rendered PNG must also be written to the page cache.
    """
    from fastapi.testclient import TestClient
    from api.app import app
    from core.config import Settings

    upload_dir = tmp_path / "uploads"
    cache_dir  = tmp_path / "page_cache"
    upload_dir.mkdir()
    cache_dir.mkdir()

    # Place a real PDF under the canonical prefix (as upload_and_spawn would after fix)
    pdf_path = upload_dir / f"{CANONICAL_UUID}_d and f block.pdf"
    pdf_path.write_bytes(_make_real_pdf())

    mock_src = {
        "source_type": "pdf",
        "source_url": None,
        "filename": "d and f block.pdf",
        "document_id": CANONICAL_UUID,
        "page_number": 0,
        "text": "d and f block elements — test content",
    }

    with (
        patch.object(Settings, "abs_page_cache_dir",
                     new_callable=PropertyMock, return_value=str(cache_dir)),
        patch.object(Settings, "abs_upload_dir",
                     new_callable=PropertyMock, return_value=str(upload_dir)),
        # get_chunk_source is called via svc (injected FlashcardService)
        patch("repositories.sql.flashcard_repo.FlashcardRepo.get_source_by_chunk",
              return_value=mock_src),
    ):
        client = TestClient(app)
        resp   = client.get("/flashcards/chunk-page-image/99")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    body = resp.json()
    assert "image_b64" in body, "Response missing image_b64"

    png_bytes = base64.b64decode(body["image_b64"])
    assert png_bytes[:4] == b"\x89PNG", "Returned bytes are not a valid PNG"
    assert body["page_number"] == 0

    # Side-effect: PNG must be written to cache
    cached = cache_dir / f"{CANONICAL_UUID}_p0000.png"
    assert cached.exists(), (
        "PNG not written to page_cache after on-demand render — "
        "next request will re-render instead of cache-hit."
    )


# ---------------------------------------------------------------------------
# Test 3 — get_chunk_page_image serves from page cache (cache hit path)
# ---------------------------------------------------------------------------

def test_chunk_page_image_serves_from_cache(tmp_path):
    """
    When a pre-generated PNG exists in page_cache (written by the indexing
    worker), the endpoint must return it without touching the upload dir.
    """
    import fitz
    from fastapi.testclient import TestClient
    from api.app import app
    from core.config import Settings

    upload_dir = tmp_path / "uploads"
    cache_dir  = tmp_path / "page_cache"
    upload_dir.mkdir()
    cache_dir.mkdir()

    # Pre-render a PNG as the indexing worker would
    cached_png = cache_dir / f"{CANONICAL_UUID}_p0000.png"
    fitz_doc = fitz.open()
    pg = fitz_doc.new_page()
    pg.insert_text((50, 100), "cached page content")
    pix = pg.get_pixmap()
    cached_png.write_bytes(pix.tobytes("png"))
    fitz_doc.close()

    mock_src = {
        "source_type": "pdf",
        "source_url": None,
        "filename": "d and f block.pdf",
        "document_id": CANONICAL_UUID,
        "page_number": 0,
        "text": "d and f block elements",
    }

    scanned_paths = []
    real_scandir  = os.scandir

    def spy_scandir(path):
        scanned_paths.append(str(path))
        return real_scandir(path)

    with (
        patch.object(Settings, "abs_page_cache_dir",
                     new_callable=PropertyMock, return_value=str(cache_dir)),
        patch.object(Settings, "abs_upload_dir",
                     new_callable=PropertyMock, return_value=str(upload_dir)),
        patch("repositories.sql.flashcard_repo.FlashcardRepo.get_source_by_chunk",
              return_value=mock_src),
        # os is imported inside the endpoint function; patch the stdlib directly
        patch("os.scandir", side_effect=spy_scandir),
    ):
        client = TestClient(app)
        resp   = client.get("/flashcards/chunk-page-image/99")

    assert resp.status_code == 200
    png_bytes = base64.b64decode(resp.json()["image_b64"])
    assert png_bytes[:4] == b"\x89PNG"

    upload_scans = [p for p in scanned_paths if "uploads" in p]
    assert not upload_scans, (
        f"BUG: upload dir was scanned even though cached PNG existed. "
        f"Cache-hit path should return early. Scanned: {upload_scans}"
    )
