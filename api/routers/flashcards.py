"""
api/routers/flashcards.py
---------------------------
Flashcard status management, bulk operations, and subtopic queries.
"""

import base64
import logging
import os

import fitz  # PyMuPDF
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import (
    FlashcardResponse,
    FlashcardStatusUpdate,
    BulkStatusUpdate,
    BulkSubtopicAction,
    FlashcardSourceResponse,
    ChunkSourceBatchRequest,
    ChunkSourceBatchResponse,
)
from api.dependencies import get_flashcard_service
from core.config import settings
from services.flashcard_service import FlashcardService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/flashcards", tags=["flashcards"])


@router.get("/subject/{subject_id}", response_model=List[FlashcardResponse])
def get_flashcards_by_subject(
    subject_id: int,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    question_type: Optional[str] = None,
    svc: FlashcardService = Depends(get_flashcard_service),
):
    return svc.get_by_subject(subject_id, status, skip, limit, question_type)


@router.get("/subtopic/{subtopic_id}", response_model=List[FlashcardResponse])
def get_flashcards_by_subtopic(
    subtopic_id: int,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    question_type: Optional[str] = None,
    svc: FlashcardService = Depends(get_flashcard_service),
):
    return svc.get_by_subtopic(subtopic_id, status, skip, limit, question_type)


@router.get("/rejected", response_model=List[FlashcardResponse])
def get_all_rejected(svc: FlashcardService = Depends(get_flashcard_service)):
    return svc.get_all_rejected()


@router.get("/chunk-source/{chunk_id}", response_model=FlashcardSourceResponse)
def get_chunk_source(
    chunk_id: int, svc: FlashcardService = Depends(get_flashcard_service)
):
    result = svc.get_chunk_source(chunk_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return result


@router.get("/chunk-page-image/{chunk_id}")
def get_chunk_page_image(
    chunk_id: int, svc: FlashcardService = Depends(get_flashcard_service)
):
    """Render the PDF page that a chunk came from and return it as a base64-encoded PNG.

    Returns {"image_b64": "<base64>", "page_number": <int>} on success,
    or 404 if the chunk/document is not found, or 422 if the file is unavailable.
    """
    src = svc.get_chunk_source(chunk_id)
    if src is None:
        raise HTTPException(status_code=404, detail="Chunk not found")

    page_number = src.get("page_number")
    doc_id = src.get("document_id")
    filename = src.get("filename")
    source_type = src.get("source_type", "pdf")

    if source_type != "pdf" or not doc_id or not filename:
        raise HTTPException(
            status_code=422,
            detail=f"No renderable PDF page: source_type={source_type!r} doc_id={doc_id!r} filename={filename!r} page_number={page_number!r}",
        )

    # Chunks indexed before page_number tracking was added will have page_number=None.
    # Default to page 0 so they at least show the document cover page.
    if page_number is None:
        page_number = 0

    # 1. Check the pre-generated page cache first (written by the indexing worker).
    #    On Modal this resolves to /data/page_cache — a persistent volume shared by
    #    all containers. Locally it resolves to ./page_cache.
    cached_path = os.path.join(settings.abs_page_cache_dir, f"{doc_id}_p{page_number:04d}.png")
    if os.path.exists(cached_path):
        with open(cached_path, "rb") as _f:
            png_bytes = _f.read()
        return {
            "image_b64": base64.b64encode(png_bytes).decode(),
            "page_number": page_number,
        }

    # 2. On Modal, this container may not have seen writes committed by other
    #    containers (the indexing worker writes PNGs; the upload container writes
    #    the PDF).  Call vol.reload() once to pull the latest volume state, then
    #    re-check the cache before falling through to on-demand rendering.
    if os.environ.get("MODAL_RUN") == "true":
        try:
            import modal as _modal
            _modal.Volume.from_name("nexus-learner-data").reload()
        except Exception as _ve:
            logger.warning("vol.reload() failed (non-fatal): %s", _ve)
        # Re-check cache after reload — worker may have committed PNGs since startup
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as _f:
                png_bytes = _f.read()
            return {
                "image_b64": base64.b64encode(png_bytes).decode(),
                "page_number": page_number,
            }

    # 3. Cache miss — render on-demand from the original PDF.
    #    File is stored as {doc_id}_{sanitized_filename}; scan to avoid
    #    having to replicate the exact sanitisation logic.
    upload_dir = settings.abs_upload_dir
    file_path = None
    prefix = f"{doc_id}_"
    try:
        for entry in os.scandir(upload_dir):
            if entry.name.startswith(prefix):
                file_path = entry.path
                break
    except FileNotFoundError:
        pass

    if not file_path:
        raise HTTPException(status_code=422, detail="PDF file not available on server")

    try:
        doc = fitz.open(file_path)
        if page_number >= len(doc):
            raise HTTPException(status_code=422, detail="Page number out of range")
        page = doc[page_number]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        png_bytes = pix.tobytes("png")
        doc.close()
        # Save to cache so subsequent requests are fast.
        try:
            with open(cached_path, "wb") as _f:
                _f.write(png_bytes)
        except Exception:
            pass  # Cache write failure is non-fatal
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to render page: {exc}") from exc

    return {
        "image_b64": base64.b64encode(png_bytes).decode(),
        "page_number": page_number,
    }


@router.post("/chunk-sources", response_model=ChunkSourceBatchResponse)
def get_chunk_sources_batch(
    body: ChunkSourceBatchRequest,
    svc: FlashcardService = Depends(get_flashcard_service),
):
    """Batch fetch source attribution for multiple chunk IDs (1 DB round-trip)."""
    sources = svc.get_chunk_sources_batch(body.chunk_ids)
    # JSON keys must be strings
    return ChunkSourceBatchResponse(sources={str(k): v for k, v in sources.items()})


@router.patch("/{flashcard_id}/status", status_code=status.HTTP_204_NO_CONTENT)
def update_flashcard_status(
    flashcard_id: int,
    body: FlashcardStatusUpdate,
    svc: FlashcardService = Depends(get_flashcard_service),
):
    svc.update_status(
        flashcard_id,
        body.status,
        body.feedback,
        body.complexity_level,
    )


@router.delete("/{flashcard_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_flashcard(
    flashcard_id: int,
    svc: FlashcardService = Depends(get_flashcard_service),
):
    svc.delete_one(flashcard_id)


@router.post("/bulk-status", status_code=status.HTTP_204_NO_CONTENT)
def bulk_update_status(
    body: BulkStatusUpdate,
    svc: FlashcardService = Depends(get_flashcard_service),
):
    svc.bulk_update_status(body.flashcard_ids, body.status)


@router.post("/bulk-subtopic-action", status_code=status.HTTP_204_NO_CONTENT)
def bulk_subtopic_action(
    body: BulkSubtopicAction,
    svc: FlashcardService = Depends(get_flashcard_service),
):
    """Approve or reject all pending flashcards for the given subtopics."""
    if body.action == "approve":
        svc.bulk_approve_subtopics(body.subtopic_ids)
    else:
        svc.bulk_reject_subtopics(body.subtopic_ids)
