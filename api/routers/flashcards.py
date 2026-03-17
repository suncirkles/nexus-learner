"""
api/routers/flashcards.py
---------------------------
Flashcard status management, bulk operations, and subtopic queries.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import (
    FlashcardResponse,
    FlashcardStatusUpdate,
    BulkStatusUpdate,
    BulkSubtopicAction,
    FlashcardSourceResponse,
)
from api.dependencies import get_flashcard_service
from services.flashcard_service import FlashcardService

router = APIRouter(prefix="/flashcards", tags=["flashcards"])


@router.get("/subject/{subject_id}", response_model=List[FlashcardResponse])
def get_flashcards_by_subject(
    subject_id: int,
    status: Optional[str] = None,
    svc: FlashcardService = Depends(get_flashcard_service),
):
    return svc.get_by_subject(subject_id, status)


@router.get("/subtopic/{subtopic_id}", response_model=List[FlashcardResponse])
def get_flashcards_by_subtopic(
    subtopic_id: int,
    status: Optional[str] = None,
    svc: FlashcardService = Depends(get_flashcard_service),
):
    return svc.get_by_subtopic(subtopic_id, status)


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
