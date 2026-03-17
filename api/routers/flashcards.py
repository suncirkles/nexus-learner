"""
api/routers/flashcards.py
---------------------------
Flashcard status management and bulk operations.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import (
    FlashcardResponse,
    FlashcardStatusUpdate,
    BulkStatusUpdate,
    BulkSubtopicAction,
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
