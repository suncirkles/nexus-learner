"""
api/routers/subjects.py
-------------------------
CRUD + archive/restore/delete endpoints for subjects.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import SubjectCreate, SubjectResponse, FlashcardStatsResponse
from api.dependencies import get_subject_service
from services.subject_service import SubjectService

router = APIRouter(prefix="/subjects", tags=["subjects"])


@router.get("/", response_model=List[SubjectResponse])
def list_active_subjects(svc: SubjectService = Depends(get_subject_service)):
    return svc.get_all_active()


@router.post("/", response_model=SubjectResponse, status_code=status.HTTP_201_CREATED)
def create_subject(body: SubjectCreate, svc: SubjectService = Depends(get_subject_service)):
    return svc.create(body.name)


@router.get("/{subject_id}", response_model=SubjectResponse)
def get_subject(subject_id: int, svc: SubjectService = Depends(get_subject_service)):
    result = svc.get_by_id(subject_id)
    if not result:
        raise HTTPException(status_code=404, detail="Subject not found")
    return result


@router.post("/{subject_id}/archive", status_code=status.HTTP_204_NO_CONTENT)
def archive_subject(subject_id: int, svc: SubjectService = Depends(get_subject_service)):
    svc.archive(subject_id)


@router.post("/{subject_id}/restore", status_code=status.HTTP_204_NO_CONTENT)
def restore_subject(subject_id: int, svc: SubjectService = Depends(get_subject_service)):
    svc.restore(subject_id)


@router.delete("/{subject_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_subject(subject_id: int, svc: SubjectService = Depends(get_subject_service)):
    svc.permanent_delete(subject_id)


@router.get("/{subject_id}/stats", response_model=FlashcardStatsResponse)
def get_flashcard_stats(subject_id: int, svc: SubjectService = Depends(get_subject_service)):
    return svc.get_flashcard_stats(subject_id)
