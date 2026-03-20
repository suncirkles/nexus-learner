"""
api/routers/subjects.py
-------------------------
CRUD + archive/restore/delete endpoints for subjects.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import (
    SubjectCreate, SubjectResponse, SubjectWithStatsResponse, FlashcardStatsResponse,
    SubjectRenameRequest, GlobalStatsResponse, DocumentResponse,
)
from api.dependencies import get_subject_service
from services.subject_service import SubjectService

router = APIRouter(prefix="/subjects", tags=["subjects"])


@router.get("/", response_model=List[SubjectResponse])
def list_active_subjects(svc: SubjectService = Depends(get_subject_service)):
    return svc.get_all_active()


@router.post("/", response_model=SubjectResponse, status_code=status.HTTP_201_CREATED)
def create_subject(body: SubjectCreate, svc: SubjectService = Depends(get_subject_service)):
    return svc.create(body.name)


# Fixed-path routes must come BEFORE /{subject_id} parameterised routes
@router.get("/with-stats", response_model=List[SubjectWithStatsResponse])
def list_subjects_with_stats(svc: SubjectService = Depends(get_subject_service)):
    """Return all active subjects with pre-aggregated topic and flashcard counts (3 DB queries total)."""
    return svc.get_all_active_with_stats()


@router.get("/archived", response_model=List[SubjectResponse])
def list_archived_subjects(svc: SubjectService = Depends(get_subject_service)):
    return svc.get_all_archived()


@router.get("/global-stats", response_model=GlobalStatsResponse)
def get_global_stats(svc: SubjectService = Depends(get_subject_service)):
    return svc.get_global_stats()


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


@router.patch("/{subject_id}/rename", status_code=status.HTTP_204_NO_CONTENT)
def rename_subject(
    subject_id: int,
    body: SubjectRenameRequest,
    svc: SubjectService = Depends(get_subject_service),
):
    svc.rename(subject_id, body.name)


@router.get("/{subject_id}/documents", response_model=List[DocumentResponse])
def get_attached_documents(
    subject_id: int, svc: SubjectService = Depends(get_subject_service)
):
    return svc.get_attached_documents(subject_id)


@router.get("/{subject_id}/documents/available", response_model=List[DocumentResponse])
def get_available_documents(
    subject_id: int, svc: SubjectService = Depends(get_subject_service)
):
    return svc.get_available_documents(subject_id)


@router.post("/{subject_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def attach_document(
    subject_id: int, doc_id: str, svc: SubjectService = Depends(get_subject_service)
):
    svc.attach_document(subject_id, doc_id)


@router.delete("/{subject_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def detach_document(
    subject_id: int, doc_id: str, svc: SubjectService = Depends(get_subject_service)
):
    svc.detach_document(subject_id, doc_id)
