"""
api/routers/library.py
------------------------
Knowledge Library document management endpoints.
"""

from typing import List
from fastapi import APIRouter, Depends, status

from api.schemas import DocumentResponse
from api.dependencies import get_library_service
from services.library_service import LibraryService

router = APIRouter(prefix="/library", tags=["library"])


@router.get("/", response_model=List[DocumentResponse])
def list_documents(svc: LibraryService = Depends(get_library_service)):
    return svc.get_all_documents()


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, svc: LibraryService = Depends(get_library_service)):
    svc.delete_document(doc_id)
