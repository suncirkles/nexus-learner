"""
api/routers/topics.py
-----------------------
Topic listing and cascade-delete endpoint (H6 invariant enforced in service).
"""

from typing import List
from fastapi import APIRouter, Depends, status

from api.schemas import TopicResponse, TopicDeleteRequest
from api.dependencies import get_topic_service
from services.topic_service import TopicService

router = APIRouter(prefix="/topics", tags=["topics"])


@router.get("/document/{doc_id}", response_model=List[TopicResponse])
def get_topics_by_document(
    doc_id: str, svc: TopicService = Depends(get_topic_service)
):
    return svc.get_by_document(doc_id)


@router.delete("/{topic_id}", status_code=status.HTTP_200_OK)
def delete_topic(
    topic_id: int,
    body: TopicDeleteRequest,
    svc: TopicService = Depends(get_topic_service),
):
    """Delete topic cascade. Returns count of preserved approved cards."""
    preserved = svc.delete_topic_cascade(topic_id, body.doc_id)
    return {"preserved_approved_cards": preserved}
