"""
api/routers/topics.py
-----------------------
Topic listing, subtopic queries, and cascade-delete endpoint (H6 invariant enforced in service).
"""

from typing import List
from fastapi import APIRouter, Depends, status

from api.schemas import TopicResponse, TopicDeleteRequest, SubtopicResponse
from api.dependencies import get_topic_service
from services.topic_service import TopicService

router = APIRouter(prefix="/topics", tags=["topics"])


@router.get("/document/{doc_id}", response_model=List[TopicResponse])
def get_topics_by_document(
    doc_id: str, svc: TopicService = Depends(get_topic_service)
):
    return svc.get_by_document(doc_id)


@router.get("/subject/{subject_id}", response_model=List[TopicResponse])
def get_topics_by_subject(
    subject_id: int, svc: TopicService = Depends(get_topic_service)
):
    return svc.get_by_subject(subject_id)


@router.get("/{topic_id}/subtopics", response_model=List[SubtopicResponse])
def get_subtopics(
    topic_id: int, svc: TopicService = Depends(get_topic_service)
):
    return svc.get_subtopics_with_counts(topic_id)


@router.delete("/{topic_id}", status_code=status.HTTP_200_OK)
def delete_topic(
    topic_id: int,
    body: TopicDeleteRequest,
    svc: TopicService = Depends(get_topic_service),
):
    """Delete topic cascade. Returns count of preserved approved cards."""
    preserved = svc.delete_topic_cascade(topic_id, body.doc_id)
    return {"preserved_approved_cards": preserved}
