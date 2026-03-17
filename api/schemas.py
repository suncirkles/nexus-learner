"""
api/schemas.py
---------------
Pydantic request/response models for the FastAPI layer.
Kept separate from ORM models so each layer can evolve independently.
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Subject schemas
# ---------------------------------------------------------------------------

class SubjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class SubjectResponse(BaseModel):
    id: int
    name: str
    is_archived: bool

    model_config = {"from_attributes": True}


class FlashcardStatsResponse(BaseModel):
    approved: int
    pending: int
    rejected: int


# ---------------------------------------------------------------------------
# Flashcard schemas
# ---------------------------------------------------------------------------

class FlashcardResponse(BaseModel):
    id: int
    subject_id: Optional[int]
    subtopic_id: Optional[int]
    chunk_id: Optional[int]
    question: str
    answer: str
    question_type: Optional[str]
    complexity_level: Optional[str]
    rubric: Optional[str]
    critic_rubric_scores: Optional[str]
    critic_score: Optional[int]
    critic_feedback: Optional[str]
    status: str
    mentor_feedback: Optional[str]
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}


class FlashcardStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(approved|pending|rejected)$")
    feedback: str = ""
    complexity_level: Optional[str] = None


class BulkStatusUpdate(BaseModel):
    flashcard_ids: List[int]
    status: str = Field(..., pattern="^(approved|pending|rejected)$")


class BulkSubtopicAction(BaseModel):
    subtopic_ids: List[int]
    action: str = Field(..., pattern="^(approve|reject)$")


# ---------------------------------------------------------------------------
# Topic schemas
# ---------------------------------------------------------------------------

class TopicResponse(BaseModel):
    id: int
    document_id: str
    name: str
    summary: Optional[str]

    model_config = {"from_attributes": True}


class TopicDeleteRequest(BaseModel):
    doc_id: str


# ---------------------------------------------------------------------------
# Library / Document schemas
# ---------------------------------------------------------------------------

class DocumentResponse(BaseModel):
    id: str
    filename: str
    title: Optional[str]
    source_type: Optional[str]
    created_at: Optional[datetime]
    relevance_rate: Optional[float]
    yield_rate: Optional[float]
    faithfulness_score: Optional[float]

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# System schemas
# ---------------------------------------------------------------------------

class ResetResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# New schemas for Phase 2b UI refactoring
# ---------------------------------------------------------------------------

class SubjectRenameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class GlobalStatsResponse(BaseModel):
    total: int
    approved: int
    pending: int
    rejected: int


class SubtopicResponse(BaseModel):
    id: int
    topic_id: int
    name: str
    summary: Optional[str]
    approved_count: int = 0
    pending_count: int = 0

    model_config = {"from_attributes": True}


class TopicWithSubtopicsResponse(BaseModel):
    id: int
    document_id: str
    name: str
    summary: Optional[str]
    subtopics: List["SubtopicResponse"] = []

    model_config = {"from_attributes": True}


class FlashcardSourceResponse(BaseModel):
    source_type: Optional[str]
    source_url: Optional[str]
    filename: Optional[str]
    document_id: Optional[str]
    page_number: Optional[int]
    text: Optional[str]
