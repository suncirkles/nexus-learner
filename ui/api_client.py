"""
ui/api_client.py
-----------------
Thin httpx-based client wrapping all FastAPI calls used by the Streamlit pages.
All Streamlit pages import from here — none call services or repos directly.

Configuration:
    API_BASE_URL env var (default: http://localhost:8000)
"""

import logging
from typing import List, Optional

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

_BASE = getattr(settings, "API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 30.0


def _get(path: str, **params) -> dict | list:
    with httpx.Client(timeout=_TIMEOUT) as client:
        r = client.get(f"{_BASE}{path}", params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        return r.json()


def _post(path: str, json: dict | None = None) -> dict | list | None:
    with httpx.Client(timeout=_TIMEOUT) as client:
        r = client.post(f"{_BASE}{path}", json=json or {})
        r.raise_for_status()
        if r.status_code == 204:
            return None
        return r.json() if r.content else None


def _patch(path: str, json: dict) -> None:
    with httpx.Client(timeout=_TIMEOUT) as client:
        r = client.patch(f"{_BASE}{path}", json=json)
        r.raise_for_status()


def _delete(path: str, json: dict | None = None) -> dict | None:
    with httpx.Client(timeout=_TIMEOUT) as client:
        kwargs = {"json": json} if json else {}
        r = client.delete(f"{_BASE}{path}", **kwargs)
        r.raise_for_status()
        return r.json() if r.content and r.status_code != 204 else None


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------

def list_active_subjects() -> List[dict]:
    return _get("/subjects/")  # type: ignore[return-value]


def create_subject(name: str) -> dict:
    return _post("/subjects/", json={"name": name})  # type: ignore[return-value]


def get_subject(subject_id: int) -> Optional[dict]:
    try:
        return _get(f"/subjects/{subject_id}")  # type: ignore[return-value]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise


def archive_subject(subject_id: int) -> None:
    _post(f"/subjects/{subject_id}/archive")


def restore_subject(subject_id: int) -> None:
    _post(f"/subjects/{subject_id}/restore")


def delete_subject(subject_id: int) -> None:
    _delete(f"/subjects/{subject_id}")


def get_flashcard_stats(subject_id: int) -> dict:
    return _get(f"/subjects/{subject_id}/stats")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Flashcards
# ---------------------------------------------------------------------------

def get_flashcards_by_subject(
    subject_id: int, status: Optional[str] = None
) -> List[dict]:
    return _get(f"/flashcards/subject/{subject_id}", status=status)  # type: ignore[return-value]


def update_flashcard_status(
    flashcard_id: int,
    status: str,
    feedback: str = "",
    complexity_level: Optional[str] = None,
) -> None:
    _patch(
        f"/flashcards/{flashcard_id}/status",
        json={"status": status, "feedback": feedback, "complexity_level": complexity_level},
    )


def bulk_update_status(flashcard_ids: List[int], status: str) -> None:
    _post("/flashcards/bulk-status", json={"flashcard_ids": flashcard_ids, "status": status})


def bulk_subtopic_action(subtopic_ids: List[int], action: str) -> None:
    """action: 'approve' or 'reject'"""
    _post(
        "/flashcards/bulk-subtopic-action",
        json={"subtopic_ids": subtopic_ids, "action": action},
    )


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------

def get_topics_by_document(doc_id: str) -> List[dict]:
    return _get(f"/topics/document/{doc_id}")  # type: ignore[return-value]


def delete_topic(topic_id: int, doc_id: str) -> dict:
    return _delete(f"/topics/{topic_id}", json={"doc_id": doc_id}) or {}  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

def list_documents() -> List[dict]:
    return _get("/library/")  # type: ignore[return-value]


def delete_document(doc_id: str) -> None:
    _delete(f"/library/{doc_id}")


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

def reset_system() -> dict:
    return _post("/system/reset")  # type: ignore[return-value]
