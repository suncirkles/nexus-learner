"""
ui/api_client.py
-----------------
Thin httpx-based client wrapping all FastAPI calls used by the Streamlit pages.
All Streamlit pages import from here — none call services or repos directly.

Configuration:
    API_BASE_URL env var (default: http://127.0.0.1:8000)

Performance note: a module-level httpx.Client is reused across all calls so
that TCP connections are kept alive. Using 127.0.0.1 (not 'localhost') avoids
the Windows IPv6→IPv4 fallback which adds ~2-3 s per fresh connection.
"""

import logging
import os
import threading
from typing import List, Optional
from urllib.parse import urlparse, urlunparse

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

# Default to 127.0.0.1 — avoids IPv6 fallback delay on Windows.
# Use urlparse so the replacement is scoped to the host component only.
_raw_base = getattr(settings, "API_BASE_URL", os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
_parsed = urlparse(_raw_base)
if _parsed.hostname and _parsed.hostname.lower() == "localhost":
    _parsed = _parsed._replace(
        netloc=_parsed.netloc.lower().replace("localhost", "127.0.0.1")
    )
_BASE = urlunparse(_parsed)
_TIMEOUT = 30.0

# Module-level persistent client — connection pool is reused across all calls.
# httpx.Client is thread-safe; Streamlit may run callbacks on multiple threads.
_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _get_client() -> httpx.Client:
    """Return the shared httpx client, creating it if closed or missing.

    Always acquires the lock to avoid a TOCTOU race on _client.is_closed.
    The lock is uncontended in the steady state so the overhead is negligible.
    """
    global _client
    with _client_lock:
        if _client is None or _client.is_closed:
            _client = httpx.Client(
                base_url=_BASE,
                timeout=_TIMEOUT,
                headers={"Connection": "keep-alive"},
            )
    return _client


def _get_headers() -> dict:
    """Return headers including the current Streamlit session ID."""
    import streamlit as st
    headers = {"Connection": "keep-alive"}
    if "session_id" in st.session_state:
        headers["X-Session-ID"] = st.session_state.session_id
    return headers


def _get(path: str, **params) -> dict | list:
    r = _get_client().get(
        path, 
        params={k: v for k, v in params.items() if v is not None},
        headers=_get_headers()
    )
    r.raise_for_status()
    return r.json()


def _post(path: str, json: dict | None = None) -> dict | list | None:
    r = _get_client().post(path, json=json, headers=_get_headers())
    r.raise_for_status()
    if r.status_code == 204:
        return None
    return r.json() if r.content else None


def _patch(path: str, json: dict) -> None:
    r = _get_client().patch(path, json=json, headers=_get_headers())
    r.raise_for_status()


def _delete(path: str, json: dict | None = None) -> dict | None:
    kwargs = {"json": json} if json else {}
    r = _get_client().delete(path, headers=_get_headers(), **kwargs)
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


# ---------------------------------------------------------------------------
# Subjects — Phase 2b additions
# ---------------------------------------------------------------------------

def list_archived_subjects() -> List[dict]:
    return _get("/subjects/archived")  # type: ignore[return-value]


def rename_subject(subject_id: int, name: str) -> None:
    _patch(f"/subjects/{subject_id}/rename", json={"name": name})


def get_global_stats() -> dict:
    return _get("/subjects/global-stats")  # type: ignore[return-value]


def list_attached_documents(subject_id: int) -> List[dict]:
    return _get(f"/subjects/{subject_id}/documents")  # type: ignore[return-value]


def list_available_documents(subject_id: int) -> List[dict]:
    return _get(f"/subjects/{subject_id}/documents/available")  # type: ignore[return-value]


def attach_document(subject_id: int, doc_id: str) -> None:
    _post(f"/subjects/{subject_id}/documents/{doc_id}")


def detach_document(subject_id: int, doc_id: str) -> None:
    _delete(f"/subjects/{subject_id}/documents/{doc_id}")


# ---------------------------------------------------------------------------
# Topics — Phase 2b additions
# ---------------------------------------------------------------------------

def get_topics_by_subject(subject_id: int) -> List[dict]:
    return _get(f"/topics/subject/{subject_id}")  # type: ignore[return-value]


def get_subtopics_by_topic(topic_id: int) -> List[dict]:
    return _get(f"/topics/{topic_id}/subtopics")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Flashcards — Phase 2b additions
# ---------------------------------------------------------------------------

def get_flashcards_by_subtopic(
    subtopic_id: int, status: Optional[str] = None
) -> List[dict]:
    return _get(f"/flashcards/subtopic/{subtopic_id}", status=status)  # type: ignore[return-value]


def get_all_rejected_flashcards() -> List[dict]:
    return _get("/flashcards/rejected")  # type: ignore[return-value]


def delete_flashcard(flashcard_id: int) -> None:
    _delete(f"/flashcards/{flashcard_id}")


def get_chunk_source(chunk_id: int) -> Optional[dict]:
    try:
        return _get(f"/flashcards/chunk-source/{chunk_id}")  # type: ignore[return-value]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        logger.warning("get_chunk_source(%d) failed: %s", chunk_id, e)
        return None
    except Exception as e:
        logger.warning("get_chunk_source(%d) error: %s", chunk_id, e)
        return None
