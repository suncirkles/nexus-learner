"""
core/background.py
-------------------
Background task manager for long-running flashcard generation.
After the initial synchronous burst is shown to the user, remaining work
is processed in a daemon thread. This module handles both:
  - PDF background tasks (remaining chunks after sync burst)
  - Web research background tasks (remaining topics after sync burst)
"""

import threading
import logging
from typing import Dict, Any, List
from workflows.phase1_ingestion import phase1_graph

logger = logging.getLogger(__name__)

# Global registry to track active background threads and their progress
# status: "processing", "completed", "failed", "stopped"
background_tasks: Dict[str, Dict[str, Any]] = {}
stop_events: Dict[str, threading.Event] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# PDF background tasks
# ---------------------------------------------------------------------------

def run_remaining_generation(state: Dict[str, Any], doc_id: str, stop_event: threading.Event, filename: str = "Unknown"):
    """Continues the LangGraph workflow for the remaining chunks in a background thread."""
    try:
        logger.info(f"Background thread started for Document: {doc_id} ({filename})")
        with _lock:
            background_tasks[doc_id] = {
                "status": "processing",
                "progress": 0,
                "total": len(state.get("chunks", [])),
                "filename": filename,
                "is_web": False,
            }

        for event in phase1_graph.stream(state):
            if stop_event.is_set():
                logger.info(f"Stop signal received for Document: {doc_id}")
                with _lock:
                    background_tasks[doc_id]["status"] = "stopped"
                return

            if "generate" in event:
                idx = event["generate"].get("current_chunk_index", 0)
                with _lock:
                    background_tasks[doc_id]["progress"] = idx + 1

        with _lock:
            background_tasks[doc_id]["status"] = "completed"
        logger.info(f"Background thread finished for Document: {doc_id}")
    except Exception as e:
        logger.error(f"Error in background generation for {doc_id}: {e}")
        with _lock:
            background_tasks[doc_id] = {"status": "failed", "error": str(e), "is_web": False}


def start_background_task(state: Dict[str, Any], doc_id: str, filename: str = "Unknown"):
    """Spawns a new thread for PDF background generation."""
    stop_event = threading.Event()
    stop_events[doc_id] = stop_event
    thread = threading.Thread(
        target=run_remaining_generation,
        args=(state, doc_id, stop_event, filename),
        daemon=True,
    )
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Web research background tasks
# ---------------------------------------------------------------------------

def run_web_research_background(
    topics: List[str],
    subject_id: int,
    subject_name: str,
    task_id: str,
    stop_event: threading.Event,
):
    """Runs the Phase 2 web ingestion pipeline for remaining topics in a background thread."""
    from workflows.phase2_web_ingestion import phase2_graph

    logger.info("Web research background thread started: %s (%d topics)", task_id, len(topics))
    with _lock:
        background_tasks[task_id] = {
            "status": "processing",
            "pages_current": 0,
            "pages_total": len(topics),   # placeholder; updated once research completes
            "flashcards_count": 0,
            "display_name": f"Web Research — {subject_name}",
            "is_web": True,
        }

    initial_state = {
        "subject_id": subject_id,
        "subject_name": subject_name,
        "topics": topics,
        "web_documents": [],
        "current_doc_index": 0,
        "doc_id": "",
        "full_text": "",
        "chunks": [],
        "hierarchy": [],
        "doc_summary": "",
        "current_chunk_index": 0,
        "generated_flashcards": [],
        "status_message": "",
        "safety_blocked": False,
        "safety_reason": "",
        "processed_urls": [],
        "status_callback": None,
        "stop_event": stop_event,
    }

    pages_ingested = 0   # local counter incremented per non-duplicate document

    try:
        for event in phase2_graph.stream(initial_state):
            if stop_event.is_set():
                logger.info("Stop signal received for web task: %s", task_id)
                with _lock:
                    background_tasks[task_id]["status"] = "stopped"
                return

            node_name = list(event.keys())[0]
            node_data = event[node_name]
            initial_state.update(node_data)

            if node_name == "research":
                # Now we know the actual number of pages found
                web_docs = node_data.get("web_documents") or []
                with _lock:
                    background_tasks[task_id]["pages_total"] = max(len(web_docs), 1)

            elif node_name == "ingest_web_document":
                # Count only non-duplicate ingestions (full_text is non-empty)
                if node_data.get("full_text"):
                    pages_ingested += 1
                    with _lock:
                        background_tasks[task_id]["pages_current"] = pages_ingested

            elif node_name == "generate":
                flashcards = node_data.get("generated_flashcards") or []
                successful = sum(1 for f in flashcards if f.get("status") == "success")
                with _lock:
                    background_tasks[task_id]["flashcards_count"] = successful

        with _lock:
            background_tasks[task_id]["status"] = "completed"
        logger.info("Web research background thread finished: %s", task_id)

    except Exception as exc:
        logger.error("Error in web research background task %s: %s", task_id, exc)
        with _lock:
            background_tasks[task_id] = {
                "status": "failed",
                "error": str(exc),
                "is_web": True,
                "display_name": f"Web Research — {subject_name}",
            }


def start_web_background_task(
    topics: List[str],
    subject_id: int,
    subject_name: str,
    task_id: str,
) -> threading.Thread:
    """Spawns a daemon thread for web research background processing."""
    stop_event = threading.Event()
    stop_events[task_id] = stop_event
    thread = threading.Thread(
        target=run_web_research_background,
        args=(topics, subject_id, subject_name, task_id, stop_event),
        daemon=True,
    )
    thread.start()
    return thread


def stop_background_task(doc_id: str):
    """Signals a background thread (PDF or web) to stop."""
    with _lock:
        if doc_id in stop_events:
            stop_events[doc_id].set()
            if doc_id in background_tasks:
                background_tasks[doc_id]["status"] = "stopped"
