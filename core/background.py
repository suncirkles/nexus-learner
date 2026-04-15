"""
core/background.py
-------------------
Background task manager for long-running flashcard generation.
After the initial synchronous burst is shown to the user, remaining work
is processed in a daemon thread. This module handles both:
  - PDF background tasks (remaining chunks after sync burst)
  - Web research background tasks (remaining topics after sync burst)
"""

import os
import threading
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Global registry to track active background threads and their progress
# status: "processing", "completed", "failed", "stopped"
background_tasks: Dict[str, Dict[str, Any]] = {}
stop_events: Dict[str, threading.Event] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# PDF background tasks
# ---------------------------------------------------------------------------

def run_document_generation(state: Dict[str, Any], doc_id: str, stop_event: threading.Event, filename: str = "Unknown"):
    """Runs the full Phase 1 LangGraph workflow in a background thread."""
    # Lazy import — keeps background.py importable without pulling in the full
    # LangGraph + agents + core.database chain into the Streamlit process.
    from workflows.phase1_ingestion import phase1_graph
    from core.context import set_request_id, set_session_id
    set_request_id(doc_id)
    set_session_id("background")
    file_path = state.get("file_path")
    try:
        logger.info(f"Background thread started for Document: {doc_id} ({filename})")
        logger.debug("[BG:%s] thread alive, overwriting pre-registration entry", doc_id)
        with _lock:
            background_tasks[doc_id] = {
                "status": "processing",
                "progress": 0,
                "total": 1,
                "filename": filename,
                "mode": state.get("mode", "INDEXING"),
                "is_web": False,
                "status_message": "Initializing...",
                # UI progress keys — initialised here so reads never KeyError
                "total_pages": 1,
                "current_page": 0,
                "current_chunk_index": 0,
                "chunks_in_page": 1,
                "total_chunks": 0,
                "flashcards_count": 0,
            }

        for event in phase1_graph.stream(state):
            if stop_event.is_set():
                logger.info(f"Stop signal received for Document: {doc_id}")
                with _lock:
                    background_tasks[doc_id]["status"] = "stopped"
                return

            # Update state with node results
            for node_name, node_update in event.items():
                if isinstance(node_update, dict):
                    state.update(node_update)
                    
                    with _lock:
                        task_info = background_tasks[doc_id]
                        if task_info["mode"] == "INDEXING":
                            if "total_pages" in node_update:
                                background_tasks[doc_id]["total_pages"] = node_update["total_pages"]
                            if "current_page" in node_update:
                                background_tasks[doc_id]["current_page"] = node_update["current_page"]
                            if "status_message" in node_update:
                                background_tasks[doc_id]["status_message"] = node_update["status_message"]
                        else:
                            # Generation Mode
                            if "chunks" in node_update:
                                total_chunks = len(node_update["chunks"])
                                background_tasks[doc_id]["total_chunks"] = total_chunks
                                background_tasks[doc_id]["status_message"] = f"Total chunks to process: {total_chunks}"
                            
                            if "current_chunk_index" in node_update:
                                background_tasks[doc_id]["current_chunk_index"] = node_update["current_chunk_index"]
                            
                            if "status_message" in node_update:
                                background_tasks[doc_id]["status_message"] = node_update["status_message"]
                            
                            if "generated_flashcards" in node_update:
                                background_tasks[doc_id]["flashcards_count"] = len(node_update["generated_flashcards"])

        with _lock:
            background_tasks[doc_id]["status"] = "completed"
            background_tasks[doc_id]["status_message"] = "Task Complete!"
        logger.info(f"Background thread finished for Document: {doc_id}")
        logger.debug("[BG:%s] status=completed", doc_id)
    except Exception as e:
        logger.error(f"Error in background generation for {doc_id}: {e}")
        logger.debug("[BG:%s] status=failed error=%s", doc_id, e)
        with _lock:
            background_tasks[doc_id] = {"status": "failed", "error": str(e), "is_web": False}
    finally:
        # Robust temp file cleanup with a small retry for Windows file locks
        if file_path and os.path.exists(file_path):
            import time
            for i in range(3):
                try:
                    os.remove(file_path)
                    logger.debug(f"Temporary file removed: {file_path}")
                    break
                except Exception as e:
                    if i < 2:
                        time.sleep(0.5)
                        continue
                    logger.warning(f"Failed to remove temporary file {file_path} after retries: {e}")


def start_background_task(state: Dict[str, Any], doc_id: str, filename: str = "Unknown"):
    """Spawns a new thread for full PDF background generation."""
    stop_event = threading.Event()
    stop_events[doc_id] = stop_event
    # Pre-register so the sidebar monitor can mount immediately on the next rerun,
    # before the thread has a chance to write its own entry.
    logger.debug("[BG:%s] pre-registered status=processing filename=%s", doc_id, filename)
    with _lock:
        background_tasks[doc_id] = {
            "status": "processing",
            "progress": 0,
            "total": 1,
            "filename": filename,
            "mode": state.get("mode", "INDEXING"),
            "is_web": False,
            "status_message": "Initializing...",
            "total_pages": 1,
            "current_page": 0,
            "current_chunk_index": 0,
            "chunks_in_page": 1,
            "total_chunks": 0,
            "flashcards_count": 0,
        }
    thread = threading.Thread(
        target=run_document_generation,
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
    # Lazy import — same reasoning as run_document_generation above.
    from workflows.phase2_web_ingestion import phase2_graph

    from core.context import set_request_id, set_session_id
    set_request_id(task_id)
    set_session_id("background")
    logger.info("Web research background thread started: %s (%d topics)", task_id, len(topics))
    logger.debug("[BG:%s] web thread alive", task_id)
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
                # H28: generated_flashcards is a list of card dicts (flashcard_id, question,
                # answer) — they have no "status" key. Count by presence of flashcard_id.
                successful = sum(1 for f in flashcards if f.get("flashcard_id") is not None)
                with _lock:
                    background_tasks[task_id]["flashcards_count"] = successful

        with _lock:
            background_tasks[task_id]["status"] = "completed"
        logger.info("Web research background thread finished: %s", task_id)
        logger.debug("[BG:%s] status=completed", task_id)

    except Exception as exc:
        logger.error("Error in web research background task %s: %s", task_id, exc)
        logger.debug("[BG:%s] status=failed error=%s", task_id, exc)
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
    # Pre-register so the sidebar monitor can mount immediately on the next rerun.
    logger.debug("[BG:%s] web pre-registered status=processing subject=%s topics=%d", task_id, subject_name, len(topics))
    with _lock:
        background_tasks[task_id] = {
            "status": "processing",
            "pages_current": 0,
            "pages_total": len(topics),
            "flashcards_count": 0,
            "display_name": f"Web Research — {subject_name}",
            "is_web": True,
        }
    thread = threading.Thread(
        target=run_web_research_background,
        args=(topics, subject_id, subject_name, task_id, stop_event),
        daemon=True,
    )
    thread.start()
    return thread


def stop_background_task(doc_id: str):
    """Signals a background thread (PDF or web) to stop."""
    logger.debug("[BG:%s] stop requested", doc_id)
    with _lock:
        if doc_id in stop_events:
            stop_events[doc_id].set()
            if doc_id in background_tasks:
                background_tasks[doc_id]["status"] = "stopped"
                logger.debug("[BG:%s] status=stopped", doc_id)


def clear_background_task(task_id: str):
    """Removes a completed/stopped/failed task from the registry under the lock."""
    logger.debug("[BG:%s] cleared from registry", task_id)
    with _lock:
        background_tasks.pop(task_id, None)
        stop_events.pop(task_id, None)
