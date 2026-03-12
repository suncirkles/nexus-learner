"""
core/background.py
-------------------
Background task manager for long-running flashcard generation.
After the initial synchronous burst of flashcards is shown to the user,
remaining chunks are processed in a daemon thread. This module provides
start, stop, and status tracking for these background tasks.
"""

import threading
import logging
from typing import Dict, Any
from workflows.phase1_ingestion import phase1_graph

logger = logging.getLogger(__name__)

# Global registry to track active background threads and their progress
# status: "processing", "completed", "failed", "stopped"
background_tasks: Dict[str, Dict[str, Any]] = {}
stop_events: Dict[str, threading.Event] = {}

def run_remaining_generation(state: Dict[str, Any], doc_id: str, stop_event: threading.Event, filename: str = "Unknown"):
    """
    Continues the LangGraph workflow for the remaining chunks in a background thread.
    Checks stop_event at each iteration.
    """
    try:
        logger.info(f"Background thread started for Document: {doc_id} ({filename})")
        background_tasks[doc_id] = {
            "status": "processing", 
            "progress": 0, 
            "total": len(state.get("chunks", [])),
            "filename": filename
        }
        
        # We resume the graph from the current state
        for event in phase1_graph.stream(state):
            # Check for interrupt
            if stop_event.is_set():
                logger.info(f"Stop signal received for Document: {doc_id}")
                background_tasks[doc_id]["status"] = "stopped"
                return

            if "generate" in event:
                idx = event["generate"].get("current_chunk_index", 0)
                background_tasks[doc_id]["progress"] = idx + 1
                
        background_tasks[doc_id]["status"] = "completed"
        logger.info(f"Background thread finished for Document: {doc_id}")
    except Exception as e:
        logger.error(f"Error in background generation for {doc_id}: {e}")
        background_tasks[doc_id] = {"status": "failed", "error": str(e)}

def start_background_task(state: Dict[str, Any], doc_id: str, filename: str = "Unknown"):
    """Spawns a new thread for background generation."""
    stop_event = threading.Event()
    stop_events[doc_id] = stop_event
    thread = threading.Thread(target=run_remaining_generation, args=(state, doc_id, stop_event, filename), daemon=True)
    thread.start()
    return thread

def stop_background_task(doc_id: str):
    """Signals a background thread to stop."""
    if doc_id in stop_events:
        stop_events[doc_id].set()
        if doc_id in background_tasks:
            background_tasks[doc_id]["status"] = "stopped"
