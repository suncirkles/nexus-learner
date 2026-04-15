"""
api/routers/ingestion.py
-------------------------
Endpoints for asynchronous document ingestion and flashcard generation.
Uses the "Spawn & Poll" pattern: BatchJob created immediately, worker runs async.

Worker priority:
  1. Modal Function.lookup().spawn()  — isolated container, 30-min timeout
  2. FastAPI background thread        — fallback; writes progress to DB so the
                                        Streamlit polling monitor still works
"""

import os
import json
import uuid
import logging
import threading
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from api.schemas import IngestionSpawnRequest, IngestionStatusResponse
from api.dependencies import get_db
from core.database import BatchJob, Subject
from core.config import settings

router = APIRouter(prefix="/ingestion", tags=["ingestion"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB-writing background thread (fallback when Modal spawn is unavailable)
# ---------------------------------------------------------------------------

def _run_ingestion_db_thread(
    job_id: str,
    doc_id: str,
    file_path: Optional[str],
    subject_id: Optional[int],
    mode: str,
    target_topics: list,
    question_type: str,
) -> None:
    """
    Runs the Phase 1 LangGraph pipeline in a daemon thread, writing granular
    progress to the BatchJob DB record so the Streamlit polling monitor works
    across container boundaries.

    Imports are lazy so this module stays light at import time.
    """
    from core.database import SessionLocal

    db = SessionLocal()
    try:
        # Lazy import — keeps the FastAPI container's startup fast
        from workflows.phase1_ingestion import phase1_graph

        job = db.query(BatchJob).filter(BatchJob.id == job_id).first()
        if not job:
            logger.error("DB thread: job %s not found", job_id)
            return

        current_state = {
            "mode": mode,
            "file_path": file_path,
            "doc_id": doc_id,
            "subject_id": subject_id,
            "target_topics": target_topics,
            "question_type": question_type,
            "total_pages": 0,
            "current_page": 0,
            "chunks": [],
            "current_chunk_index": 0,
            "hierarchy": [],
            "pending_qdrant_docs": [],
            "matched_subtopic_ids": None,
            "current_new_cards": [],
            "subtopic_embeddings": [],
            "generated_flashcards": [],
            "status_message": "Starting...",
        }

        job.status = "generating" if mode == "GENERATION" else "indexing"
        job.status_message = "Starting..."
        db.commit()

        _COMMIT_EVERY = 3
        _event_count = 0

        for event in phase1_graph.stream(current_state):
            for node_name, node_update in event.items():
                if not isinstance(node_update, dict):
                    continue
                current_state.update(node_update)

                if "status_message" in node_update:
                    job.status_message = node_update["status_message"]
                if "total_pages" in node_update:
                    job.total_pages = node_update["total_pages"]
                if "current_page" in node_update:
                    job.current_page = node_update["current_page"]
                if "current_chunk_index" in node_update:
                    job.current_chunk_index = node_update["current_chunk_index"]
                if "chunks" in node_update:
                    job.total_chunks = len(node_update["chunks"])
                if "generated_flashcards" in node_update:
                    job.flashcards_count = len(node_update["generated_flashcards"])

                _event_count += 1
                if _event_count % _COMMIT_EVERY == 0:
                    db.commit()

        job.status = "completed"
        job.status_message = "Done!"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("DB thread: job %s completed", job_id)

    except Exception as exc:
        logger.error("DB thread: job %s failed — %s", job_id, exc, exc_info=True)
        try:
            job = db.query(BatchJob).filter(BatchJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.error = str(exc)
                job.status_message = f"Error: {exc}"
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _commit_volume() -> None:
    """
    Flush this container's pending volume writes so other Modal containers can see
    the new files immediately.  No-op outside Modal or if the SDK call fails.
    """
    if os.environ.get("MODAL_RUN") != "true":
        return
    try:
        import modal
        vol = modal.Volume.from_name("nexus-learner-data")
        vol.commit()
        logger.debug("Volume committed after file write")
    except Exception as exc:
        logger.warning("Volume commit failed (non-fatal): %s", exc)


def _sanitize_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in " ._-").strip() or "upload.pdf"


def _spawn_worker(
    job_id: str,
    doc_id: str,
    file_path: Optional[str],
    subject_id: Optional[int],
    mode: str,
    target_topics: list,
    question_type: str,
    new_job: BatchJob,
    db: Session,
) -> Optional[str]:
    """
    Try Modal spawn first; fall back to FastAPI background thread.
    Returns the name of the spawner used, or sets new_job.status='failed'.
    """
    spawner: Optional[str] = None
    spawn_error: Optional[str] = None

    if os.environ.get("MODAL_RUN") == "true":
        try:
            import modal
            func = modal.Function.from_name("nexus-learner", "run_ingestion_background")
            func.spawn(job_id, doc_id, file_path, subject_id, mode, target_topics, question_type)
            spawner = "modal"
            logger.info("Spawned Modal worker for job %s", job_id)
        except Exception as exc:
            spawn_error = str(exc)
            logger.warning("Modal spawn failed for job %s (%s) — falling back to thread", job_id, exc)

    if spawner is None:
        try:
            thread = threading.Thread(
                target=_run_ingestion_db_thread,
                args=(job_id, doc_id, file_path, subject_id, mode, target_topics, question_type),
                daemon=True,
                name=f"ingestion-{job_id[:8]}",
            )
            thread.start()
            spawner = "thread"
            logger.info("Started fallback thread for job %s", job_id)
        except Exception as exc:
            combined = f"Modal: {spawn_error}; Thread: {exc}" if spawn_error else str(exc)
            logger.error("All spawn methods failed for job %s: %s", job_id, combined)
            new_job.status = "failed"
            new_job.error = combined
            new_job.status_message = f"Failed to start: {combined}"
            db.commit()
            return None

    return spawner


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/upload-and-spawn",
    response_model=IngestionStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def upload_and_spawn(
    file: UploadFile = File(...),
    mode: str = Form("INDEXING"),
    subject_id: Optional[int] = Form(None),
    question_type: str = Form("active_recall"),
    target_topics: str = Form("[]"),   # JSON-encoded list
    db: Session = Depends(get_db),
):
    """
    Single-request upload + spawn for INDEXING flows.

    Declared as a sync `def` so FastAPI runs it in a thread pool executor,
    making blocking operations (file I/O, volume.commit()) safe.  An `async def`
    would block the event loop on those calls and hang the app.

    The Streamlit container cannot reliably commit Modal Volume writes — it runs
    as a long-lived subprocess and never exits, so auto-commit never fires.
    By receiving the file HERE (in the FastAPI container), we write it to the
    volume and immediately call volume.commit(), making the file visible to the
    Modal worker before it is spawned.
    """
    import json as _json

    # 1. Parse target_topics
    try:
        topics: list = _json.loads(target_topics)
    except Exception:
        topics = []

    # 2. Validate subject
    if subject_id:
        subject = db.query(Subject).filter(Subject.id == subject_id).first()
        if not subject:
            raise HTTPException(status_code=404, detail="Subject not found")

    # 3. Write file to volume FROM THIS CONTAINER (FastAPI), then commit
    doc_id = str(uuid.uuid4())
    safe_name = _sanitize_filename(file.filename or "upload.pdf")
    upload_dir = settings.abs_upload_dir
    file_path = os.path.join(upload_dir, f"{doc_id}_{safe_name}")

    # Sync read from SpooledTemporaryFile — safe in thread pool
    contents = file.file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    logger.info("Saved uploaded file to %s (%d bytes)", file_path, len(contents))

    # Commit so the Modal worker container can see it immediately
    _commit_volume()

    # 3b. Create the Document record now — mirrors the old Streamlit-side call to
    #     ingestion_agent.create_document_record().  node_extract_hierarchy runs
    #     BEFORE node_ingest in the graph, so the FK-referenced documents row must
    #     exist before phase1_graph starts.  Content-hash dedup: if the same file
    #     was uploaded before, reuse the existing doc_id.
    from agents.ingestion import IngestionAgent as _IngestionAgent
    try:
        _agent = _IngestionAgent()
        canonical_doc_id = _agent.create_document_record(file_path, subject_id=subject_id)
        logger.info("Document record ensured: %s", canonical_doc_id)

        # The file was written with the router-generated UUID prefix, but
        # create_document_record generates its own UUID (or returns an existing one
        # for dedup).  Rename the file so its prefix matches the canonical doc_id —
        # this is what the page-image endpoint uses to locate the PDF later.
        if canonical_doc_id != doc_id:
            canonical_path = os.path.join(upload_dir, f"{canonical_doc_id}_{safe_name}")
            if not os.path.exists(canonical_path):
                os.rename(file_path, canonical_path)
            else:
                os.remove(file_path)  # dedup: canonical copy already exists
            file_path = canonical_path
            _commit_volume()
            logger.info("Renamed upload to canonical path: %s", file_path)

        doc_id = canonical_doc_id
    except Exception as _doc_exc:
        logger.warning("create_document_record failed (%s) — worker will create it", _doc_exc)

    # 4. Create BatchJob
    job_id = str(uuid.uuid4())
    new_job = BatchJob(
        id=job_id,
        subject_id=subject_id,
        status="queued",
        doc_ids=_json.dumps([doc_id]),
        question_types=_json.dumps([question_type]),
        filename=safe_name,
        status_message="Queued, waiting to start...",
        request_count=0,
        completed_count=0,
        created_at=datetime.now(timezone.utc),
    )
    db.add(new_job)
    db.commit()

    # 5. Spawn worker
    spawner = _spawn_worker(
        job_id, doc_id, file_path, subject_id, mode, topics, question_type, new_job, db
    )

    return IngestionStatusResponse(
        job_id=job_id,
        status=new_job.status,
        status_message=new_job.status_message,
        filename=safe_name,
        error=new_job.error,
    )


@router.post("/spawn", response_model=IngestionStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def spawn_ingestion(
    body: IngestionSpawnRequest,
    db: Session = Depends(get_db),
):
    """
    Creates a BatchJob record, then tries to run it:
      1. Modal Function.lookup().spawn()  — preferred (isolated, long-lived worker)
      2. FastAPI background thread        — automatic fallback; still DB-tracked

    Always returns 202 with the job_id so the UI can poll for progress.
    Errors are written to the BatchJob and returned to the caller inside the
    202 body — never a 500 that the UI would swallow.
    """
    # 1. Verify subject exists
    if body.subject_id:
        subject = db.query(Subject).filter(Subject.id == body.subject_id).first()
        if not subject:
            raise HTTPException(status_code=404, detail="Subject not found")

    # 2. Extract display filename
    #    INDEXING: strip the UUID prefix the UI adds to the file path.
    #    GENERATION: file_path is None — look up the document title from DB.
    display_filename = "Unknown"
    if body.file_path:
        raw = body.file_path.split("/")[-1].split("\\")[-1]
        parts = raw.split("_", 1)
        display_filename = parts[1] if len(parts) == 2 else raw
    elif body.doc_id:
        from core.database import Document as _Doc
        _doc = db.query(_Doc).filter(_Doc.id == body.doc_id).first()
        if _doc:
            display_filename = _doc.title or _doc.filename or display_filename

    # 3. Create BatchJob record
    job_id = str(uuid.uuid4())
    new_job = BatchJob(
        id=job_id,
        subject_id=body.subject_id,
        status="queued",
        doc_ids=json.dumps([body.doc_id]),
        question_types=json.dumps([body.question_type]),
        filename=display_filename,
        status_message="Queued, waiting to start...",
        request_count=0,
        completed_count=0,
        created_at=datetime.now(timezone.utc),
    )
    db.add(new_job)
    db.commit()
    logger.info("spawn_ingestion: created job %s mode=%s doc=%s", job_id, body.mode, body.doc_id)

    # 4. Spawn worker (Modal → thread fallback)
    _spawn_worker(
        job_id, body.doc_id, body.file_path, body.subject_id,
        body.mode, body.target_topics, body.question_type, new_job, db,
    )

    return IngestionStatusResponse(
        job_id=job_id,
        status=new_job.status,
        status_message=new_job.status_message,
        filename=display_filename,
        error=new_job.error,
    )


@router.get("/status/{job_id}", response_model=IngestionStatusResponse)
def get_ingestion_status(job_id: str, db: Session = Depends(get_db)):
    """Polls the status (and granular progress) of a background ingestion job."""
    job = db.query(BatchJob).filter(BatchJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    msg = job.status_message
    if not msg:
        msg = job.error if job.status == "failed" else f"Processed {job.completed_count} chunks."

    return IngestionStatusResponse(
        job_id=job.id,
        status=job.status,
        status_message=msg,
        error=job.error,
        filename=job.filename,
        total_pages=job.total_pages or 0,
        current_page=job.current_page or 0,
        total_chunks=job.total_chunks or 0,
        current_chunk_index=job.current_chunk_index or 0,
        flashcards_count=job.flashcards_count or 0,
    )
