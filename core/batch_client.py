"""
core/batch_client.py
---------------------
Anthropic Messages Batch API wrapper for offline PDF flashcard generation.

Typical usage (fire-and-forget):

    client = BatchClient()
    doc_id  = client.index_pdf_if_needed(Path("./notes.pdf"), subject_id=1)
    reqs    = client.build_requests(job_id, [doc_id], subject_id=1, question_types=["active_recall"])
    client.submit(job_id, reqs)
    # Process exits. Later:
    result  = client.collect(job_id)

Requires ANTHROPIC_API_KEY in .env. LiteLLM is NOT used here — the Anthropic
SDK is called directly because LiteLLM does not support the Messages Batch API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agents.critic import CriticAgent
from agents.socratic import PROMPTS, FlashcardOutput, _get_novel_numerical_examples
from core.config import settings
from core.database import (
    BatchJob,
    BatchRequest,
    ContentChunk,
    Document as DBDocument,
    Flashcard,
    SessionLocal,
    SubjectDocumentAssociation,
    Subtopic,
    Topic,
)
from repositories.sql.flashcard_repo import FlashcardRepo

logger = logging.getLogger(__name__)

_BATCH_TOOL_NAME = "flashcard_output"


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class CollectResult:
    """Returned by BatchClient.collect()."""
    status: str                          # "in_progress" | "completed" | "failed"
    completed_requests: int = 0
    total_requests: int = 0
    flashcards_created: int = 0
    flashcards_rejected: int = 0
    eta_seconds: Optional[float] = None  # None = unknown
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flashcard_tool_schema() -> dict:
    """Convert FlashcardOutput Pydantic model to Anthropic tool definition."""
    return {
        "name": _BATCH_TOOL_NAME,
        "description": "Generate 1-3 educational flashcards from the source text.",
        "input_schema": FlashcardOutput.model_json_schema(),
    }


def _hash_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# BatchClient
# ---------------------------------------------------------------------------

class BatchClient:
    """Wraps the Anthropic Messages Batch API for offline flashcard generation.

    Pass ``_client`` in tests to inject a mock Anthropic client without
    touching the network or requiring a real API key.
    """

    def __init__(self, api_key: Optional[str] = None, _client=None):
        if _client is not None:
            self._client = _client
        else:
            import anthropic as _sdk  # deferred — keeps import lightweight for non-batch code
            key = api_key or settings.ANTHROPIC_API_KEY
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required for batch processing. "
                    "Set it in your .env file."
                )
            self._client = _sdk.Anthropic(api_key=key)
        self._fc_repo = FlashcardRepo()
        self._critic = CriticAgent()

    # ------------------------------------------------------------------
    # Step 1 — index a PDF (idempotent via content hash)
    # ------------------------------------------------------------------

    def index_pdf_if_needed(self, pdf_path: Path, subject_id: int) -> str:
        """Index a PDF into the library if not already present, then link to subject.

        Returns the document UUID (new or existing).
        Requires PGVector (via DB_URL) to be reachable.
        """
        content_hash = _hash_file(pdf_path)
        with SessionLocal() as db:
            existing = (
                db.query(DBDocument)
                .filter(DBDocument.content_hash == content_hash)
                .first()
            )
            if existing:
                logger.info("Already indexed: %s (%s)", pdf_path.name, existing.id)
                self._link_subject(existing.id, subject_id)
                return existing.id

        doc_id = str(uuid.uuid4())
        logger.info("Indexing %s as %s …", pdf_path.name, doc_id)

        # Reuse the existing LangGraph INDEXING workflow unchanged
        from workflows.phase1_ingestion import phase1_graph

        initial_state = {
            "mode": "INDEXING",
            "file_path": str(pdf_path),
            "doc_id": doc_id,
            "subject_id": None,      # library documents have no subject initially
            "target_topics": [],
            "question_type": "active_recall",
            "chunks": [],
            "current_page": 0,
            "current_chunk_index": 0,
            "hierarchy": [],
            "subtopic_embeddings": [],
            "pending_qdrant_docs": [],
            "matched_subtopic_ids": None,
            "current_new_cards": [],
            "generated_flashcards": [],
            "status_message": "Batch CLI: starting indexing…",
        }

        for event in phase1_graph.stream(
            initial_state,
            config={"recursion_limit": 500},
        ):
            msg = event.get("status_message", "")
            if msg:
                logger.info("  [index] %s", msg)

        self._link_subject(doc_id, subject_id)
        return doc_id

    # ------------------------------------------------------------------
    # Step 2 — build Anthropic batch request dicts
    # ------------------------------------------------------------------

    def build_requests(
        self,
        job_id: str,
        doc_ids: list[str],
        subject_id: int,
        question_types: list[str],
    ) -> list[dict]:
        """Build Anthropic batch request dicts for all eligible chunks.

        Skips chunks whose subtopics already have pending/approved cards for
        the subject (same deduplication rule as node_ingest in phase1_ingestion.py).
        """
        if not question_types:
            question_types = ["active_recall"]

        tool = _flashcard_tool_schema()
        requests: list[dict] = []

        with SessionLocal() as db:
            rows = (
                db.query(
                    ContentChunk,
                    Subtopic.name.label("subtopic_name"),
                    Topic.name.label("topic_name"),
                )
                .join(Subtopic, Subtopic.id == ContentChunk.subtopic_id, isouter=True)
                .join(Topic, Topic.id == Subtopic.topic_id, isouter=True)
                .filter(ContentChunk.document_id.in_(doc_ids))
                .all()
            )

            for row in rows:
                chunk = row.ContentChunk
                subtopic_name = row.subtopic_name or "General Knowledge"
                topic_name = row.topic_name or "General Knowledge"

                for qtype in question_types:
                    # Deduplication: skip this (chunk, qtype) pair if the subject
                    # already has pending/approved cards of the same question_type
                    # for this subtopic. Different types are always allowed.
                    if chunk.subtopic_id is not None:
                        has_cards = (
                            db.query(Flashcard)
                            .filter(
                                Flashcard.subject_id == subject_id,
                                Flashcard.subtopic_id == chunk.subtopic_id,
                                Flashcard.question_type == qtype,
                                Flashcard.status.in_(["approved", "pending"]),
                            )
                            .count()
                            > 0
                        )
                        if has_cards:
                            continue
                    system_prompt = PROMPTS.get(qtype, PROMPTS["active_recall"])
                    if qtype == "numerical":
                        system_prompt += "\n\n" + _get_novel_numerical_examples()

                    user_content = (
                        f"Topic: {topic_name}\n"
                        f"Subtopic: {subtopic_name}\n\n"
                        f"Source text:\n\n{chunk.text}"
                    )
                    custom_id = f"{job_id}:{chunk.id}:{qtype}"
                    requests.append(
                        {
                            "custom_id": custom_id,
                            "params": {
                                "model": settings.BATCH_MODEL,
                                "max_tokens": 4096,
                                "system": system_prompt,
                                "messages": [
                                    {"role": "user", "content": user_content}
                                ],
                                "tools": [tool],
                                "tool_choice": {
                                    "type": "tool",
                                    "name": _BATCH_TOOL_NAME,
                                },
                            },
                        }
                    )

        logger.info("Built %d batch request(s) for job %s", len(requests), job_id)
        return requests

    # ------------------------------------------------------------------
    # Step 3 — submit to Anthropic
    # ------------------------------------------------------------------

    def submit(self, job_id: str, requests: list[dict]) -> str:
        """Submit requests to the Anthropic Batch API. Returns anthropic_batch_id.

        Writes BatchRequest rows for every request and updates the BatchJob
        status to "submitted". Safe to call once per job.
        """
        logger.info("Submitting %d request(s) to Anthropic Batch API…", len(requests))
        batch = self._client.beta.messages.batches.create(requests=requests)
        anthropic_batch_id = batch.id
        now = datetime.now(timezone.utc)

        with SessionLocal() as db:
            db.query(BatchJob).filter(BatchJob.id == job_id).update(
                {
                    "anthropic_batch_id": anthropic_batch_id,
                    "status": "submitted",
                    "request_count": len(requests),
                    "submitted_at": now,
                }
            )
            for req in requests:
                _, chunk_id_str, qtype = req["custom_id"].split(":", 2)
                db.merge(
                    BatchRequest(
                        id=req["custom_id"],
                        job_id=job_id,
                        chunk_id=int(chunk_id_str),
                        question_type=qtype,
                        status="pending",
                    )
                )
            db.commit()

        logger.info(
            "Batch submitted — job_id=%s anthropic_batch_id=%s",
            job_id,
            anthropic_batch_id,
        )
        return anthropic_batch_id

    # ------------------------------------------------------------------
    # Step 4 — collect results (call anytime, idempotent)
    # ------------------------------------------------------------------

    def collect(self, job_id: str) -> CollectResult:
        """Check batch status. If complete, download results and create flashcards.

        Safe to call multiple times — already-processed requests are skipped.
        Returns CollectResult with status="in_progress" when the batch is still
        running (call again later), or status="completed" when done.
        """
        with SessionLocal() as db:
            job = db.get(BatchJob, job_id)
            if job is None:
                return CollectResult(status="failed", error="Job not found")
            if job.status == "completed":
                return CollectResult(
                    status="completed",
                    completed_requests=job.completed_count,
                    total_requests=job.request_count,
                )
            anthropic_batch_id = job.anthropic_batch_id
            subject_id = job.subject_id
            submitted_at = job.submitted_at
            total_requests = job.request_count

        if not anthropic_batch_id:
            return CollectResult(status="failed", error="No anthropic_batch_id recorded")

        batch = self._client.beta.messages.batches.retrieve(anthropic_batch_id)
        completed = (batch.request_counts.succeeded or 0) + (
            batch.request_counts.errored or 0
        )

        # ETA estimate from elapsed time + completion rate
        eta_seconds = None
        if submitted_at and completed > 0 and total_requests > 0:
            elapsed = (datetime.now(timezone.utc) - submitted_at.replace(tzinfo=timezone.utc)
                       if submitted_at.tzinfo is None
                       else (datetime.now(timezone.utc) - submitted_at)
                       ).total_seconds()
            rate = completed / total_requests
            if rate > 0:
                eta_seconds = max(0.0, elapsed / rate - elapsed)

        if batch.processing_status != "ended":
            return CollectResult(
                status="in_progress",
                completed_requests=completed,
                total_requests=total_requests,
                eta_seconds=eta_seconds,
            )

        # Batch ended — download and process all results
        with SessionLocal() as db:
            db.query(BatchJob).filter(BatchJob.id == job_id).update(
                {"status": "collecting"}
            )
            db.commit()

        initial_status = "approved" if settings.AUTO_ACCEPT_CONTENT else "pending"
        flashcards_created = 0
        flashcards_rejected = 0

        for result in self._client.beta.messages.batches.results(anthropic_batch_id):
            custom_id = result.custom_id
            parts = custom_id.split(":", 2)
            if len(parts) != 3:
                logger.warning("Unexpected custom_id format: %s — skipping", custom_id)
                continue
            _, chunk_id_str, qtype = parts
            chunk_id = int(chunk_id_str)

            with SessionLocal() as db:
                br = db.get(BatchRequest, custom_id)
                if br and br.status != "pending":
                    continue  # already processed — idempotent

                if result.result.type != "succeeded":
                    err = getattr(result.result, "error", "unknown error")
                    if br:
                        db.query(BatchRequest).filter(
                            BatchRequest.id == custom_id
                        ).update({"status": "errored", "error": str(err)})
                        db.commit()
                    logger.warning("Request %s errored: %s", custom_id, err)
                    continue

                # Parse FlashcardOutput from the tool_use response block
                fc_output = None
                for block in result.result.message.content:
                    if block.type == "tool_use" and block.name == _BATCH_TOOL_NAME:
                        try:
                            fc_output = FlashcardOutput.model_validate(block.input)
                        except Exception as e:
                            logger.warning("Failed to parse flashcard output for %s: %s", custom_id, e)
                        break

                if not fc_output or not fc_output.flashcards:
                    if br:
                        db.query(BatchRequest).filter(
                            BatchRequest.id == custom_id
                        ).update({"status": "errored", "error": "No flashcards in response"})
                        db.commit()
                    continue

                chunk = db.get(ContentChunk, chunk_id)
                source_text = chunk.text if chunk else ""
                subtopic_id = chunk.subtopic_id if chunk else None

                created_ids: list[int] = []
                for card in fc_output.flashcards:
                    rubric_json = json.dumps([r.model_dump() for r in card.rubric])
                    saved = self._fc_repo.create(
                        subject_id=subject_id,
                        subtopic_id=subtopic_id,
                        chunk_id=chunk_id,
                        question=card.question,
                        answer=card.answer,
                        question_type=qtype,
                        rubric_json=rubric_json,
                        status=initial_status,
                    )
                    fc_id = saved["id"]
                    created_ids.append(fc_id)
                    flashcards_created += 1

                    # Run critic synchronously; rejection updates status in-place
                    critic_result = self._critic.evaluate_flashcard(
                        source_text=source_text,
                        question=card.question,
                        answer=card.answer,
                        flashcard_id=fc_id,
                    )
                    if not critic_result.error:
                        self._fc_repo.update_critic_scores(
                            flashcard_id=fc_id,
                            aggregate_score=critic_result.aggregate_score,
                            rubric_scores_json=critic_result.rubric_scores_json,
                            feedback=critic_result.feedback,
                            complexity_level=critic_result.suggested_complexity,
                        )
                        if critic_result.should_reject:
                            self._fc_repo.update_status(fc_id, "rejected")
                            flashcards_rejected += 1

                db.query(BatchRequest).filter(BatchRequest.id == custom_id).update(
                    {
                        "status": "succeeded",
                        "flashcard_ids": json.dumps(created_ids),
                    }
                )
                db.commit()

        # Mark job complete
        with SessionLocal() as db:
            db.query(BatchJob).filter(BatchJob.id == job_id).update(
                {
                    "status": "completed",
                    "completed_count": flashcards_created,
                    "completed_at": datetime.now(timezone.utc),
                }
            )
            db.commit()

        logger.info(
            "Job %s complete — %d card(s) created, %d rejected",
            job_id,
            flashcards_created,
            flashcards_rejected,
        )
        return CollectResult(
            status="completed",
            completed_requests=completed,
            total_requests=total_requests,
            flashcards_created=flashcards_created,
            flashcards_rejected=flashcards_rejected,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _link_subject(self, doc_id: str, subject_id: int) -> None:
        """Create SubjectDocumentAssociation if it does not already exist."""
        with SessionLocal() as db:
            exists = (
                db.query(SubjectDocumentAssociation)
                .filter(
                    SubjectDocumentAssociation.document_id == doc_id,
                    SubjectDocumentAssociation.subject_id == subject_id,
                )
                .first()
            )
            if not exists:
                db.add(
                    SubjectDocumentAssociation(
                        document_id=doc_id,
                        subject_id=subject_id,
                    )
                )
                db.commit()
