"""
workflows/phase1_ingestion.py
------------------------------
LangGraph workflow for Phase 1: Knowledge Library Ingestion & Generation.
- INDEXING: Pure Document -> Topics/Subtopics mapping.
- GENERATION: Subject + Topics -> Targeted Flashcards.
"""

from typing import TypedDict, List, Annotated, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from agents.ingestion import IngestionAgent
from agents.socratic import SocraticAgent
from agents.critic import CriticAgent
from agents.curator import CuratorAgent
from agents.topic_assigner import TopicAssignerAgent
from agents.topic_matcher import TopicMatcherAgent
from core.database import SessionLocal, Topic, Subtopic, Flashcard, ContentChunk, Document as DBDocument, SubjectDocumentAssociation, SubjectTopicAssociation
from repositories.sql.flashcard_repo import FlashcardRepo
from repositories.vector.factory import get_vector_store
from core.config import settings
import logging
import os
import uuid as _uuid
import numpy as _np

logger = logging.getLogger(__name__)

# 1. Define the State
class GraphState(TypedDict):
    mode: str                 # "INDEXING" or "GENERATION"
    file_path: Optional[str]
    doc_id: str
    subject_id: Optional[int] # Only for GENERATION
    target_topics: List[str]  # Only for GENERATION
    question_type: str        # Card type for GENERATION (default: "active_recall")

    # State tracking
    total_pages: int
    current_page: int
    chunks: List[Any]
    current_chunk_index: int

    # Discovery context
    hierarchy: List[Dict[str, Any]]

    # Vector batch buffer: accumulates per-page docs, flushed at page boundary
    pending_vector_docs: List[Dict[str, Any]]  # {"text": str, "metadata": dict}

    # Semantic topic matching results (GENERATION only)
    # None  → no filter (all chunks)
    # []    → filter ran but found no matches
    # [ids] → filter to these subtopic IDs
    matched_subtopic_ids: Optional[List[int]]

    # Cards produced by the current chunk — consumed by node_critic
    current_new_cards: List[Dict[str, Any]]

    # Subtopic embeddings for fast similarity-based chunk assignment (INDEXING only)
    # Set by node_extract_hierarchy; empty list = fall back to per-chunk LLM assignment
    subtopic_embeddings: List[Dict[str, Any]]  # [{id, name, embedding}]

    # Results
    generated_flashcards: List[Dict[str, Any]]
    status_message: str

# 2. Lazy-init Agents — deferred until first graph execution so imports of
#    this module (e.g. `from workflows.phase1_ingestion import phase1_graph`)
#    no longer block startup with embedding-model loading.
_ingestion_agent: "IngestionAgent | None" = None
_curator_agent: "CuratorAgent | None" = None
_topic_assigner: "TopicAssignerAgent | None" = None
_topic_matcher: "TopicMatcherAgent | None" = None
_socratic_agent: "SocraticAgent | None" = None
_critic_agent: "CriticAgent | None" = None


def _ingestion() -> "IngestionAgent":
    global _ingestion_agent
    if _ingestion_agent is None:
        _ingestion_agent = IngestionAgent()
    return _ingestion_agent


def _curator() -> "CuratorAgent":
    global _curator_agent
    if _curator_agent is None:
        _curator_agent = CuratorAgent()
    return _curator_agent


def _assigner() -> "TopicAssignerAgent":
    global _topic_assigner
    if _topic_assigner is None:
        _topic_assigner = TopicAssignerAgent()
    return _topic_assigner


def _cosine_sim(a: List[float], b: List[float]) -> float:
    va, vb = _np.array(a, dtype=float), _np.array(b, dtype=float)
    denom = float(_np.linalg.norm(va) * _np.linalg.norm(vb))
    return float(_np.dot(va, vb) / denom) if denom > 0 else 0.0


def _matcher() -> "TopicMatcherAgent":
    global _topic_matcher
    if _topic_matcher is None:
        _topic_matcher = TopicMatcherAgent()
    return _topic_matcher


def _socratic() -> "SocraticAgent":
    global _socratic_agent
    if _socratic_agent is None:
        _socratic_agent = SocraticAgent()
    return _socratic_agent


def _critic() -> "CriticAgent":
    global _critic_agent
    if _critic_agent is None:
        _critic_agent = CriticAgent()
    return _critic_agent

# 3. Define Nodes

def node_match_topics(state: GraphState):
    """[GENERATION ONLY] Semantically maps user-provided topic names to indexed subtopic IDs.

    Uses TopicMatcherAgent (LLM-based) so that broad or loosely-worded user
    inputs like "RDD" still match indexed subtopics called "RDDs Overview" or
    "Resilient Distributed Datasets".

    Sets state["matched_subtopic_ids"]:
      None  → no target_topics supplied — all chunks will be used
      []    → topics supplied but no semantic match found
      [ids] → list of matching subtopic IDs to filter chunks by
    """
    if state.get("mode") != "GENERATION":
        return {}  # no-op for INDEXING

    target_topics = state.get("target_topics", [])
    doc_id = state["doc_id"]

    if not target_topics:
        return {
            "matched_subtopic_ids": None,
            "status_message": "No topic filter requested — all chunks will be used.",
        }

    db = SessionLocal()
    try:
        rows = (
            db.query(Subtopic, Topic.name.label("topic_name"))
            .join(Topic, Topic.id == Subtopic.topic_id)
            .filter(Topic.document_id == doc_id)
            .all()
        )
        indexed_subtopics = [
            {"id": r.Subtopic.id, "name": r.Subtopic.name, "topic_name": r.topic_name}
            for r in rows
        ]
    finally:
        db.close()

    if not indexed_subtopics:
        logger.warning(f"No subtopics indexed for doc {doc_id} — skipping semantic match.")
        return {
            "matched_subtopic_ids": [],
            "status_message": f"No indexed subtopics found for document {doc_id}.",
        }

    logger.info(
        f"TopicMatcher: matching {len(target_topics)} user topic(s) "
        f"against {len(indexed_subtopics)} indexed subtopics."
    )
    matches = _matcher().match_topics(target_topics, indexed_subtopics)

    matched_ids: List[int] = list({
        sid
        for m in matches
        for sid in m.matched_subtopic_ids
    })

    if matched_ids:
        logger.info("TopicMatcher: resolved %d subtopic ID(s): %s", len(matched_ids), matched_ids)
        return {
            "matched_subtopic_ids": matched_ids,
            "status_message": (
                f"Matched {len(matched_ids)} subtopic(s) for "
                f"{len(target_topics)} requested topic(s)."
            ),
        }
    else:
        logger.warning(
            "DIAG node_match_topics: NO MATCH — target_topics=%s "
            "indexed_subtopics=%s",
            target_topics,
            [s["name"] for s in indexed_subtopics],
        )
        return {
            "matched_subtopic_ids": [],
            "status_message": f"No indexed subtopics matched: {target_topics}",
        }


def node_ingest(state: GraphState):
    """
    INDEXING: Loads and chunks PDF pages globally.
    GENERATION: Loads existing chunks from library for targeted subtopics.
    """
    mode = state.get("mode", "INDEXING")
    doc_id = state["doc_id"]
    
    if mode == "INDEXING":
        file_path = state["file_path"]
        total_pages = state.get("total_pages") or _ingestion().get_page_count(file_path)
        current_page = state.get("current_page", 0)

        if current_page >= total_pages:
            return {"status_message": "Document indexing complete."}

        # Ensure document record exists on the first page.
        # upload_and_spawn creates it via create_document_record() before spawning the
        # worker, so this is normally a no-op.  It acts as a fallback for the legacy
        # direct-spawn path that skips upload_and_spawn.
        if current_page == 0:
            db = SessionLocal()
            try:
                # Prefer lookup by primary key (doc_id is known) — avoids any hash
                # mismatch between the upload path and this worker path.
                existing_doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
                if existing_doc:
                    db.expunge(existing_doc)  # detach before session close (C16)
                else:
                    # Fallback: document wasn't pre-created.  Strip UUID prefix that
                    # upload_and_spawn prepends so the stored filename is clean.
                    raw_basename = os.path.basename(file_path)
                    _parts = raw_basename.split("_", 1)
                    clean_filename = (
                        _parts[1]
                        if len(_parts) == 2 and len(_parts[0]) == 36 and _parts[0].count("-") == 4
                        else raw_basename
                    )
                    sample_text = _ingestion().load_page_text(file_path, 0)[:10000]
                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    content_hash = _ingestion().get_content_hash(sample_text + str(file_size) + clean_filename)
                    new_doc = DBDocument(
                        id=doc_id, filename=clean_filename, title=clean_filename,
                        content_hash=content_hash,
                    )
                    db.add(new_doc)
                    db.commit()
                    logger.info("node_ingest: created fallback Document record %s", doc_id)
            finally:
                db.close()

        logger.info(f"--- LIBRARY INDEXING: Page {current_page + 1}/{total_pages} ---")
        page_text = _ingestion().load_page_text(file_path, current_page)
        temp_chunks = _ingestion().text_splitter.split_text(page_text)

        # Render and cache the page image for source-snippet display in Mentor Review
        if file_path and file_path.lower().endswith(".pdf"):
            from core.config import settings as _settings
            cache_dir = _settings.abs_page_cache_dir  # /data/page_cache on Modal, ./page_cache locally
            img_path = os.path.join(cache_dir, f"{doc_id}_p{current_page:04d}.png")
            if not os.path.exists(img_path):
                try:
                    import fitz as _fitz
                    with _fitz.open(file_path) as _doc:
                        pix = _doc.load_page(current_page).get_pixmap(dpi=150)
                        pix.save(img_path)
                    logger.debug("Saved page image: %s", img_path)
                except Exception as _e:
                    logger.warning("Could not render page image for %s p%d: %s", doc_id, current_page, _e)

        return {
            "total_pages": total_pages,
            "chunks": temp_chunks,
            "current_chunk_index": 0,
            "current_page": current_page,
            "status_message": f"Global Indexing Page {current_page + 1}/{total_pages}...",
            "doc_id": doc_id
        }
    
    else:  # GENERATION mode
        logger.info("--- GENERATION: Loading Chunks for Targeted Subtopics ---")
        logger.info("DIAG node_ingest: doc_id=%s subject_id=%s", doc_id, state.get("subject_id"))
        matched_subtopic_ids = state.get("matched_subtopic_ids")  # set by node_match_topics
        logger.info("DIAG node_ingest: matched_subtopic_ids=%s", matched_subtopic_ids)

        db = SessionLocal()
        try:
            # Diagnostic: count all chunks for this doc regardless of subtopic assignment
            total_chunks_for_doc = db.query(ContentChunk).filter(
                ContentChunk.document_id == doc_id
            ).count()
            chunks_with_subtopic = db.query(ContentChunk).filter(
                ContentChunk.document_id == doc_id,
                ContentChunk.subtopic_id.isnot(None),
            ).count()
            topics_for_doc = db.query(Topic).filter(Topic.document_id == doc_id).all()
            logger.info(
                "DIAG node_ingest: doc has %d total chunks, %d with subtopic_id set, %d topics",
                total_chunks_for_doc, chunks_with_subtopic, len(topics_for_doc),
            )
            for t in topics_for_doc:
                subs = db.query(Subtopic).filter(Subtopic.topic_id == t.id).all()
                logger.info("DIAG   topic '%s' (id=%d) → %d subtopics: %s",
                            t.name, t.id, len(subs), [s.name for s in subs])

            # Join Subtopic to capture topic_id alongside the chunk — no extra queries.
            query = (
                db.query(ContentChunk, Subtopic.topic_id)
                .join(Subtopic, ContentChunk.subtopic_id == Subtopic.id)
                .join(Topic, Subtopic.topic_id == Topic.id)
                .filter(Topic.document_id == doc_id)
            )

            if matched_subtopic_ids is None:
                # No topic filter — process all chunks for this document
                logger.info("DIAG node_ingest: no topic filter — using all chunks with subtopic")
            elif matched_subtopic_ids:
                # Filter to semantically matched subtopics
                query = query.filter(ContentChunk.subtopic_id.in_(matched_subtopic_ids))
                logger.info("DIAG node_ingest: filtering to subtopic_ids %s", matched_subtopic_ids)
            else:
                # TopicMatcher ran but found no matches — nothing to process
                logger.warning(
                    "DIAG node_ingest: TopicMatcher returned [] — no chunks to process. "
                    "target_topics=%s", state.get("target_topics")
                )
                return {
                    "chunks": [],
                    "current_chunk_index": 0,
                    "status_message": "No chunks matched the requested topics.",
                }

            db_rows = query.all()
            logger.info("DIAG node_ingest: JOIN query returned %d chunk(s).", len(db_rows))

            question_type = state.get("question_type", "active_recall")
            chunks_to_process = []
            skipped_h22 = 0
            for c, topic_id in db_rows:
                # H22: only skip subtopics with approved/pending cards of the SAME
                # question_type — different types (e.g. active_recall vs numerical)
                # should always be allowed to generate independently.
                already_has_cards = db.query(Flashcard).filter(
                    Flashcard.subject_id == state["subject_id"],
                    Flashcard.subtopic_id == c.subtopic_id,
                    Flashcard.question_type == question_type,
                    Flashcard.status.in_(["approved", "pending"]),
                ).count() > 0
                if already_has_cards:
                    skipped_h22 += 1
                    continue
                chunks_to_process.append({
                    "id": c.id,
                    "text": c.text,
                    "subtopic_id": c.subtopic_id,
                    "topic_id": topic_id,   # full hierarchy: subject→topic→subtopic→flashcard
                })

            logger.info(
                "DIAG node_ingest: %d chunk(s) to process, %d skipped by H22 filter "
                "(already have %s cards for subject %s)",
                len(chunks_to_process), skipped_h22, question_type, state.get("subject_id"),
            )

            # Per-topic-type limit: cap chunks per (topic_id, question_type) so that
            # at most MAX_CARDS_PER_TOPIC_TYPE cards are generated per topic per type.
            # Existing approved/pending cards count toward the limit, so re-running only
            # fills remaining capacity. Hard-capped at 50 regardless of env setting.
            _topic_limit = min(max(int(settings.MAX_CARDS_PER_TOPIC_TYPE), 1), 50)
            subject_id_for_limit = state.get("subject_id")
            if subject_id_for_limit:
                from sqlalchemy import func as _func
                # Count existing cards per topic for this subject+question_type
                existing_per_topic: dict = {}
                unique_tids = {c["topic_id"] for c in chunks_to_process if c.get("topic_id")}
                if unique_tids:
                    rows = (
                        db.query(Flashcard.topic_id, _func.count(Flashcard.id))
                        .filter(
                            Flashcard.subject_id == subject_id_for_limit,
                            Flashcard.topic_id.in_(unique_tids),
                            Flashcard.question_type == question_type,
                            Flashcard.status.in_(["approved", "pending"]),
                        )
                        .group_by(Flashcard.topic_id)
                        .all()
                    )
                    existing_per_topic = {tid: cnt for tid, cnt in rows}

                # Keep at most (limit - existing) chunks per topic
                topic_chunk_counts: dict = {}
                limited_chunks = []
                for c in chunks_to_process:
                    tid = c.get("topic_id")
                    if tid is None:
                        limited_chunks.append(c)
                        continue
                    existing = existing_per_topic.get(tid, 0)
                    capacity = max(0, _topic_limit - existing)
                    used = topic_chunk_counts.get(tid, 0)
                    if used < capacity:
                        limited_chunks.append(c)
                        topic_chunk_counts[tid] = used + 1
                    # else: topic at capacity — skip this chunk

                skipped_limit = len(chunks_to_process) - len(limited_chunks)
                if skipped_limit:
                    logger.info(
                        "DIAG node_ingest: topic-type limit=%d — dropped %d chunk(s) "
                        "at capacity. Per-topic used: %s",
                        _topic_limit, skipped_limit, topic_chunk_counts,
                    )
                chunks_to_process = limited_chunks

            # Invalidate and refresh SubjectTopicAssociation for this subject+document.
            # This guarantees the explicit subject→topic bridge is current before any
            # flashcard insertion, so the full subject→topic→subtopic chain is valid.
            subject_id = state.get("subject_id")
            unique_topic_ids = {c["topic_id"] for c in chunks_to_process if c.get("topic_id")}
            if subject_id and unique_topic_ids:
                try:
                    db.query(SubjectTopicAssociation).filter(
                        SubjectTopicAssociation.subject_id == subject_id,
                        SubjectTopicAssociation.topic_id.in_(unique_topic_ids),
                    ).delete(synchronize_session=False)
                    for tid in unique_topic_ids:
                        db.add(SubjectTopicAssociation(subject_id=subject_id, topic_id=tid))
                    db.commit()
                    logger.info(
                        "DIAG node_ingest: refreshed SubjectTopicAssociation for "
                        "subject=%s topic_ids=%s", subject_id, sorted(unique_topic_ids),
                    )
                except Exception as _sta_err:
                    logger.warning("SubjectTopicAssociation refresh failed: %s", _sta_err)
                    db.rollback()

            return {
                "chunks": chunks_to_process,
                "current_chunk_index": 0,
                "status_message": f"Identified {len(chunks_to_process)} chunk(s) for card generation.",
            }
        finally:
            db.close()

def node_extract_hierarchy(state: GraphState):
    """[INDEXING ONLY] Extract topic/subtopic hierarchy via one CuratorAgent LLM call,
    persist all topics and subtopics upfront, then compute subtopic embeddings
    so that node_assign_topic can assign chunks via cosine similarity (no per-chunk LLM).

    Falls back gracefully: if CuratorAgent fails, returns subtopic_embeddings=[]
    and node_assign_topic falls back to the original per-chunk LLM path.
    """
    if state.get("mode") != "INDEXING":
        return {}  # no-op for GENERATION

    file_path = state["file_path"]
    doc_id = state["doc_id"]

    # ------------------------------------------------------------------
    # Ensure the Document record exists BEFORE inserting FK-dependent
    # Topic rows.  node_ingest also does this at current_page==0, but
    # extract_hierarchy runs first in the graph, so we must do it here.
    # If the content hash matches an existing document we reuse its id.
    # ------------------------------------------------------------------
    _doc_db = SessionLocal()
    try:
        sample_text = _ingestion().load_page_text(file_path, 0)[:10000]
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        content_hash = _ingestion().get_content_hash(
            sample_text + str(file_size) + os.path.basename(file_path)
        )
        existing = _doc_db.query(DBDocument).filter(DBDocument.content_hash == content_hash).first()
        if existing:
            doc_id = str(existing.id)
            logger.info("node_extract_hierarchy: reusing existing doc %s", doc_id)
        elif not _doc_db.query(DBDocument).filter(DBDocument.id == doc_id).first():
            clean_filename = os.path.basename(file_path)
            new_doc = DBDocument(
                id=doc_id,
                filename=clean_filename,
                title=clean_filename,
                content_hash=content_hash,
            )
            _doc_db.add(new_doc)
            _doc_db.commit()
            logger.info("node_extract_hierarchy: created document record %s", doc_id)
    except Exception as e:
        logger.error("node_extract_hierarchy: failed to ensure document record: %s", e)
    finally:
        _doc_db.close()

    # Read up to 10 pages (or total_pages if already set) to sample document content
    max_pages = state.get("total_pages") or _ingestion().get_page_count(file_path)
    text_parts: List[str] = []
    for p in range(min(max_pages, 10)):
        text_parts.append(_ingestion().load_page_text(file_path, p))
        if sum(len(t) for t in text_parts) >= 15000:
            break
    full_text = "\n\n".join(text_parts)

    # Build existing_structure_text from topics already indexed for this document
    # so the CuratorAgent can produce specific chapter-level topics (e.g.
    # "D and F Block Elements") rather than generic domain labels ("Chemistry").
    _ex_db = SessionLocal()
    try:
        existing_topics = (
            _ex_db.query(Topic)
            .filter(Topic.document_id == doc_id)
            .all()
        )
        if existing_topics:
            lines = []
            for t in existing_topics:
                subs = _ex_db.query(Subtopic).filter(Subtopic.topic_id == t.id).all()
                sub_names = ", ".join(s.name for s in subs) if subs else "none"
                lines.append(f"- {t.name}: {sub_names}")
            existing_structure_text = "\n".join(lines)
        else:
            existing_structure_text = "No existing topics."
    except Exception:
        existing_structure_text = "No existing topics."
    finally:
        _ex_db.close()

    try:
        result = _curator().curate_structure(full_text, existing_structure_text)
    except Exception as e:
        logger.warning(
            "CuratorAgent failed (%s) — falling back to per-chunk LLM topic assignment", e
        )
        return {"doc_id": doc_id, "subtopic_embeddings": [], "status_message": f"Hierarchy extraction failed: {e}"}

    # Persist topics and subtopics; collect metadata for embedding
    db = SessionLocal()
    subtopic_meta: List[Dict[str, Any]] = []
    hierarchy: List[Dict[str, Any]] = []
    try:
        for topic_data in result["hierarchy"]:
            topic_obj = db.query(Topic).filter(
                Topic.document_id == doc_id,
                Topic.name.ilike(topic_data["name"]),
            ).first()
            if not topic_obj:
                topic_obj = Topic(document_id=doc_id, name=topic_data["name"])
                db.add(topic_obj)
                db.commit()
                db.refresh(topic_obj)

            sub_names: List[str] = []
            for sub_data in topic_data["subtopics"]:
                sub_obj = db.query(Subtopic).filter(
                    Subtopic.topic_id == topic_obj.id,
                    Subtopic.name == sub_data["name"],
                ).first()
                if not sub_obj:
                    sub_obj = Subtopic(topic_id=topic_obj.id, name=sub_data["name"])
                    db.add(sub_obj)
                    db.commit()
                    db.refresh(sub_obj)
                subtopic_meta.append({
                    "id": sub_obj.id,
                    "name": sub_data["name"],
                    "summary": sub_data.get("summary", ""),
                })
                sub_names.append(sub_data["name"])
            hierarchy.append({"topic": topic_data["name"], "subtopics": sub_names})
    except Exception as e:
        db.rollback()
        logger.error("Failed to persist topic hierarchy: %s", e, exc_info=True)
        return {"doc_id": doc_id, "subtopic_embeddings": [], "status_message": f"Hierarchy persist failed: {e}"}
    finally:
        db.close()

    if not subtopic_meta:
        logger.warning("CuratorAgent returned empty hierarchy — falling back to per-chunk LLM.")
        return {"doc_id": doc_id, "subtopic_embeddings": [], "hierarchy": hierarchy, "status_message": "Empty hierarchy."}

    # Embed subtopic names + summaries using the local ONNX model (no network call)
    embed_texts = [
        f"{s['name']}: {s['summary']}" if s["summary"] else s["name"]
        for s in subtopic_meta
    ]
    vectors = _ingestion().embeddings.embed_documents(embed_texts)
    subtopic_embeddings = [
        {"id": s["id"], "name": s["name"], "embedding": v}
        for s, v in zip(subtopic_meta, vectors)
    ]

    logger.info(
        "node_extract_hierarchy: %d subtopic(s) extracted — chunk assignment will use cosine similarity.",
        len(subtopic_embeddings),
    )
    return {
        "doc_id": doc_id,   # may have been updated to an existing doc's id
        "hierarchy": hierarchy,
        "subtopic_embeddings": subtopic_embeddings,
        "status_message": f"Extracted {len(subtopic_embeddings)} subtopic(s) for {doc_id}.",
    }


def node_assign_topic(state: GraphState):
    """[INDEXING ONLY] Assigns a chunk to a subtopic and persists it.

    Fast path (default): cosine similarity against subtopic embeddings pre-computed
    by node_extract_hierarchy — no LLM call, runs in milliseconds.

    Fallback path: per-chunk LLM assignment via TopicAssignerAgent, used only when
    node_extract_hierarchy was skipped or failed (subtopic_embeddings is empty).
    """
    idx = state["current_chunk_index"]
    chunk_text = state["chunks"][idx]
    doc_id = state["doc_id"]
    hierarchy = state.get("hierarchy", [])
    pending = list(state.get("pending_vector_docs", []))
    subtopic_embeddings = state.get("subtopic_embeddings") or []

    assigned_subtopic_id = None

    if subtopic_embeddings:
        # Fast path: embed chunk and pick nearest subtopic by cosine similarity
        chunk_vec = _ingestion().embeddings.embed_query(chunk_text[:3000])
        _, assigned_subtopic_id = max(
            ((_cosine_sim(chunk_vec, s["embedding"]), s["id"]) for s in subtopic_embeddings),
            key=lambda x: x[0],
        )
    else:
        # Fallback: LLM-based assignment (original behaviour)
        try:
            assignment = _assigner().assign_topic(chunk_text, hierarchy)
            db = SessionLocal()
            try:
                topic_obj = db.query(Topic).filter(
                    Topic.document_id == doc_id,
                    Topic.name.ilike(assignment.topic_name),
                ).first()
                if not topic_obj:
                    topic_obj = Topic(document_id=doc_id, name=assignment.topic_name)
                    db.add(topic_obj)
                    db.commit()
                    db.refresh(topic_obj)
                    hierarchy.append({"topic": assignment.topic_name, "subtopics": [assignment.subtopic_name]})

                sub_obj = db.query(Subtopic).filter(
                    Subtopic.topic_id == topic_obj.id,
                    Subtopic.name == assignment.subtopic_name,
                ).first()
                if not sub_obj:
                    sub_obj = Subtopic(topic_id=topic_obj.id, name=assignment.subtopic_name)
                    db.add(sub_obj)
                    db.commit()
                    db.refresh(sub_obj)
                    for item in hierarchy:
                        if item["topic"] == assignment.topic_name and assignment.subtopic_name not in item["subtopics"]:
                            item["subtopics"].append(assignment.subtopic_name)

                assigned_subtopic_id = sub_obj.id
            except Exception as e:
                db.rollback()
                logger.error("LLM topic assignment DB write failed for chunk %d: %s", idx, e, exc_info=True)
            finally:
                db.close()
        except Exception as e:
            logger.error("LLM topic assignment failed for chunk %d: %s", idx, e, exc_info=True)

    # Persist the chunk (regardless of which assignment path ran)
    db = SessionLocal()
    try:
        new_chunk = ContentChunk(
            document_id=doc_id,
            text=chunk_text,
            subtopic_id=assigned_subtopic_id,
            page_number=state.get("current_page"),
        )
        db.add(new_chunk)
        db.commit()
        db.refresh(new_chunk)
        pending.append({"text": chunk_text, "metadata": {"document_id": doc_id, "db_chunk_id": new_chunk.id}})
        return {
            "hierarchy": hierarchy,
            "pending_vector_docs": pending,
            "status_message": f"Indexed chunk to subtopic id={assigned_subtopic_id}",
        }
    except Exception as e:
        db.rollback()
        logger.error("Error persisting chunk %d for doc %s: %s", idx, doc_id, e, exc_info=True)
        return {
            "hierarchy": hierarchy,
            "pending_vector_docs": pending,
            "status_message": f"Warning: failed to index chunk {idx} — {e}",
        }
    finally:
        db.close()


def _flush_vector_batch(pending_docs: List[Dict[str, Any]]):
    """Upserts a batch of pending docs to the configured vector store."""
    if not pending_docs:
        return

    try:
        store = get_vector_store()
        store.upsert_chunks(pending_docs)
    except Exception as e:
        logger.error(f"Vector batch flush failed ({len(pending_docs)} docs): {e}", exc_info=True)
        raise

def node_generate(state: GraphState):
    """[GENERATION ONLY] Subject-specific cards.

    Phase 2b: SocraticAgent.generate_flashcard() now returns FlashcardDraft objects
    with no DB side-effects. This node persists them via FlashcardRepo.create().
    AUTO_ACCEPT_CONTENT is read here (not inside the agent) so the agent has zero
    settings dependency.
    """
    idx = state["current_chunk_index"]
    chunk_data = state["chunks"][idx]
    subject_id = state["subject_id"]
    question_type = state.get("question_type", "active_recall")
    topic_id = chunk_data.get("topic_id")
    initial_status = "approved" if settings.AUTO_ACCEPT_CONTENT else "pending"

    # Hard card-level cap per (subject, topic, question_type).
    # node_ingest pre-filters chunks, but SocraticAgent returns 1-3 cards per chunk
    # so the chunk limit alone doesn't bound total cards.  Query current count once
    # per chunk (includes cards written earlier in this run) and stop saving once
    # the topic hits the limit.
    _topic_limit = min(max(int(settings.MAX_CARDS_PER_TOPIC_TYPE), 1), 50)
    _cards_already = 0
    if subject_id and topic_id:
        from sqlalchemy import func as _func
        _ldb = SessionLocal()
        try:
            _cards_already = (
                _ldb.query(_func.count(Flashcard.id))
                .filter(
                    Flashcard.subject_id == subject_id,
                    Flashcard.topic_id == topic_id,
                    Flashcard.question_type == question_type,
                    Flashcard.status.in_(["approved", "pending"]),
                )
                .scalar() or 0
            )
        finally:
            _ldb.close()

    drafts = _socratic().generate_flashcard(
        source_text=chunk_data["text"],
        question_type=question_type,
    )

    logger.info(
        "DIAG node_generate: chunk idx=%d subject_id=%s topic_id=%s subtopic_id=%s drafts=%d "
        "cards_already=%d limit=%d initial_status=%s",
        idx, subject_id, topic_id, chunk_data.get("subtopic_id"),
        len(drafts) if drafts else 0, _cards_already, _topic_limit, initial_status,
    )
    new_cards = []
    _cards_saved = 0
    if drafts:
        fc_repo = FlashcardRepo()
        for draft in drafts:
            if _cards_already + _cards_saved >= _topic_limit:
                logger.info(
                    "node_generate: topic %s at card limit (%d already + %d this chunk >= %d) "
                    "— skipping remaining draft(s)",
                    topic_id, _cards_already, _cards_saved, _topic_limit,
                )
                break
            try:
                saved = fc_repo.create(
                    subject_id=subject_id,
                    topic_id=chunk_data.get("topic_id"),
                    subtopic_id=chunk_data.get("subtopic_id"),
                    chunk_id=chunk_data["id"],
                    question=draft.question,
                    answer=draft.answer,
                    question_type=draft.question_type,
                    rubric_json=draft.rubric_json,
                    status=initial_status,
                )
                logger.info(
                    "DIAG node_generate: saved flashcard id=%s subject_id=%s subtopic_id=%s status=%s",
                    saved["id"], saved["subject_id"], saved["subtopic_id"], saved["status"],
                )
                new_cards.append({
                    "flashcard_id": saved["id"],
                    "question": saved["question"],
                    "answer": saved["answer"],
                    "question_type": saved["question_type"],
                    "suggested_complexity": draft.suggested_complexity,
                })
                _cards_saved += 1
            except Exception as _fc_err:
                logger.error(
                    "DIAG node_generate: fc_repo.create() FAILED for chunk %d: %s",
                    chunk_data.get("id"), _fc_err, exc_info=True,
                )

    return {
        "generated_flashcards": state.get("generated_flashcards", []) + new_cards,
        "current_new_cards": new_cards,
        "status_message": f"Generated {len(new_cards)} Q&A(s) for Subject {subject_id}.",
    }


def node_critic(state: GraphState):
    """[GENERATION ONLY] Evaluates ALL cards produced by the current chunk.

    Phase 2b: CriticAgent.evaluate_flashcard() is now pure LLM I/O — it returns a
    CriticResult dataclass with no DB side-effects. This node writes the scores back
    via FlashcardRepo so the agent has zero DB knowledge.
    """
    new_cards = state.get("current_new_cards", [])
    if not new_cards:
        return {"status_message": "No cards to verify."}

    source_text = state["chunks"][state["current_chunk_index"]]["text"]
    fc_repo = FlashcardRepo()

    for card in new_cards:
        fc_id = card.get("flashcard_id")
        if not fc_id:
            continue
        result = _critic().evaluate_flashcard(
            source_text=source_text,
            question=card["question"],
            answer=card["answer"],
            flashcard_id=fc_id,
        )
        if result.error:
            logger.warning("Critic failed for flashcard %d: %s", fc_id, result.error)
            continue
        fc_repo.update_critic_scores(
            flashcard_id=fc_id,
            aggregate_score=result.aggregate_score,
            rubric_scores_json=result.rubric_scores_json,
            feedback=result.feedback,
            complexity_level=result.suggested_complexity,
        )
        if result.should_reject:
            fc_repo.update_status(fc_id, "rejected")
            logger.info("Flashcard %d auto-rejected (%s).", fc_id, result.reject_reason)

    return {"status_message": "Verification complete.", "current_new_cards": []}


def node_increment(state: GraphState):
    return {"current_chunk_index": state["current_chunk_index"] + 1}


def node_next_page(state: GraphState):
    """Flushes the vector batch for the completed page, then advances."""
    _flush_vector_batch(state.get("pending_vector_docs", []))
    return {"current_page": state["current_page"] + 1, "pending_vector_docs": []}


def node_flush_vectors(state: GraphState):
    """Flushes the final vector batch at the end of INDEXING (last page)."""
    _flush_vector_batch(state.get("pending_vector_docs", []))
    return {"pending_vector_docs": [], "status_message": "Document indexing complete."}

# 4. Build Graph
workflow = StateGraph(GraphState)

workflow.add_node("match_topics", node_match_topics)
workflow.add_node("extract_hierarchy", node_extract_hierarchy)
workflow.add_node("ingest", node_ingest)
workflow.add_node("assign_topic", node_assign_topic)
workflow.add_node("generate", node_generate)
workflow.add_node("critic", node_critic)
workflow.add_node("increment", node_increment)
workflow.add_node("next_page", node_next_page)
workflow.add_node("flush_vectors", node_flush_vectors)

# match_topics is a no-op for INDEXING; for GENERATION it resolves topic names
# to subtopic IDs before node_ingest loads the chunk list.
# extract_hierarchy is a no-op for GENERATION; for INDEXING it runs CuratorAgent
# once upfront and computes subtopic embeddings for fast chunk assignment.
workflow.add_edge(START, "match_topics")
workflow.add_edge("match_topics", "extract_hierarchy")
workflow.add_edge("extract_hierarchy", "ingest")


def router_after_ingest(state: GraphState):
    if not state.get("chunks"):
        return END
    return "assign_topic" if state["mode"] == "INDEXING" else "generate"


workflow.add_conditional_edges(
    "ingest", router_after_ingest,
    {"assign_topic": "assign_topic", "generate": "generate", END: END},
)

workflow.add_edge("assign_topic", "increment")
workflow.add_edge("generate", "critic")
workflow.add_edge("critic", "increment")


def router_after_increment(state: GraphState):
    idx = state["current_chunk_index"]
    chunks = state.get("chunks", [])

    if state["mode"] == "GENERATION":
        # Card limit: configurable via MAX_CARDS_PER_PDF, hard cap of 50.
        _limit = min(max(int(settings.MAX_CARDS_PER_PDF), 1), 50)
        _total = len(state.get("generated_flashcards", []))
        if _total >= _limit:
            logger.info(
                "Generation limit reached: %d/%d cards — stopping early.",
                _total, _limit,
            )
            return END

    if idx < len(chunks):
        return "assign_topic" if state["mode"] == "INDEXING" else "generate"
    if state["mode"] == "INDEXING":
        if state["current_page"] < state["total_pages"] - 1:
            return "next_page"
        return "flush_vectors"  # last page: flush batch before END
    return END


workflow.add_conditional_edges(
    "increment", router_after_increment,
    {"assign_topic": "assign_topic", "generate": "generate",
     "next_page": "next_page", "flush_vectors": "flush_vectors", END: END},
)

workflow.add_edge("next_page", "ingest")
workflow.add_edge("flush_vectors", END)

phase1_graph = workflow.compile()
