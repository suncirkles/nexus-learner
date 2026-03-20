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
from core.database import SessionLocal, Topic, Subtopic, Flashcard, ContentChunk, Document as DBDocument, SubjectDocumentAssociation
from repositories.sql.flashcard_repo import FlashcardRepo
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

    # Qdrant batch buffer: accumulates per-page docs, flushed at page boundary
    pending_qdrant_docs: List[Dict[str, Any]]  # {"text": str, "metadata": dict}

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
        logger.info(f"TopicMatcher: resolved {len(matched_ids)} subtopic ID(s): {matched_ids}")
        return {
            "matched_subtopic_ids": matched_ids,
            "status_message": (
                f"Matched {len(matched_ids)} subtopic(s) for "
                f"{len(target_topics)} requested topic(s)."
            ),
        }
    else:
        logger.warning(f"TopicMatcher: no subtopics matched for topics {target_topics}")
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

        # Create doc record if first visit (subject_id=None as it's global Library)
        if current_page == 0:
            db = SessionLocal()
            try:
                # Hash = sample_text + file_size + basename — must match IngestionAgent.create_document_record
                sample_text = _ingestion().load_page_text(file_path, 0)[:10000]
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                content_hash = _ingestion().get_content_hash(sample_text + str(file_size) + os.path.basename(file_path))
                
                existing_doc = db.query(DBDocument).filter(DBDocument.content_hash == content_hash).first()
                if existing_doc:
                    doc_id = str(existing_doc.id)  # C16: capture as primitive string
                    db.expunge(existing_doc)        # C16: detach before session close
                else:
                    # Fix path separator for Windows
                    clean_filename = os.path.basename(file_path)
                    new_doc = DBDocument(id=doc_id, filename=clean_filename, title=clean_filename, content_hash=content_hash)
                    db.add(new_doc)
                    db.commit()
                    db.refresh(new_doc)
            finally:
                db.close()

        logger.info(f"--- LIBRARY INDEXING: Page {current_page + 1}/{total_pages} ---")
        page_text = _ingestion().load_page_text(file_path, current_page)
        temp_chunks = _ingestion().text_splitter.split_text(page_text)

        # Render and cache the page image for source-snippet display in Mentor Review
        if file_path and file_path.lower().endswith(".pdf"):
            cache_dir = "page_cache"
            os.makedirs(cache_dir, exist_ok=True)
            img_path = os.path.join(cache_dir, f"{doc_id}_p{current_page:04d}.png")
            if not os.path.exists(img_path):
                try:
                    import fitz as _fitz
                    with _fitz.open(file_path) as _doc:
                        pix = _doc.load_page(current_page).get_pixmap(dpi=120)
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
        matched_subtopic_ids = state.get("matched_subtopic_ids")  # set by node_match_topics

        db = SessionLocal()
        try:
            query = db.query(ContentChunk).join(Subtopic).join(Topic).filter(
                Topic.document_id == doc_id
            )

            if matched_subtopic_ids is None:
                # No topic filter — process all chunks for this document
                pass
            elif matched_subtopic_ids:
                # Filter to semantically matched subtopics
                query = query.filter(ContentChunk.subtopic_id.in_(matched_subtopic_ids))
            else:
                # TopicMatcher ran but found no matches — nothing to process
                logger.warning("TopicMatcher found no matching subtopics — returning empty chunk list.")
                return {
                    "chunks": [],
                    "current_chunk_index": 0,
                    "status_message": "No chunks matched the requested topics.",
                }

            db_chunks = query.all()
            logger.info(f"Found {len(db_chunks)} candidate chunk(s) for generation.")

            question_type = state.get("question_type", "active_recall")
            chunks_to_process = []
            for c in db_chunks:
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
                    continue
                chunks_to_process.append({
                    "id": c.id,
                    "text": c.text,
                    "subtopic_id": c.subtopic_id,
                })

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

    # Read up to 10 pages (or total_pages if already set) to sample document content
    max_pages = state.get("total_pages") or _ingestion().get_page_count(file_path)
    text_parts: List[str] = []
    for p in range(min(max_pages, 10)):
        text_parts.append(_ingestion().load_page_text(file_path, p))
        if sum(len(t) for t in text_parts) >= 15000:
            break
    full_text = "\n\n".join(text_parts)

    try:
        result = _curator().curate_structure(full_text)
    except Exception as e:
        logger.warning(
            "CuratorAgent failed (%s) — falling back to per-chunk LLM topic assignment", e
        )
        return {"subtopic_embeddings": [], "status_message": f"Hierarchy extraction failed: {e}"}

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
        return {"subtopic_embeddings": [], "status_message": f"Hierarchy persist failed: {e}"}
    finally:
        db.close()

    if not subtopic_meta:
        logger.warning("CuratorAgent returned empty hierarchy — falling back to per-chunk LLM.")
        return {"subtopic_embeddings": [], "hierarchy": hierarchy, "status_message": "Empty hierarchy."}

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
    pending = list(state.get("pending_qdrant_docs", []))
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
            "pending_qdrant_docs": pending,
            "status_message": f"Indexed chunk to subtopic id={assigned_subtopic_id}",
        }
    except Exception as e:
        db.rollback()
        logger.error("Error persisting chunk %d for doc %s: %s", idx, doc_id, e, exc_info=True)
        return {
            "hierarchy": hierarchy,
            "pending_qdrant_docs": pending,
            "status_message": f"Warning: failed to index chunk {idx} — {e}",
        }
    finally:
        db.close()


def _flush_qdrant_batch(pending_docs: List[Dict[str, Any]]):
    """Upserts a batch of pending docs to Qdrant in a single call."""
    if not pending_docs:
        return
    from langchain_core.documents import Document as LCDoc
    from langchain_qdrant import QdrantVectorStore
    from core.config import settings
    lc_docs = [LCDoc(page_content=d["text"], metadata=d["metadata"]) for d in pending_docs]
    try:
        QdrantVectorStore.from_documents(
            lc_docs,
            _ingestion().embeddings,
            url=settings.QDRANT_URL,
            collection_name=_ingestion().collection_name,
        )
    except Exception as e:
        logger.error(f"Qdrant batch flush failed ({len(pending_docs)} docs): {e}", exc_info=True)
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
    initial_status = "approved" if settings.AUTO_ACCEPT_CONTENT else "pending"

    drafts = _socratic().generate_flashcard(
        source_text=chunk_data["text"],
        question_type=question_type,
    )

    new_cards = []
    if drafts:
        fc_repo = FlashcardRepo()
        for draft in drafts:
            saved = fc_repo.create(
                subject_id=subject_id,
                subtopic_id=chunk_data.get("subtopic_id"),
                chunk_id=chunk_data["id"],
                question=draft.question,
                answer=draft.answer,
                question_type=draft.question_type,
                rubric_json=draft.rubric_json,
                status=initial_status,
            )
            new_cards.append({
                "flashcard_id": saved["id"],
                "question": saved["question"],
                "answer": saved["answer"],
                "question_type": saved["question_type"],
                "suggested_complexity": draft.suggested_complexity,
            })

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
    """Flushes the Qdrant batch for the completed page, then advances."""
    _flush_qdrant_batch(state.get("pending_qdrant_docs", []))
    return {"current_page": state["current_page"] + 1, "pending_qdrant_docs": []}


def node_flush_qdrant(state: GraphState):
    """Flushes the final Qdrant batch at the end of INDEXING (last page)."""
    _flush_qdrant_batch(state.get("pending_qdrant_docs", []))
    return {"pending_qdrant_docs": [], "status_message": "Document indexing complete."}

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
workflow.add_node("flush_qdrant", node_flush_qdrant)

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
    if idx < len(chunks):
        return "assign_topic" if state["mode"] == "INDEXING" else "generate"
    if state["mode"] == "INDEXING":
        if state["current_page"] < state["total_pages"] - 1:
            return "next_page"
        return "flush_qdrant"  # last page: flush batch before END
    return END


workflow.add_conditional_edges(
    "increment", router_after_increment,
    {"assign_topic": "assign_topic", "generate": "generate",
     "next_page": "next_page", "flush_qdrant": "flush_qdrant", END: END},
)

workflow.add_edge("next_page", "ingest")
workflow.add_edge("flush_qdrant", END)

phase1_graph = workflow.compile()
