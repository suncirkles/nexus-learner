"""
workflows/phase2_web_ingestion.py
-----------------------------------
LangGraph workflow for Phase 2: Web Content Ingestion.

Mirrors phase1_ingestion.py but sources content from trusted websites
rather than user-uploaded files.

Graph:  safety_check -> research -> ingest_web_document
          -> curate -> generate -> critic
          -> [increment_chunk | next_document | END]

Existing CuratorAgent, SocraticAgent and CriticAgent are reused
completely unchanged — the web workflow simply feeds them different
document/chunk records.
"""

import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document as LCDocument
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from agents.curator import CuratorAgent
from agents.critic import CriticAgent
from agents.safety import SafetyAgent
from agents.socratic import SocraticAgent
from agents.web_researcher import WebResearchAgent
from core.config import settings
from core.database import SessionLocal, ContentChunk, Document as DBDocument, SubjectDocumentAssociation, Topic, Subtopic
from repositories.sql.topic_repo import TopicRepo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    subject_id: int
    topics: List[str]
    subject_name: str
    web_documents: List[dict]       # WebDocument dicts from WebResearchAgent
    current_doc_index: int          # Which web document is being processed
    doc_id: str                     # UUID for current document
    full_text: str                  # Content of current web document
    chunks: List[Any]               # LangChain Document objects for current doc
    hierarchy: List[Dict]           # Topic/Subtopic structure
    doc_summary: str
    current_chunk_index: int
    generated_flashcards: List[Dict]
    status_message: str
    safety_blocked: bool
    safety_reason: str
    processed_urls: List[str]       # Deduplication tracking across documents
    status_callback: Optional[Any]  # Optional callable(str) for UI progress
    stop_event: Optional[Any]       # Optional threading.Event; nodes check this to halt early


# ---------------------------------------------------------------------------
# Agent singletons (lazy init avoids import-time LLM instantiation errors)
# ---------------------------------------------------------------------------
_safety_agent: Optional[SafetyAgent] = None
_web_researcher: Optional[WebResearchAgent] = None
_curator_agent: Optional[CuratorAgent] = None
_socratic_agent: Optional[SocraticAgent] = None
_critic_agent: Optional[CriticAgent] = None
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
_embeddings: Optional[OpenAIEmbeddings] = None


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
    return _embeddings


def _get_safety() -> SafetyAgent:
    global _safety_agent
    if _safety_agent is None:
        _safety_agent = SafetyAgent()
    return _safety_agent


def _get_researcher() -> WebResearchAgent:
    global _web_researcher
    if _web_researcher is None:
        _web_researcher = WebResearchAgent()
    return _web_researcher


def _get_curator() -> CuratorAgent:
    global _curator_agent
    if _curator_agent is None:
        _curator_agent = CuratorAgent()
    return _curator_agent


def _get_socratic() -> SocraticAgent:
    global _socratic_agent
    if _socratic_agent is None:
        _socratic_agent = SocraticAgent()
    return _socratic_agent


def _get_critic() -> CriticAgent:
    global _critic_agent
    if _critic_agent is None:
        _critic_agent = CriticAgent()
    return _critic_agent


# ---------------------------------------------------------------------------
# Node: safety_check
# ---------------------------------------------------------------------------

def node_safety_check(state: GraphState) -> dict:
    """Run subject-level safety screening.  Blocks the entire pipeline on failure."""
    logger.info("--- [Phase 2] SAFETY CHECK: %s ---", state["subject_name"])
    cb = state.get("status_callback")
    if cb:
        cb(f"Running safety check for subject: '{state['subject_name']}'")

    result = _get_safety().check_subject_safety(state["subject_name"])
    if not result.is_safe:
        logger.warning("Subject blocked by safety check: %s", result.reason)
        if cb:
            cb(f"Subject blocked: {result.reason}")
        return {
            "safety_blocked": True,
            "safety_reason": result.reason,
            "status_message": f"Safety check blocked: {result.reason}",
        }

    if cb:
        cb(f"Subject safety check passed.")
    return {
        "safety_blocked": False,
        "safety_reason": "",
        "status_message": "Safety check passed.",
    }


def _route_after_safety(state: GraphState) -> str:
    return "end" if state.get("safety_blocked") else "research"


# ---------------------------------------------------------------------------
# Node: research
# ---------------------------------------------------------------------------

def node_research(state: GraphState) -> dict:
    """Use WebResearchAgent to gather pages for every topic."""
    logger.info("--- [Phase 2] RESEARCH: %d topics ---", len(state["topics"]))
    cb = state.get("status_callback")

    def _cb_wrapper(msg: str):
        logger.info("[Research] %s", msg)
        if cb:
            cb(msg)

    docs = _get_researcher().research_topics(
        topics=state["topics"],
        subject_name=state["subject_name"],
        subject_id=state["subject_id"],
        status_callback=_cb_wrapper,
        stop_event=state.get("stop_event"),
    )

    if not docs:
        return {
            "web_documents": [],
            "current_doc_index": 0,
            "status_message": "No content found on trusted sources for the given topics.",
        }

    return {
        "web_documents": [d.model_dump() for d in docs],
        "current_doc_index": 0,
        "status_message": f"Research complete. Found {len(docs)} pages.",
    }


def _route_after_research(state: GraphState) -> str:
    return "end" if not state.get("web_documents") else "ingest_web_document"


# ---------------------------------------------------------------------------
# Node: ingest_web_document
# ---------------------------------------------------------------------------

def node_ingest_web_document(state: GraphState) -> dict:
    """Persist the current web document to DB and Qdrant, then chunk it."""
    idx = state["current_doc_index"]
    doc_data = state["web_documents"][idx]

    url = doc_data["url"]
    title = doc_data["title"]
    content = doc_data["content"]
    content_hash = doc_data["content_hash"]
    subject_id = state["subject_id"]

    logger.info("--- [Phase 2] INGEST WEB DOC %d/%d: %s ---", idx + 1, len(state["web_documents"]), url)
    cb = state.get("status_callback")
    if cb:
        cb(f"Ingesting: {doc_data['domain']} — \"{title}\"")

    doc_id = str(uuid.uuid4())

    # --- Duplicate detection + document creation (single atomic commit) ---
    db = SessionLocal()
    try:
        from core.database import SubjectDocumentAssociation
        existing = db.query(DBDocument).filter(DBDocument.content_hash == content_hash).first()

        if existing:
            # C19: read id into plain string before session closes
            existing_doc_id = str(existing.id)
            logger.info("Duplicate content found (previously ingested as '%s').", existing.filename)
            existing_assoc = db.query(SubjectDocumentAssociation).filter(
                SubjectDocumentAssociation.subject_id == subject_id,
                SubjectDocumentAssociation.document_id == existing_doc_id,
            ).first()
            if not existing_assoc:
                db.add(SubjectDocumentAssociation(subject_id=subject_id, document_id=existing_doc_id))
                db.commit()
                logger.info("Added new association for existing document to subject %d", subject_id)

            if cb:
                cb(f"Re-using existing content: {url}")

            return {
                "doc_id": existing_doc_id,
                "full_text": "",
                "chunks": [],
                "hierarchy": state.get("hierarchy", []),
                "doc_summary": state.get("doc_summary", ""),
                "current_chunk_index": 0,
                "generated_flashcards": state.get("generated_flashcards", []),
                "status_message": f"Using existing content: {title}",
            }

        # C21: create document + association in one commit so neither is orphaned
        new_doc = DBDocument(
            id=doc_id,
            filename=title,
            title=title,
            content_hash=content_hash,
            source_type="web",
            source_url=url,
        )
        db.add(new_doc)
        db.add(SubjectDocumentAssociation(subject_id=subject_id, document_id=doc_id))
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("DB error persisting web document: %s", exc)
        raise
    finally:
        db.close()

    # --- Chunking ---
    raw_chunks = _text_splitter.split_text(content)

    # Save chunks to relational DB
    db = SessionLocal()
    saved_chunk_ids: List[int] = []
    try:
        for chunk_text in raw_chunks:
            db_chunk = ContentChunk(
                document_id=doc_id,
                text=chunk_text,
                source_type="web",
                source_url=url,
            )
            db.add(db_chunk)
            db.flush()   # assigns db_chunk.id without committing
            saved_chunk_ids.append(db_chunk.id)
        db.commit()      # single atomic commit for all chunks
    except Exception as exc:
        db.rollback()
        logger.error("DB error saving web chunks: %s", exc)
        raise
    finally:
        db.close()

    # --- Build LangChain Document objects ---
    lc_documents: List[LCDocument] = []
    for i, chunk_text in enumerate(raw_chunks):
        lc_doc = LCDocument(
            page_content=chunk_text,
            metadata={
                "document_id": doc_id,
                "db_chunk_id": saved_chunk_ids[i],
                "source_url": url,
                "source_type": "web",
            },
        )
        lc_documents.append(lc_doc)

    # --- Embed into Qdrant ---
    # C20: if Qdrant fails, log clearly so caller knows chunks exist in SQLite but have no vectors
    try:
        QdrantVectorStore.from_documents(
            lc_documents,
            _get_embeddings(),
            url=settings.QDRANT_URL,
            collection_name=settings.QDRANT_COLLECTION_NAME,
        )
    except Exception as exc:
        logger.error(
            "Qdrant embedding failed for '%s' (%d chunks persisted in SQLite but NOT vectorised): %s",
            url, len(lc_documents), exc,
        )

    tracked_urls = list(state.get("processed_urls") or [])
    tracked_urls.append(url)

    if cb:
        cb(f"Chunked into {len(lc_documents)} sections. Analysing structure...")

    return {
        "doc_id": doc_id,
        "full_text": content,
        "chunks": lc_documents,
        "hierarchy": [],          # Reset — curator will rebuild for this document
        "doc_summary": "",
        "current_chunk_index": 0,
        "generated_flashcards": [],
        "processed_urls": tracked_urls,
        "status_message": f"Ingested: {title} ({len(lc_documents)} chunks)",
    }


# ---------------------------------------------------------------------------
# Node: curate (delegates to CuratorAgent unchanged)
# ---------------------------------------------------------------------------

def node_curate(state: GraphState) -> dict:
    """[Phase 2] Extract topic hierarchy via LLM, then persist via TopicRepo.

    Phase 2b: CuratorAgent.curate_structure() is now pure LLM I/O — no DB writes.
    This node pre-queries the existing hierarchy as a formatted string, calls the
    agent, then calls topic_repo.get_or_create() / get_or_create_subtopic() for
    each topic/subtopic in the LLM result.
    """
    if not state.get("full_text"):
        # Duplicate was skipped — nothing to curate
        return {}

    logger.info("--- [Phase 2] CURATE ---")
    cb = state.get("status_callback")
    if cb:
        cb("Extracting topics and subtopics...")

    subject_id = state["subject_id"]
    doc_id = state["doc_id"]

    # Pre-query existing hierarchy for context (workflow responsibility, Phase 3 moves to repo)
    existing_structure_text = "No existing topics."
    db = SessionLocal()
    try:
        existing_topics = (
            db.query(Topic)
            .join(SubjectDocumentAssociation, Topic.document_id == SubjectDocumentAssociation.document_id)
            .filter(SubjectDocumentAssociation.subject_id == subject_id)
            .all()
        )
        if existing_topics:
            lines = []
            for t in existing_topics:
                subs = db.query(Subtopic).filter(Subtopic.topic_id == t.id).all()
                sub_names = ", ".join(s.name for s in subs)
                lines.append(f"- Topic: {t.name} (Sub-topics: {sub_names})")
            existing_structure_text = "\n".join(lines)
    finally:
        db.close()

    # LLM call — pure, no DB side effects
    result = _get_curator().curate_structure(
        full_text=state["full_text"],
        existing_structure_text=existing_structure_text,
    )

    # Persist via TopicRepo (get_or_create is idempotent — case-insensitive)
    topic_repo = TopicRepo()
    topics_data = []
    for t in result["hierarchy"]:
        topic = topic_repo.get_or_create(doc_id, t["name"], t.get("summary", ""))
        subtopics_list = []
        for s in t["subtopics"]:
            sub = topic_repo.get_or_create_subtopic(topic["id"], s["name"], s.get("summary", ""))
            subtopics_list.append(sub)
        topics_data.append({**topic, "subtopics": subtopics_list})

    return {
        "hierarchy": topics_data,
        "doc_summary": result["doc_summary"],
        "status_message": f"Integrated content into {len(topics_data)} topics.",
    }


# ---------------------------------------------------------------------------
# Node: generate (delegates to SocraticAgent unchanged)
# ---------------------------------------------------------------------------

def node_generate(state: GraphState) -> dict:
    if not state.get("chunks"):
        return {}

    idx = state["current_chunk_index"]
    chunk = state["chunks"][idx]
    logger.info("--- [Phase 2] GENERATE chunk %d/%d ---", idx + 1, len(state["chunks"]))

    cb = state.get("status_callback")

    # Classify chunk into a subtopic (same logic as phase1)
    subtopic_id = None
    subtopic_name = "General"

    if state.get("hierarchy"):
        from core.models import get_llm
        from langchain_core.prompts import ChatPromptTemplate
        from pydantic import BaseModel, Field

        class Classification(BaseModel):
            subtopic_id: int = Field(description="The ID of the most relevant subtopic.")

        classifier = get_llm(purpose="routing").with_structured_output(Classification)

        all_subs = []
        for t in state["hierarchy"]:
            for s in t["subtopics"]:
                all_subs.append(f"ID {s['id']}: {s['name']} ({s['summary']})")

        mapping_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a classifier. Given a text chunk and a list of subtopics, "
                "return the ID of the ONE subtopic it most closely belongs to.\n\nSubtopics:\n{subtopics}",
            ),
            ("user", "Chunk Text: {text}"),
        ])

        try:
            mapping_result = classifier.invoke(
                mapping_prompt.invoke({
                    "subtopics": "\n".join(all_subs),
                    "text": chunk.page_content[:1000],
                })
            )
            subtopic_id = mapping_result.subtopic_id
            for t in state["hierarchy"]:
                for s in t["subtopics"]:
                    if s["id"] == subtopic_id:
                        subtopic_name = s["name"]
                        break
        except Exception as exc:
            logger.warning("Subtopic classification failed: %s", exc)
            if state["hierarchy"] and state["hierarchy"][0]["subtopics"]:
                subtopic_id = state["hierarchy"][0]["subtopics"][0]["id"]
                subtopic_name = state["hierarchy"][0]["subtopics"][0]["name"]

    existing_flashcards = list(state.get("generated_flashcards") or [])
    flashcard = _get_socratic().generate_flashcard(state["doc_id"], chunk, subtopic_id=subtopic_id)

    if cb:
        fc_count = len(existing_flashcards) + (1 if flashcard.get("status") == "success" else 0)
        cb(f"Generating flashcards... ({fc_count} generated so far) [chunk {idx + 1}/{len(state['chunks'])}]")

    return {
        "generated_flashcards": existing_flashcards + [flashcard],
        "status_message": (
            f"Generating Q&A for Subtopic: '{subtopic_name}' ({idx + 1}/{len(state['chunks'])})"
        ),
    }


# ---------------------------------------------------------------------------
# Node: critic (delegates to CriticAgent unchanged)
# ---------------------------------------------------------------------------

def node_critic(state: GraphState) -> dict:
    if not state.get("chunks"):
        return {}

    flashcards = state.get("generated_flashcards") or []
    if not flashcards:
        return {}

    flashcard = flashcards[-1]
    idx = state["current_chunk_index"]
    chunk = state["chunks"][idx]

    logger.info("--- [Phase 2] CRITIC chunk %d ---", idx + 1)
    fc_id = flashcard.get("flashcard_id")
    if fc_id:
        _get_critic().evaluate_flashcard(
            flashcard_id=fc_id,
            source_text=chunk.page_content,
            question=flashcard.get("question", ""),
            answer=flashcard.get("answer", ""),
        )

    return {
        "status_message": f"Verified grounding for content {idx + 1}. Ready for review.",
    }


# ---------------------------------------------------------------------------
# Node: increment_chunk
# ---------------------------------------------------------------------------

def node_increment_chunk(state: GraphState) -> dict:
    return {"current_chunk_index": state["current_chunk_index"] + 1}


# ---------------------------------------------------------------------------
# Node: next_document
# ---------------------------------------------------------------------------

def node_next_document(state: GraphState) -> dict:
    return {
        "current_doc_index": state["current_doc_index"] + 1,
        "current_chunk_index": 0,
        "chunks": [],
        "hierarchy": [],
        "doc_summary": "",
        "generated_flashcards": [],
        "full_text": "",
        "doc_id": "",
    }


# ---------------------------------------------------------------------------
# Conditional routing after critic
# ---------------------------------------------------------------------------

def _route_after_critic(state: GraphState) -> str:
    chunks = state.get("chunks") or []
    web_docs = state.get("web_documents") or []
    idx = state["current_chunk_index"]
    doc_idx = state["current_doc_index"]

    if not chunks:
        # Duplicate was skipped — advance to next document or end
        if doc_idx < len(web_docs) - 1:
            return "next_document"
        return "end"

    if idx < len(chunks) - 1:
        return "increment_chunk"

    if doc_idx < len(web_docs) - 1:
        return "next_document"

    return "end"


# ---------------------------------------------------------------------------
# Build & compile the graph
# ---------------------------------------------------------------------------

_workflow = StateGraph(GraphState)

_workflow.add_node("safety_check", node_safety_check)
_workflow.add_node("research", node_research)
_workflow.add_node("ingest_web_document", node_ingest_web_document)
_workflow.add_node("curate", node_curate)
_workflow.add_node("generate", node_generate)
_workflow.add_node("critic", node_critic)
_workflow.add_node("increment_chunk", node_increment_chunk)
_workflow.add_node("next_document", node_next_document)

_workflow.add_edge(START, "safety_check")
_workflow.add_conditional_edges("safety_check", _route_after_safety, {"end": END, "research": "research"})
_workflow.add_conditional_edges(
    "research",
    _route_after_research,
    {"end": END, "ingest_web_document": "ingest_web_document"},
)
def _route_after_ingest(state: GraphState) -> str:
    """Skip curate/generate/critic entirely when a duplicate was detected."""
    if not state.get("full_text"):
        # Duplicate skipped — jump straight to routing logic
        doc_idx = state["current_doc_index"]
        web_docs = state.get("web_documents") or []
        if doc_idx < len(web_docs) - 1:
            return "next_document"
        return "end"
    return "curate"


_workflow.add_conditional_edges(
    "ingest_web_document",
    _route_after_ingest,
    {"curate": "curate", "next_document": "next_document", "end": END},
)
_workflow.add_edge("curate", "generate")
_workflow.add_edge("generate", "critic")
_workflow.add_conditional_edges(
    "critic",
    _route_after_critic,
    {
        "increment_chunk": "increment_chunk",
        "next_document": "next_document",
        "end": END,
    },
)
_workflow.add_edge("increment_chunk", "generate")
_workflow.add_edge("next_document", "ingest_web_document")

phase2_graph = _workflow.compile()
