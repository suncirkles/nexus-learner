"""
agents/curator.py
------------------
Content structuring agent. Analyzes raw text to produce a hierarchical
Topic → Subtopic structure using LLM-powered extraction. Merges new
content into existing Subject hierarchies, avoiding duplicate topics.
Persists the hierarchy to the relational database.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from core.models import get_llm, call_structured_chain
from core.database import SessionLocal, Topic, Subtopic
from core.config import settings

logger = logging.getLogger(__name__)

class SubtopicStructure(BaseModel):
    name: str = Field(description="Name of the sub-topic or lesson.")
    summary: str = Field(description="A 1-sentence summary of what this sub-topic covers.")

class TopicStructure(BaseModel):
    name: str = Field(description="Name of the broad topic or chapter.")
    summary: str = Field(description="A concise summary of the topic.")
    subtopics: List[SubtopicStructure] = Field(description="List of sub-topics within this topic.")

class DocumentStructure(BaseModel):
    summary: str = Field(description="An overall summary of the document.")
    topics: List[TopicStructure] = Field(description="Hierarchical list of topics and their sub-topics.")

_CURATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Educational Content Curator.
            Your task is to analyze the provided educational material and organize it into a logical hierarchy of Topics and Sub-topics.

            IMPORTANT: You are adding content to an existing Subject. Below is the current hierarchy of Topics and Sub-topics for this Subject.

            Current Hierarchy:
            {existing_structure}

            Instructions:
            1. Analyze the new content provided.
            2. Match the new content to the EXISTING Topics and Sub-topics where possible.
            3. If the content covers a NEW broad area, create a new Topic.
            4. If the content covers a NEW lesson within an existing Topic, add a new Sub-topic to that topic.
            5. Provide a concise summary for the new artifact, and update/provide summaries for topics and sub-topics.
            6. Return the FULL hierarchy for the new content (including matching existing ones).
            """),
    ("human", "Analyze the following content and integrate it into the subject hierarchy:\n\n{content}")
])


class CuratorAgent:
    def __init__(self):
        self._chain = self._build_chain()

    def _build_chain(self):
        return _CURATOR_PROMPT | get_llm(purpose="primary").with_structured_output(DocumentStructure)

    def curate_structure(self, subject_id: int, doc_id: str, full_text: str):
        """Analyzes text and merges it into the Subject's topic hierarchy."""
        db = SessionLocal()
        existing_structure_text = "No existing topics."
        try:
            from core.database import SubjectDocumentAssociation
            # Get existing topics for this subject via all associated documents
            existing_topics = db.query(Topic).join(SubjectDocumentAssociation, Topic.document_id == SubjectDocumentAssociation.document_id).\
                filter(SubjectDocumentAssociation.subject_id == subject_id).all()
            
            if existing_topics:
                lines = []
                for t in existing_topics:
                    subs = db.query(Subtopic).filter(Subtopic.topic_id == t.id).all()
                    sub_names = ", ".join([s.name for s in subs])
                    lines.append(f"- Topic: {t.name} (Sub-topics: {sub_names})")
                existing_structure_text = "\n".join(lines)
            # C11 (defensive): existing_topics ORM objects are no longer accessed beyond this
            # point — all data has been converted to plain strings in existing_structure_text.
            # Do NOT access existing_topics after this line to avoid DetachedInstanceError if
            # the session state changes during the long LLM call below.

            analysis_text = full_text[:15000]
            # Rebuild chain per-call when hopping so provider selection is fresh
            if settings.MODEL_HOP_ENABLED:
                self._chain = self._build_chain()
            structure = call_structured_chain(
                self._chain,
                DocumentStructure,
                {"content": analysis_text, "existing_structure": existing_structure_text},
            )
            
            # H13: deduplicate topics/subtopics within the LLM response before DB writes.
            # The model occasionally returns the same name twice; merging here prevents
            # duplicate DB records even if the ilike check misses a flush-timing edge case.
            seen_topic_keys: dict = {}
            deduped_topics = []
            for t in structure.topics:
                key = t.name.strip().lower()
                if key not in seen_topic_keys:
                    seen_topic_keys[key] = t
                    seen_sub_keys: set = set()
                    unique_subs = []
                    for s in t.subtopics:
                        sub_key = s.name.strip().lower()
                        if sub_key not in seen_sub_keys:
                            seen_sub_keys.add(sub_key)
                            unique_subs.append(s)
                    t.subtopics = unique_subs
                    deduped_topics.append(t)
                else:
                    existing = seen_topic_keys[key]
                    existing_sub_keys = {s.name.strip().lower() for s in existing.subtopics}
                    for s in t.subtopics:
                        if s.name.strip().lower() not in existing_sub_keys:
                            existing.subtopics.append(s)
                            existing_sub_keys.add(s.name.strip().lower())
                    logger.warning("Duplicate topic '%s' in LLM response — subtopics merged", t.name)

            topics_data = []
            for t in deduped_topics:
                db_topic = db.query(Topic).filter(
                    Topic.document_id == doc_id, Topic.name.ilike(t.name)
                ).first()
                
                if not db_topic:
                    db_topic = Topic(
                        document_id=doc_id,
                        name=t.name,
                        summary=t.summary
                    )
                    db.add(db_topic)
                    db.flush()
                
                subtopics_list = []
                for s in t.subtopics:
                    # Check for existing subtopic within this topic
                    db_subtopic = db.query(Subtopic).filter(
                        Subtopic.topic_id == db_topic.id,
                        Subtopic.name.ilike(s.name)
                    ).first()
                    
                    if not db_subtopic:
                        db_subtopic = Subtopic(
                            topic_id=db_topic.id,
                            name=s.name,
                            summary=s.summary
                        )
                        db.add(db_subtopic)
                        db.flush()
                    
                    subtopics_list.append({
                        "id": db_subtopic.id,
                        "name": db_subtopic.name,
                        "summary": db_subtopic.summary
                    })
                
                topics_data.append({
                    "id": db_topic.id,
                    "name": db_topic.name,
                    "summary": db_topic.summary,
                    "subtopics": subtopics_list
                })
            
            db.commit()
            return {
                "doc_summary": structure.summary,
                "hierarchy": topics_data
            }
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
