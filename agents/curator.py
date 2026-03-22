"""
agents/curator.py
------------------
Responsibility: Analyse raw text and extract a Topic → Subtopic hierarchy.
Merges new topics into the existing subject structure passed in by the caller.
Returns plain data — no DB reads or writes (caller persists via topic_repo).

Do Not:
- Assign individual chunks to subtopics (TopicAssignerAgent's job).
- Generate flashcards or evaluate card quality (SocraticAgent / CriticAgent).
- Embed or store any text content (IngestionAgent).
- Query or write the database directly; receive the existing hierarchy as input
  and return the new structure for the caller to persist.
- Invent topics that have no basis in the provided text; extraction only.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from core.models import get_llm, call_structured_chain
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

    def curate_structure(
        self,
        full_text: str,
        existing_structure_text: str = "No existing topics.",
    ) -> Dict[str, Any]:
        """Analyzes text and returns a topic hierarchy — no DB writes.

        Args:
            full_text: The document content to analyze (truncated to 15 000 chars internally).
            existing_structure_text: Pre-formatted string describing the current topic
                hierarchy for this subject (pre-queried by the calling workflow node).

        Returns:
            {
                "doc_summary": str,
                "hierarchy": [
                    {
                        "name": str,
                        "summary": str,
                        "subtopics": [{"name": str, "summary": str}, ...]
                    },
                    ...
                ]
            }
        The caller is responsible for persisting topics/subtopics via topic_repo.
        """
        analysis_text = full_text[:15000]
        # Rebuild chain per-call when hopping so provider selection is fresh
        if settings.MODEL_HOP_ENABLED:
            self._chain = self._build_chain()
        structure = call_structured_chain(
            self._chain,
            DocumentStructure,
            {"content": analysis_text, "existing_structure": existing_structure_text},
        )

        # H13: deduplicate topics/subtopics within the LLM response before returning.
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

        hierarchy = [
            {
                "name": t.name,
                "summary": t.summary,
                "subtopics": [{"name": s.name, "summary": s.summary} for s in t.subtopics],
            }
            for t in deduped_topics
        ]

        return {
            "doc_summary": structure.summary,
            "hierarchy": hierarchy,
        }
