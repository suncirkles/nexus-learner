"""
agents/topic_assigner.py
-------------------------
Responsibility: Map each text chunk to exactly one existing subtopic, or propose
a new descriptive subtopic name when no match exists. Enforces the "No General
Content" policy — every chunk must belong to a specific subtopic.

Do Not:
- Build or restructure the topic hierarchy (CuratorAgent's job).
- Match user study-intent topics to indexed subtopics (TopicMatcherAgent).
- Filter chunks by relevance to a study session (RelevanceAgent).
- Generate flashcards or evaluate content (SocraticAgent / CriticAgent).
- Rename or merge existing subtopics; only select from the provided list or
  propose a net-new name — never modify what already exists.
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from core.models import get_llm, invoke_with_retry
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class TopicAssignment(BaseModel):
    topic_name: str = Field(description="The broad category name (e.g., 'Spark Basics').")
    subtopic_name: str = Field(description="The specific subtopic name (e.g., 'RDD Creation').")
    reasoning: str = Field(description="Brief reasoning for why this chunk belongs here.")

class TopicAssignerAgent:
    def __init__(self):
        pass

    def assign_topic(self, chunk_text: str, existing_hierarchy: List[dict] = None) -> TopicAssignment:
        """
        Assigns a topic and subtopic to a chunk of text.
        
        Args:
            chunk_text: The text to categorize.
            existing_hierarchy: Optional list of already discovered topics/subtopics 
                               to encourage reuse and structure.
        """
        hierarchy_str = ""
        if existing_hierarchy:
            hierarchy_str = "Existing Context (Reuse these if they fit):\n"
            for item in existing_hierarchy:
                subtopics = ", ".join(item.get("subtopics", []))
                hierarchy_str += f"- Topic: {item['topic']} (Subtopics: {subtopics})\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert academic organizer. Your task is to assign a specific Topic and Subtopic name to a chunk of text from a study document.

RULES:
1. NO 'General Content' or 'Uncategorized'. Every chunk must be given a descriptive name.
2. If the chunk is introductory, use topics like 'Introduction', 'Overview', or similar.
3. If the chunk is too short or purely structural (e.g. 'Chapter 1'), use the most relevant surrounding context.
4. Aim for consistency. If you've seen similar content, use similar names.
5. Topics should be broad (e.g. 'Machine Learning'), Subtopics should be specific (e.g. 'Backpropagation').

{hierarchy_context}
"""),
            ("user", "Text Chunk: {text}")
        ])

        from core.context import get_langchain_config
        from scripts.model_hop import is_quota_error
        last_exc = None
        for _attempt in range(2):  # retry once with next model on daily quota exhaustion
            try:
                llm = get_llm(purpose="routing", temperature=0)
                chain = prompt | llm.with_structured_output(TopicAssignment)
                result = invoke_with_retry(
                    chain.invoke,
                    {"hierarchy_context": hierarchy_str, "text": chunk_text[:3000]},
                    config=get_langchain_config()
                )
                if result is None:
                    raise RuntimeError("LLM returned None for topic assignment")
                result.topic_name = result.topic_name.strip()
                result.subtopic_name = result.subtopic_name.strip()
                return result
            except Exception as e:
                last_exc = e
                if is_quota_error(e) and _attempt == 0:
                    continue  # model just blacklisted; next iteration picks the next one
                break
        # H19: do NOT fall back to "General Overview"/"Introduction" — that violates
        # the No General Content policy and pollutes the topic hierarchy.
        # Re-raise so node_assign_topic can log and skip this chunk cleanly.
        logger.error("TopicAssigner failed for chunk (first 100 chars: %r): %s", chunk_text[:100], last_exc)
        raise last_exc
