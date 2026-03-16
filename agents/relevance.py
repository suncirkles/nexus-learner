"""
agents/relevance.py
-------------------
Agent for determining the relevance of content chunks to specific topics.
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from core.models import call_structured
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class RelevanceScore(BaseModel):
    is_relevant: bool = Field(description="Whether the text chunk is relevant to the target topics.")
    matched_topic: Optional[str] = Field(description="The topic that matched, if any.")
    reasoning: str = Field(description="Brief reasoning for the decision.")

class RelevanceAgent:
    def __init__(self):
        pass

    def check_relevance(self, chunk_text: str, target_topics: List[str]) -> RelevanceScore:
        """Determines if the chunk_text is relevant to any of the target_topics."""
        if not target_topics:
            return RelevanceScore(is_relevant=True, matched_topic=None, reasoning="No target topics specified.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert educational content filter. 
Your task is to determine if a given text chunk is highly relevant to any of the specified target topics.

Target Topics: {topics}

Rules:
1. If the text provides definitions, explanations, technical details, or useful educational context related to the topics, mark it as relevant.
2. Even if the text is a code snippet, list of terms, or brief mention, mark as relevant if it relates to a target topic.
3. Only mark as NOT relevant if the content is completely unrelated (e.g., generic footer, page number, blank page).
4. Be extremely inclusive; when in doubt, mark as relevant to ensure the learner doesn't miss potential study material."""),
            ("user", "Text Chunk: {text}")
        ])

        try:
            result = call_structured(
                RelevanceScore,
                prompt.format(topics=", ".join(target_topics), text=chunk_text[:2000]),
                purpose="routing",
            )
            if result is None:
                logger.warning("call_structured returned None for relevance check — defaulting to relevant")
                return RelevanceScore(is_relevant=True, matched_topic=None, reasoning="Quota exhausted — defaulting to relevant")
            return result
        except Exception as e:
            logger.error(f"Error checking relevance: {e}")
            # Fallback to true if LLM fails to avoid losing data
            return RelevanceScore(is_relevant=True, matched_topic=None, reasoning=f"Error: {e}")
