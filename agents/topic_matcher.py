"""
agents/topic_matcher.py
-------------------------
Responsibility: Semantically match user-provided study topic strings against
the pre-indexed subtopics for a subject and return the best-matching subtopic IDs.

Do Not:
- Create, rename, or restructure topics (CuratorAgent / TopicAssignerAgent).
- Score the relevance of individual content chunks (RelevanceAgent's job).
- Generate flashcards or evaluate card quality (SocraticAgent / CriticAgent).
- Fetch or embed any document content (IngestionAgent).
- Reject a user topic outright; always return the closest match even if approximate,
  and let the RelevanceAgent filter weak results downstream.
"""

import logging
from typing import List
from pydantic import BaseModel, Field
from core.models import call_structured
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class TopicMatch(BaseModel):
    provided_topic: str = Field(description="The user's input topic name.")
    matched_subtopic_ids: List[int] = Field(description="List of DB Subtopic IDs that are semantically relevant.")
    reasoning: str = Field(description="Brief reasoning for the match.")

class TopicMatcherAgent:
    def __init__(self):
        pass

    def match_topics(self, user_topics: List[str], indexed_subtopics: List[dict]) -> List[TopicMatch]:
        """
        Matches a list of user topics to indexed subtopics.
        
        indexed_subtopics should be a list of dicts: {"id": int, "name": str, "topic_name": str}
        """
        if not user_topics or not indexed_subtopics:
            return []

        subtopic_context = "\n".join([
            f"ID: {s['id']} | Subtopic: {s['name']} (Parent Topic: {s['topic_name']})"
            for s in indexed_subtopics
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert librarian. Your task is to match a USER'S REQUESTED TOPIC to one or more PRE-INDEXED SUBTOPICS from a document.

RULES:
1. Be inclusive: if the user's topic is broad (e.g. 'Spark'), match all relevant Spark subtopics.
2. Use semantic similarity: 'Lazy Evaluation' matches 'Spark Optimizations'.
3. Only return matches that are TRULY relevant.
4. If no match is found, return an empty list for matched_subtopic_ids.

PRE-INDEXED SUBTOPICS:
{context}
"""),
            ("user", "User's Topic: {user_topic}")
        ])

        results = []
        for ut in user_topics:
            try:
                match = call_structured(
                    TopicMatch,
                    prompt.format(context=subtopic_context, user_topic=ut),
                    purpose="routing",
                )
                if match is not None:
                    results.append(match)
            except Exception as e:
                logger.error(f"Error matching topic '{ut}': {e}")
        
        return results
