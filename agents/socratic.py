"""
agents/socratic.py
-------------------
AI-powered flashcard generator supporting multiple question types.
Generates high-quality Q&A pairs from text chunks, saves them to the
database, and supports flashcard recreation based on mentor feedback.
Also provides LLM-suggested answers for the review workflow.
"""

import json
import logging
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from core.models import get_llm
from core.database import SessionLocal, Flashcard, ContentChunk
from core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class RubricItem(BaseModel):
    criterion: str = Field(description="Short name for the grading criterion.")
    description: str = Field(description="What a correct response must satisfy for this criterion.")

class FlashcardItem(BaseModel):
    question: str = Field(description="The question based on the source text.")
    answer: str = Field(description="The precise answer grounded in the source text.")
    question_type: str = Field(description="Type of question: active_recall | fill_blank | short_answer | long_answer | numerical | scenario")
    rubric: List[RubricItem] = Field(description="Exactly 3 grading criteria for evaluating a learner's response.")
    suggested_complexity: str = Field(description="Estimated complexity: simple | medium | complex")

class FlashcardOutput(BaseModel):
    flashcards: List[FlashcardItem] = Field(description="List of one or more flashcards (1-3).")

# ---------------------------------------------------------------------------
# Per-type system prompts
# ---------------------------------------------------------------------------

_BASE_RULES = """
CRITICAL GUIDELINES:
1. Ground every question and answer STRICTLY in the provided source text.
2. Generate BETWEEN 1 and 3 flashcards. Do not overlap questions.
3. If the text lacks significant testable content, return an empty list.
4. For each flashcard include EXACTLY 3 rubric criteria.
5. Cite the specific part of the source that supports the answer in the rubric.
6. suggested_complexity: simple (recall only), medium (synthesis/application), complex (multi-step reasoning).
"""

PROMPTS: Dict[str, str] = {
    "active_recall": f"""You are an expert educator generating Active Recall flashcards.
Focus on: essential concepts, core definitions, high-impact knowledge.
Questions should promote deep understanding and test whether the learner truly grasped the concept.
{_BASE_RULES}""",

    "fill_blank": f"""You are an expert educator generating Fill-in-the-Blank flashcards.
Focus on: identifying the single most critical term, formula, or phrase in a concept.
Format question as a sentence with '___' replacing the key term.
The answer is the exact term(s) that fill the blank.
{_BASE_RULES}""",

    "short_answer": f"""You are an expert educator generating Short Answer flashcards.
Focus on: synthesis across two ideas present in the source text.
The answer should be 2-3 sentences that connect the ideas concisely.
{_BASE_RULES}""",

    "long_answer": f"""You are an expert educator generating Long Answer flashcards.
Focus on: multi-part reasoning that requires a structured argument or step-by-step explanation.
The answer should outline the key steps/components a complete response must include.
{_BASE_RULES}""",

    "numerical": f"""You are an expert educator generating Numerical/Derivation flashcards.
Focus on: step-by-step derivations, calculations, or proofs drawn from the source.
Each step in the answer must be linked to a formula or statement in the source text.
{_BASE_RULES}""",

    "scenario": f"""You are an expert educator generating Scenario-based flashcards.
Focus on: "What happens if…" or "What would you do when…" problems derived from code snippets, processes, or case studies in the source.
The answer should explain the outcome or correct action with reference to the source.
{_BASE_RULES}""",
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SocraticAgent:
    def __init__(self):
        self.llm = get_llm(purpose="primary", temperature=0.3)
        # Build one chain per question type; cached at construction time.
        self._chains: Dict[str, Any] = {}
        for qtype, system_msg in PROMPTS.items():
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_msg),
                ("user", "Source text:\n\n{text}"),
            ])
            self._chains[qtype] = prompt | self.llm.with_structured_output(FlashcardOutput)

    def _get_chain(self, question_type: str):
        """Returns the chain for the given type, falling back to active_recall."""
        if question_type not in self._chains:
            logger.warning("Unknown question_type '%s', falling back to active_recall.", question_type)
            question_type = "active_recall"
        return self._chains[question_type]

    def generate_flashcard(
        self,
        doc_id: str,
        chunk,
        subtopic_id: int = None,
        subject_id: int = None,
        question_type: str = "active_recall",
    ) -> dict:
        """Generates flashcard(s) based on a chunk and stages them to the database.

        ``chunk`` may be either a LangChain ``Document`` (has ``page_content`` and
        ``metadata`` dict) or a SQLAlchemy ``ContentChunk`` ORM object (has ``text``
        and ``id``).
        """
        # Resolve source text and chunk_id regardless of chunk type
        if hasattr(chunk, 'page_content'):  # LangChain Document
            source_text = chunk.page_content
            chunk_id = chunk.metadata.get("db_chunk_id") if isinstance(getattr(chunk, 'metadata', None), dict) else None
        else:  # ORM ContentChunk
            source_text = chunk.text
            chunk_id = chunk.id

        if chunk_id is None:
            logger.warning(
                "generate_flashcard: chunk_id is None for doc %s — flashcard will lack source traceability",
                doc_id,
            )

        chain = self._get_chain(question_type)
        result = chain.invoke({"text": source_text})

        if not result.flashcards:
            return {"status": "skipped", "reason": "Insufficient information in chunk."}

        db = SessionLocal()
        saved_cards = []
        try:
            initial_status = "approved" if settings.AUTO_ACCEPT_CONTENT else "pending"
            for card in result.flashcards:
                rubric_json = json.dumps([r.model_dump() for r in card.rubric])
                fc = Flashcard(
                    chunk_id=chunk_id,
                    subtopic_id=subtopic_id,
                    subject_id=subject_id,
                    question=card.question,
                    answer=card.answer,
                    # Use the caller-requested type; LLM self-label can drift
                    question_type=question_type,
                    rubric=rubric_json,
                    complexity_level=None,  # confirmed by mentor
                    status=initial_status,
                )
                db.add(fc)
                db.commit()
                db.refresh(fc)
                saved_cards.append({
                    "flashcard_id": fc.id,
                    "question": fc.question,
                    "answer": fc.answer,
                    "question_type": fc.question_type,
                    "suggested_complexity": card.suggested_complexity,
                })

            return {"status": "success", "flashcards": saved_cards}
        finally:
            db.close()

    def recreate_flashcard(self, flashcard_id: int, feedback: str) -> dict:
        """Regenerates a flashcard based on mentor feedback and its original source."""
        db = SessionLocal()
        try:
            fc = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
            if not fc:
                return {"status": "error", "message": "Flashcard not found."}

            chunk = db.query(ContentChunk).filter(ContentChunk.id == fc.chunk_id).first()
            source_text = chunk.text if chunk else "Source text unavailable."
            question_type = fc.question_type or "active_recall"

            recreate_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""{PROMPTS.get(question_type, PROMPTS['active_recall'])}

You are REVISING an existing flashcard based on mentor feedback.
Address the 'Mentor Feedback' specifically.
Return exactly one flashcard."""),
                ("user", "Source text:\n\n{source_text}\n\nMentor Feedback: {feedback}")
            ])

            recreate_chain = recreate_prompt | self.llm.with_structured_output(FlashcardOutput)
            result = recreate_chain.invoke({"source_text": source_text, "feedback": feedback})

            if not result.flashcards:
                return {"status": "error", "message": "LLM returned no flashcard."}

            card = result.flashcards[0]
            fc.question = card.question
            fc.answer = card.answer
            fc.question_type = card.question_type
            fc.rubric = json.dumps([r.model_dump() for r in card.rubric])
            fc.mentor_feedback = feedback
            fc.status = "pending"

            new_question = fc.question
            new_answer = fc.answer
            fc_id = fc.id
            db.commit()

            return {
                "status": "success",
                "flashcard_id": fc_id,
                "question": new_question,
                "answer": new_answer,
            }
        finally:
            db.close()

    def suggest_answer(self, question: str, flashcard_id: int) -> str:
        """Queries the LLM to get a suggested answer for an existing question."""
        db = SessionLocal()
        try:
            fc = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
            chunk = db.query(ContentChunk).filter(ContentChunk.id == fc.chunk_id).first()
            source_text = chunk.text if chunk else "Source text unavailable."

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert educator. Based on the source text, provide a concise and precise answer to the given question."),
                ("user", f"Source text:\n\n{source_text}\n\nQuestion: {question}")
            ])

            chain = prompt | self.llm
            res = chain.invoke({})
            return res.content.strip()
        finally:
            db.close()
