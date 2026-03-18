"""
agents/socratic.py
-------------------
AI-powered flashcard generator supporting multiple question types.
Generates high-quality Q&A pairs from text chunks and returns FlashcardDraft
objects — no DB writes.

Phase 2b change: generate_flashcard() no longer writes to the database.
It returns a list of FlashcardDraft dataclasses. The calling workflow node
persists them via flashcard_repo.create().

recreate_flashcard() and suggest_answer() still access the DB directly because
they are UI-driven (Mentor Review) and will be decoupled in Phase 3.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Any as AnyType
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from core.models import get_llm
from core.database import SessionLocal, Flashcard, ContentChunk
from core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output models (Pydantic — used for LLM structured output)
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
# Pure-data result returned to the workflow node
# ---------------------------------------------------------------------------

@dataclass
class FlashcardDraft:
    """Immutable data transfer object returned by SocraticAgent.generate_flashcard().

    The calling workflow node writes these to the DB via flashcard_repo.create().
    """
    question: str
    answer: str
    question_type: str
    rubric_json: str           # json.dumps([{"criterion": ..., "description": ...}, ...])
    suggested_complexity: str


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
6. suggested_complexity — Bloom's Taxonomy / CBSE HOTS:
   simple  = answer is a single fact directly quotable from one sentence in the source (Remember/Understand).
             Action verbs: Define, List, State, Identify, Summarize.
   medium  = requires applying a concept, multi-step reasoning, or comparing/contrasting ideas in the source (Apply/Analyse).
             Action verbs: Calculate, Solve, Compare, Contrast, Explain.
   complex = requires evaluation, justification, design, or Transfer of Learning to a novel context NOT described in the source (Evaluate/Create — HOTS only).
             Action verbs: Justify, Criticise, Formulate, Design, Integrate.
             RULE: if the answer can be directly quoted from one passage → simple or medium, never complex.
             Default to medium when uncertain.
7. SELF-CONTAINED RULE (applies to ALL question types):
   The question text MUST include every piece of data the student needs to answer it.
   NEVER write phrases like "Using Table 1...", "From the figure...", "Based on the data above...",
   "Refer to the table...", "According to the graph..." unless the actual table/figure/data
   is embedded directly inside the question text.
   If the source text contains a garbled, OCR-mangled, or partially illegible table, you MUST
   synthesise a clean, internally consistent replacement with the same concept and units.
   Synthesised data is allowed — it does not need to come verbatim from the source —
   but it MUST be conceptually correct and every number in the question must match the answer exactly.
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
Focus on: Creating NOVEL, variations of mathematical problems and derivations based on the concepts in the source text. 
DO NOT simply copy equations, problems, or examples verbatim. You MUST create new scenarios, change the numbers, vary the datasets, and combine concepts to test true understanding.

NUMERICAL-SPECIFIC RULES (in addition to CRITICAL GUIDELINES below):
- You MUST invent synthetic data (new numbers, new tables, new entities) that test the same underlying principles found in the text.
- The question MUST be entirely self-contained. Embed ALL required data — tables, given values, formulas, equations — directly inside the question text using Markdown.
- Format data tables as Markdown: | Hour | Temp (°F) | ... with a header separator row.
- If the source has a problem, create a functionally equivalent but distinctly different problem. Introduce variety.
- The numbers and logic in your answer MUST strictly match the synthetic numbers you provided in the question.
- NEVER reference "Table 1", "Table 2", "the table above", "the figure", or "the graph" as external resources. If you need a table, build it and put it in the question.
- OVERRIDE BASE RULE 1: For numerical questions, it is ACCEPTABLE and EXPECTED that the specific numbers and scenario entities are NOT found in the source text, provided the mathematical logic is derived from the source.
{_BASE_RULES}""",

    "scenario": f"""You are an expert educator generating Scenario-based flashcards.
Focus on: "What happens if…" or "What would you do when…" problems derived from code snippets, processes, or case studies in the source.
The answer should explain the outcome or correct action with reference to the source.
{_BASE_RULES}""",
}


# ---------------------------------------------------------------------------
# Few-Shot Extraction (for numericals)
# ---------------------------------------------------------------------------
def _get_novel_numerical_examples() -> str:
    """Returns few-shot examples of finding a concept and creating a NOVEL variation."""
    return """
EXAMPLE SOURCE TEXT:
A system has 3 components. The probability of each failing is 0.1. What is the probability that all 3 fail? 
Solution: Assuming independence, P(all 3 fail) = (0.1)^3 = 0.001.

EXPECTED NOVEL FLASHCARD OUTPUT:
Question: An IoT network depends on 4 independent sensors. The probability that any single sensor goes offline during a storm is 0.05. Calculate the probability that all 4 sensors go offline simultaneously.
Answer: Assuming the sensor failures are independent events, the probability of intersection is the product of individual probabilities. P(all 4 offline) = (0.05)^4 = 0.00000625.
""".replace("{", "{{").replace("}", "}}")

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SocraticAgent:
    def __init__(self, llm: AnyType = None):
        """Accept an optional llm for injection in tests; default to get_llm()."""
        self.llm = llm if llm is not None else get_llm(purpose="primary", temperature=0.3)
        self._chains: Dict[str, Any] = {}
        for qtype, system_msg in PROMPTS.items():
            if qtype == "numerical":
                system_msg += "\n\n" + _get_novel_numerical_examples()
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_msg),
                ("user", "Topic: {topic}\nSubtopic: {subtopic}\n\nSource text:\n\n{text}"),
            ])
            self._chains[qtype] = prompt | self.llm.with_structured_output(FlashcardOutput)

    def _get_chain(self, question_type: str):
        if question_type not in self._chains:
            logger.warning("Unknown question_type '%s', falling back to active_recall.", question_type)
            question_type = "active_recall"
        return self._chains[question_type]

    def generate_flashcard(
        self,
        source_text: str,
        question_type: str = "active_recall",
        topic: str = "General Knowledge",
        subtopic: str = "General Knowledge",
        context: str = "",
        # Legacy positional args kept for backward compatibility during Phase 2b transition
        doc_id: Optional[str] = None,
        chunk: Optional[Any] = None,
        subtopic_id: Optional[int] = None,
        subject_id: Optional[int] = None,
    ) -> List[FlashcardDraft]:
        """Generate flashcard(s) from source_text.

        Returns a list of FlashcardDraft objects — no DB writes.
        The calling workflow node creates DB records via flashcard_repo.create().

        If called with the old (chunk, doc_id) signature, source_text is resolved
        from the chunk object for backward compatibility.
        """
        # Resolve source_text when called with legacy chunk argument
        if chunk is not None and not source_text:
            if hasattr(chunk, "page_content"):
                source_text = chunk.page_content
            elif hasattr(chunk, "text"):
                source_text = chunk.text
            else:
                source_text = str(chunk)

        chain = self._get_chain(question_type)
        result = chain.invoke({"text": source_text, "topic": topic, "subtopic": subtopic})

        if not result.flashcards:
            return []

        drafts = []
        for card in result.flashcards:
            drafts.append(FlashcardDraft(
                question=card.question,
                answer=card.answer,
                question_type=question_type,  # caller-requested type; LLM label can drift
                rubric_json=json.dumps([r.model_dump() for r in card.rubric]),
                suggested_complexity=card.suggested_complexity,
            ))
        return drafts

    def recreate_flashcard(self, flashcard_id: int, feedback: str) -> dict:
        """Regenerates a flashcard based on mentor feedback and its original source.

        Still writes to DB directly — this is a UI-driven flow that will be
        decoupled in Phase 3 when the Mentor Review page calls the API.
        """
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
