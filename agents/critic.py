"""
agents/critic.py
-----------------
Grounding evaluation agent. Scores AI-generated flashcards across four
dimensions (accuracy, logic, grounding, clarity) on a 1–4 scale each.
Auto-rejects cards where grounding_score < 2.
Backward-compatible aggregate critic_score = round(mean of 4 sub-scores).
"""

import json
import logging
from math import ceil
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from core.models import get_llm
from core.database import SessionLocal, Flashcard
from core.config import settings

logger = logging.getLogger(__name__)


class RubricEvaluation(BaseModel):
    accuracy_score: int = Field(
        description="1-4: Is the answer factually correct per the source text? 4=fully correct, 1=factually wrong.",
        ge=1, le=4,
    )
    logic_score: int = Field(
        description="1-4: Is the reasoning/derivation sound? 4=flawless, 1=logically broken.",
        ge=1, le=4,
    )
    grounding_score: int = Field(
        description="1-4: Is the answer traceable to specific text in the source? 4=directly quoted/paraphrased, 1=not found in source.",
        ge=1, le=4,
    )
    clarity_score: int = Field(
        description="1-4: Is the question/answer unambiguous and well-phrased? 4=crystal clear, 1=confusing.",
        ge=1, le=4,
    )
    feedback: str = Field(description="Brief explanation justifying the scores.")
    suggested_complexity: str = Field(
        description="Complexity of this card: simple | medium | complex. Use the heuristic: grounding>=3 and logic<=2 → simple; grounding>=3 and logic=3 → medium; logic=4 and accuracy=4 → complex."
    )


_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an objective AI critic evaluating an AI-generated flashcard against the original Source Text.

Score each dimension on a scale of 1–4:
- accuracy_score  (1-4): Is the answer factually correct per the source?
- logic_score     (1-4): Is the reasoning sound?
- grounding_score (1-4): Is the answer directly traceable to text in the source?
- clarity_score   (1-4): Is the question/answer unambiguous?

Complexity heuristic (apply exactly):
- grounding_score >= 3 AND logic_score <= 2  → simple
- grounding_score >= 3 AND logic_score == 3  → medium
- logic_score == 4 AND accuracy_score == 4   → complex
- otherwise                                  → medium

Return structured output with all 6 fields."""),
    ("user", "Source Text:\n{source_text}\n\nGenerated Question: {question}\nGenerated Answer: {answer}"),
])


# Backward-compatible alias (existing tests import GroundingEvaluation)
GroundingEvaluation = RubricEvaluation


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _aggregate(acc: int, log: int, grd: int, cla: int) -> int:
    return round((acc + log + grd + cla) / 4)


class CriticAgent:
    def __init__(self):
        self.llm = get_llm(purpose="primary", temperature=0.0)
        self.chain = _EVAL_PROMPT | self.llm.with_structured_output(RubricEvaluation)

    def evaluate_flashcard(
        self,
        flashcard_id: int,
        source_text: str,
        question: str,
        answer: str,
    ) -> dict:
        """Evaluates a flashcard, writes 4-score rubric + aggregate to DB."""
        try:
            eval_result = self.chain.invoke({
                "source_text": source_text,
                "question": question,
                "answer": answer,
            })
            acc = _clamp(eval_result.accuracy_score, 1, 4)
            log = _clamp(eval_result.logic_score, 1, 4)
            grd = _clamp(eval_result.grounding_score, 1, 4)
            cla = _clamp(eval_result.clarity_score, 1, 4)
            feedback = eval_result.feedback or ""
            suggested_complexity = eval_result.suggested_complexity or "medium"
            # Normalise complexity to known values
            if suggested_complexity not in ("simple", "medium", "complex"):
                suggested_complexity = "medium"
        except Exception as e:
            logger.error("CriticAgent evaluation failed for flashcard %d: %s", flashcard_id, e)
            return {"error": f"Evaluation failed: {e}"}

        aggregate_score = _aggregate(acc, log, grd, cla)
        rubric_scores = {
            "accuracy": acc,
            "logic": log,
            "grounding": grd,
            "clarity": cla,
        }

        db = SessionLocal()
        try:
            fc = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
            if not fc:
                return {"error": "Flashcard not found"}

            fc.critic_score = aggregate_score
            fc.critic_feedback = feedback
            fc.critic_rubric_scores = json.dumps(rubric_scores)
            # Write suggested_complexity so Mentor HITL can pre-populate the selectbox
            fc.complexity_level = suggested_complexity

            # Auto-reject rule: grounding_score < 2 means answer is not in the source
            if grd < 2:
                if settings.AUTO_ACCEPT_CONTENT:
                    logger.warning(
                        "Flashcard %d grounding_score=%d/4 (not in source) but AUTO_ACCEPT_CONTENT=True "
                        "— card remains approved. Feedback: %s",
                        flashcard_id, grd, feedback,
                    )
                else:
                    fc.status = "rejected"
                    logger.info(
                        "Flashcard %d auto-rejected (grounding_score=%d/4). Feedback: %s",
                        flashcard_id, grd, feedback,
                    )

            db.commit()
            return {
                "flashcard_id": fc.id,
                "score": aggregate_score,
                "rubric_scores": rubric_scores,
                "suggested_complexity": suggested_complexity,
                "feedback": feedback,
            }
        finally:
            db.close()
