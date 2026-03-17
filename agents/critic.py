"""
agents/critic.py
-----------------
Grounding evaluation agent. Scores AI-generated flashcards across four
dimensions (accuracy, logic, grounding, clarity) on a 1–4 scale each.
Auto-rejects cards where grounding_score < 2 or clarity_score < 2.

Phase 2b change: evaluate_flashcard() no longer writes to the database.
It returns a CriticResult dataclass. The calling workflow node is
responsible for persisting scores via flashcard_repo.update_critic_scores()
and, when should_reject=True, updating the status via flashcard_repo.update_status().
"""

import json
import logging
from dataclasses import dataclass
from math import ceil
from typing import Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from core.models import get_llm, call_structured_chain
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
        description=(
            "1-4: Is the question/answer unambiguous and well-phrased? 4=crystal clear, 1=confusing. "
            "CRITICAL: if the question refers to 'Table N', 'the table', 'the figure', 'the graph', "
            "or any external dataset that is NOT actually embedded in the question text, assign 1 — "
            "the student has no way to answer without the missing data."
        ),
        ge=1, le=4,
    )
    feedback: str = Field(description="Brief explanation justifying the scores.")
    suggested_complexity: str = Field(
        description=(
            "Complexity of this card using Bloom's Taxonomy (CBSE HOTS framework): "
            "simple = answer is a single fact directly quotable from one sentence in the source (Remember/Understand); "
            "medium = requires applying a concept, multi-step assembly, or comparing ideas (Apply/Analyse); "
            "complex = requires evaluation, justification, design, or transfer to an unfamiliar context — "
            "a card whose answer can be directly quoted is NEVER complex (Evaluate/Create). "
            "Default to medium when uncertain."
        )
    )


_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an objective AI critic evaluating an AI-generated flashcard against the original Source Text.

Score each dimension on a scale of 1–4:
- accuracy_score  (1-4): Is the answer factually correct per the source?
- logic_score     (1-4): Is the reasoning sound?
- grounding_score (1-4): Is the answer directly traceable to text in the source?
- clarity_score   (1-4): Is the question/answer unambiguous and fully self-contained?

PHANTOM DATA REFERENCE — INSTANT FAIL:
If the question mentions "Table 1", "Table 2", "the table", "the data table", "the figure",
"the graph", "the chart", or ANY external dataset/figure that is NOT actually embedded
inside the question text as a Markdown table or explicit list of values, this is a critical flaw:
  → Set clarity_score = 1 (student cannot answer — required data is missing from the question)
  → Set grounding_score = 1 (question is not self-contained)
A question that says "Using the temperature data table, calculate..." with NO table in the
question text scores clarity = 1, grounding = 1, and will be auto-rejected.

Complexity classification — Bloom's Taxonomy / CBSE HOTS framework:

SIMPLE   (Bloom's: Remember / Understand)
  • The answer is a single fact that can be directly quoted or paraphrased from ONE sentence in the source.
  • A student who has read the passage once can answer it with no additional reasoning.
  • Typical action verbs: Define, List, State, Identify, Summarize.
  • Score signal: grounding_score = 4 AND logic_score ≤ 2.

MEDIUM   (Bloom's: Apply / Analyse)
  • Requires "horizontal" thinking: applying a formula/concept to a scenario, explaining a
    multi-step process, or comparing/contrasting ideas — but all evidence is in the source.
  • Typical action verbs: Calculate, Solve, Compare, Contrast, Explain, Illustrate.
  • Score signal: grounding_score ≥ 3 AND logic_score = 3.

COMPLEX  (Bloom's: Evaluate / Create — HOTS only)
  • Requires "vertical" integration: connecting concepts across different sections, evaluating or
    justifying a claim, designing a solution, OR applying a principle to a completely novel/unfamiliar
    scenario not described in the source (Transfer of Learning).
  • CRITICAL RULE: if the answer can be directly quoted from a single passage → NOT complex.
  • Typical action verbs: Justify, Criticise, Formulate, Design, Integrate, Evaluate.
  • Score signal: logic_score = 4 AND accuracy_score = 4 AND grounding_score ≤ 3
    (synthesis is required — the answer cannot be found in a single passage).

DEFAULT: when in doubt, assign MEDIUM. Reserve COMPLEX for genuine HOTS questions only.

Return structured output with all 6 fields."""),
    ("user", "Source Text:\n{source_text}\n\nGenerated Question: {question}\nGenerated Answer: {answer}"),
])


# Backward-compatible alias (existing tests import GroundingEvaluation)
GroundingEvaluation = RubricEvaluation


@dataclass
class CriticResult:
    """Pure-data result returned by CriticAgent.evaluate_flashcard().

    The caller (workflow node) is responsible for persisting these values
    via flashcard_repo.update_critic_scores() and handling should_reject.
    """
    aggregate_score: int
    rubric_scores: dict          # {"accuracy": int, "logic": int, "grounding": int, "clarity": int}
    rubric_scores_json: str      # json.dumps(rubric_scores) — pre-serialised for repo call
    feedback: str
    suggested_complexity: str
    should_reject: bool
    reject_reason: str           # human-readable, empty when should_reject=False
    error: Optional[str] = None  # set when LLM call fails


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _aggregate(acc: int, log: int, grd: int, cla: int) -> int:
    return round((acc + log + grd + cla) / 4)


class CriticAgent:
    def __init__(self, llm: Any = None):
        """Accept an optional llm for injection in tests; default to get_llm()."""
        self._llm = llm
        self._chain = self._build_chain()

    def _build_chain(self):
        llm = self._llm if self._llm is not None else get_llm(purpose="primary", temperature=0.0)
        return _EVAL_PROMPT | llm.with_structured_output(RubricEvaluation)

    def evaluate_flashcard(
        self,
        source_text: str,
        question: str,
        answer: str,
        flashcard_id: Optional[int] = None,  # kept for logging only
    ) -> CriticResult:
        """Evaluate a flashcard against its source text.

        Returns a CriticResult dataclass — no DB writes.
        The calling workflow node persists scores via flashcard_repo.
        """
        try:
            # Rebuild chain per-call when hopping so provider selection is fresh
            if settings.MODEL_HOP_ENABLED:
                self._chain = self._build_chain()
            eval_result = call_structured_chain(
                self._chain,
                RubricEvaluation,
                {"source_text": source_text, "question": question, "answer": answer},
            )
            acc = _clamp(eval_result.accuracy_score, 1, 4)
            log = _clamp(eval_result.logic_score, 1, 4)
            grd = _clamp(eval_result.grounding_score, 1, 4)
            cla = _clamp(eval_result.clarity_score, 1, 4)
            feedback = eval_result.feedback or ""
            suggested_complexity = eval_result.suggested_complexity or "medium"
            if suggested_complexity not in ("simple", "medium", "complex"):
                suggested_complexity = "medium"
        except Exception as e:
            logger.error(
                "CriticAgent evaluation failed for flashcard %s: %s",
                flashcard_id, e,
            )
            return CriticResult(
                aggregate_score=0,
                rubric_scores={},
                rubric_scores_json="{}",
                feedback="",
                suggested_complexity="medium",
                should_reject=False,
                reject_reason="",
                error=f"Evaluation failed: {e}",
            )

        aggregate_score = _aggregate(acc, log, grd, cla)
        rubric_scores = {"accuracy": acc, "logic": log, "grounding": grd, "clarity": cla}

        # Auto-reject rule 1: answer not traceable to source
        # Auto-reject rule 2: phantom data reference (table/figure not embedded)
        should_reject = grd < 2 or cla < 2
        reject_reason = (
            f"grounding_score={grd}/4" if grd < 2
            else f"clarity_score={cla}/4 (phantom data reference)"
        ) if should_reject else ""

        if should_reject:
            if settings.AUTO_ACCEPT_CONTENT:
                logger.warning(
                    "Flashcard %s %s but AUTO_ACCEPT_CONTENT=True — card stays approved.",
                    flashcard_id, reject_reason,
                )
                should_reject = False
                reject_reason = ""
            else:
                logger.info(
                    "Flashcard %s auto-reject (%s). Feedback: %s",
                    flashcard_id, reject_reason, feedback,
                )

        return CriticResult(
            aggregate_score=aggregate_score,
            rubric_scores=rubric_scores,
            rubric_scores_json=json.dumps(rubric_scores),
            feedback=feedback,
            suggested_complexity=suggested_complexity,
            should_reject=should_reject,
            reject_reason=reject_reason,
        )
