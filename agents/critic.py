"""
agents/critic.py
-----------------
Grounding evaluation agent. Scores AI-generated flashcards (1–5) by
comparing the question and answer against the original source text.
Auto-rejects severely hallucinated content (score < 3).
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from core.models import get_llm
from core.database import SessionLocal, Flashcard
from core.config import settings

logger = logging.getLogger(__name__)

class GroundingEvaluation(BaseModel):
    score: int = Field(description="Score from 1 to 5. 5 means completely accurate and grounded in source text. 1 means hallucinated or completely incorrect.", ge=1, le=5)
    feedback: str = Field(description="Brief explanation of the score.")

class CriticAgent:
    def __init__(self):
        # The critic acts as an evaluator, utilizing the Primary model.
        # In a larger system, we might use a specialized NLI evaluation model.
        self.llm = get_llm(purpose="primary", temperature=0.0)
        
        self.prompt = ChatPromptTemplate.from_messages([
             ("system", "You are an objective AI critic. Your task is to evaluate an AI-generated flashcard (Question & Answer) against the original Source Text. Give a score from 1-5 where:\n5: Answer is perfectly correct and fully supported by the Source Text.\n3: Answer is partially correct but misses nuance.\n1: Answer hallucinates facts NOT present in the Source Text.\nProvide brief feedback justifying the score."),
             ("user", "Source Text:\n{source_text}\n\nGenerated Question: {question}\nGenerated Answer: {answer}")
         ])
         
        self.chain = self.prompt | self.llm.with_structured_output(GroundingEvaluation)
        
    def evaluate_flashcard(self, flashcard_id: int, source_text: str, question: str, answer: str) -> dict:
        """Evaluates a flashcard and updates its score in the database."""
        # H18: wrap LLM call — structured output can fail if model returns unexpected format
        try:
            eval_result = self.chain.invoke({
                "source_text": source_text,
                "question": question,
                "answer": answer
            })
            # H18: clamp score to valid range defensively (Pydantic ge/le should catch it,
            # but guard against None or non-int responses from edge-case LLM outputs)
            score = max(1, min(5, int(eval_result.score)))
            feedback = eval_result.feedback or ""
        except Exception as e:
            logger.error("CriticAgent evaluation failed for flashcard %d: %s", flashcard_id, e)
            return {"error": f"Evaluation failed: {e}"}

        db = SessionLocal()
        try:
            fc = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
            if fc:
                fc.critic_score = score
                fc.critic_feedback = feedback

                # H17: respect AUTO_ACCEPT_CONTENT.
                # When auto-accept is ON the user has explicitly opted out of HITL review,
                # so we should not silently force-reject cards — even low-scoring ones.
                # Log a warning instead so the quality signal is still visible in logs.
                if score < 3:
                    if settings.AUTO_ACCEPT_CONTENT:
                        logger.warning(
                            "Flashcard %d scored %d/5 (low quality) but AUTO_ACCEPT_CONTENT=True "
                            "— card remains approved. Feedback: %s",
                            flashcard_id, score, feedback,
                        )
                    else:
                        fc.status = "rejected"
                        logger.info(
                            "Flashcard %d auto-rejected (score %d/5). Feedback: %s",
                            flashcard_id, score, feedback,
                        )

                db.commit()
                return {
                    "flashcard_id": fc.id,
                    "score": score,
                    "feedback": feedback,
                }
        finally:
            db.close()

        return {"error": "Flashcard not found"}
