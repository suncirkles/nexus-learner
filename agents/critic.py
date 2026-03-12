"""
agents/critic.py
-----------------
Grounding evaluation agent. Scores AI-generated flashcards (1–5) by
comparing the question and answer against the original source text.
Auto-rejects severely hallucinated content (score < 3).
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from core.models import get_llm
from core.database import SessionLocal, Flashcard

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
        eval_result = self.chain.invoke({
            "source_text": source_text,
            "question": question,
            "answer": answer
        })
        
        db = SessionLocal()
        try:
             fc = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
             if fc:
                 fc.critic_score = eval_result.score
                 fc.critic_feedback = eval_result.feedback
                 
                 # Optional auto-reject heuristic based on eval
                 if eval_result.score < 3:
                      # If it severely hallucinated, force disapproval regardless of auto_accept
                      fc.status = "rejected"
                 
                 db.commit()
                 
                 return {
                     "flashcard_id": fc.id,
                     "score": eval_result.score,
                     "feedback": eval_result.feedback
                 }
        finally:
             db.close()
             
        return {"error": "Flashcard not found"}
