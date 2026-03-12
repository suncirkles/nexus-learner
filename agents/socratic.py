"""
agents/socratic.py
-------------------
AI-powered flashcard generator using Active Recall principles.
Generates high-quality Q&A pairs from text chunks, saves them to the
database, and supports flashcard recreation based on mentor feedback.
Also provides LLM-suggested answers for the review workflow.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from core.models import get_llm
from core.database import SessionLocal, Flashcard, ContentChunk
from core.config import settings

class FlashcardOutput(BaseModel):
    question: str = Field(description="The active recall question based on the text.")
    answer: str = Field(description="The precise answer found in the text.")

class SocraticAgent:
    def __init__(self):
        # We use the primary model (e.g. GPT-4o or Sonnet) for robust generation
        self.llm = get_llm(purpose="primary", temperature=0.3)
        
        # Guardrails enforced via prompt engineering
        self.prompt = ChatPromptTemplate.from_messages([
             ("system", """You are an expert educator. Your goal is to generate a SINGLE high-quality 'Active Recall' question and answer pair based STRICTLY on the provided source text. 
             
             CRITICAL GUIDELINES:
             1. Focus ONLY on 'Essential Concepts', 'High-Impact Knowledge', or 'Core Definitions'. 
             2. DO NOT generate questions for trivial details, examples, or filler text.
             3. If the text does not contain a significant, testable concept, return 'Not enough info' for both fields.
             4. The question should be challenging and promote deep understanding (Active Recall)."""),
             ("user", "Source text:\n\n{text}")
         ])
         
        self.chain = self.prompt | self.llm.with_structured_output(FlashcardOutput)

    def generate_flashcard(self, doc_id: str, chunk: ContentChunk, subtopic_id: int = None) -> dict:
        """Generates a flashcard based on a chunk and stages it to the database."""
        
        # Generate Q&A
        result = self.chain.invoke({"text": chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)})
        
        if result.question == "Not enough info":
            return {"status": "skipped", "reason": "Insufficient information in chunk."}

        # Save to DB with 'pending' or 'approved' state
        db = SessionLocal()
        try:
             initial_status = "approved" if settings.AUTO_ACCEPT_CONTENT else "pending"
             fc = Flashcard(
                 chunk_id=chunk.metadata.get("db_chunk_id") if hasattr(chunk, 'metadata') else None,
                 subtopic_id=subtopic_id,
                 question=result.question,
                 answer=result.answer,
                 status=initial_status
             )
             db.add(fc)
             db.commit()
             db.refresh(fc)
             
             return {
                 "status": "success",
                 "flashcard_id": fc.id,
                 "question": fc.question,
                 "answer": fc.answer
             }
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
            
            # Revised prompt for recreation
            recreate_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert educator. Your goal is to REVISE a flashcard based on mentor feedback and the original source text.
                
                CRITICAL GUIDELINES:
                1. Address the 'Mentor Feedback' specifically.
                2. Ensure the new 'Active Recall' question is high-quality and grounded in the source text.
                3. Return exactly one question and answer pair."""),
                ("user", f"Source text:\n\n{source_text}\n\nMentor Feedback: {feedback}")
            ])
            
            recreate_chain = recreate_prompt | self.llm.with_structured_output(FlashcardOutput)
            result = recreate_chain.invoke({})
            
            # Update the existing flashcard or create a new one? 
            # Request says "recreate", let's update the existing one and move to pending.
            fc.question = result.question
            fc.answer = result.answer
            fc.mentor_feedback = feedback
            fc.status = "pending"
            db.commit()
            
            return {
                "status": "success",
                "flashcard_id": fc.id,
                "question": fc.question,
                "answer": fc.answer
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
