"""
agents/curator.py
------------------
Content structuring agent. Analyzes raw text to produce a hierarchical
Topic → Subtopic structure using LLM-powered extraction. Merges new
content into existing Subject hierarchies, avoiding duplicate topics.
Persists the hierarchy to the relational database.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from core.models import get_llm
from core.database import SessionLocal, Topic, Subtopic

class SubtopicStructure(BaseModel):
    name: str = Field(description="Name of the sub-topic or lesson.")
    summary: str = Field(description="A 1-sentence summary of what this sub-topic covers.")

class TopicStructure(BaseModel):
    name: str = Field(description="Name of the broad topic or chapter.")
    summary: str = Field(description="A concise summary of the topic.")
    subtopics: List[SubtopicStructure] = Field(description="List of sub-topics within this topic.")

class DocumentStructure(BaseModel):
    summary: str = Field(description="An overall summary of the document.")
    topics: List[TopicStructure] = Field(description="Hierarchical list of topics and their sub-topics.")

class CuratorAgent:
    def __init__(self):
        self.llm = get_llm(purpose="primary")
        self.structured_llm = self.llm.with_structured_output(DocumentStructure)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Educational Content Curator. 
            Your task is to analyze the provided educational material and organize it into a logical hierarchy of Topics and Sub-topics.
            
            IMPORTANT: You are adding content to an existing Subject. Below is the current hierarchy of Topics and Sub-topics for this Subject.
            
            Current Hierarchy:
            {existing_structure}
            
            Instructions:
            1. Analyze the new content provided.
            2. Match the new content to the EXISTING Topics and Sub-topics where possible.
            3. If the content covers a NEW broad area, create a new Topic.
            4. If the content covers a NEW lesson within an existing Topic, add a new Sub-topic to that topic.
            5. Provide a concise summary for the new artifact, and update/provide summaries for topics and sub-topics.
            6. Return the FULL hierarchy for the new content (including matching existing ones).
            """),
            ("human", "Analyze the following content and integrate it into the subject hierarchy:\n\n{content}")
        ])
        
        self.chain = self.prompt | self.structured_llm

    def curate_structure(self, subject_id: int, doc_id: str, full_text: str):
        """Analyzes text and merges it into the Subject's topic hierarchy."""
        db = SessionLocal()
        existing_structure_text = "No existing topics."
        try:
            existing_topics = db.query(Topic).filter(Topic.subject_id == subject_id).all()
            if existing_topics:
                lines = []
                for t in existing_topics:
                    subs = db.query(Subtopic).filter(Subtopic.topic_id == t.id).all()
                    sub_names = ", ".join([s.name for s in subs])
                    lines.append(f"- Topic: {t.name} (Sub-topics: {sub_names})")
                existing_structure_text = "\n".join(lines)
            
            analysis_text = full_text[:15000] 
            structure = self.chain.invoke({
                "content": analysis_text,
                "existing_structure": existing_structure_text
            })
            
            topics_data = []
            for t in structure.topics:
                # Check for existing topic by name (case-insensitive)
                db_topic = db.query(Topic).filter(
                    Topic.subject_id == subject_id,
                    Topic.name.ilike(t.name)
                ).first()
                
                if not db_topic:
                    db_topic = Topic(
                        subject_id=subject_id,
                        document_id=doc_id, # Link it to the first document that created it
                        name=t.name,
                        summary=t.summary
                    )
                    db.add(db_topic)
                    db.flush()
                
                subtopics_list = []
                for s in t.subtopics:
                    # Check for existing subtopic within this topic
                    db_subtopic = db.query(Subtopic).filter(
                        Subtopic.topic_id == db_topic.id,
                        Subtopic.name.ilike(s.name)
                    ).first()
                    
                    if not db_subtopic:
                        db_subtopic = Subtopic(
                            topic_id=db_topic.id,
                            name=s.name,
                            summary=s.summary
                        )
                        db.add(db_subtopic)
                        db.flush()
                    
                    subtopics_list.append({
                        "id": db_subtopic.id,
                        "name": db_subtopic.name,
                        "summary": db_subtopic.summary
                    })
                
                topics_data.append({
                    "id": db_topic.id,
                    "name": db_topic.name,
                    "summary": db_topic.summary,
                    "subtopics": subtopics_list
                })
            
            db.commit()
            return {
                "doc_summary": structure.summary,
                "hierarchy": topics_data
            }
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
