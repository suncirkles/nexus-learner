"""
workflows/phase1_ingestion.py
------------------------------
LangGraph workflow for Phase 1: Document Ingestion & Flashcard Generation.
Defines the stateful graph: Ingest → Curate → Generate → Critic → (Loop).
Each chunk is processed through the generate/critic cycle, with a conditional
edge to loop back for remaining chunks or terminate at the end.
"""

from typing import TypedDict, List, Annotated, Dict, Any
from langgraph.graph import StateGraph, START, END
from agents.ingestion import IngestionAgent
from agents.socratic import SocraticAgent
from agents.critic import CriticAgent
from agents.curator import CuratorAgent
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Define the State
class GraphState(TypedDict):
    file_path: str
    doc_id: str
    subject_id: int
    full_text: str 
    chunks: List[Any]
    hierarchy: List[Dict[str, Any]]
    doc_summary: str
    current_chunk_index: int
    generated_flashcards: List[Dict[str, Any]]
    status_message: str

# 2. Initialize Agents
ingestion_agent = IngestionAgent()
socratic_agent = SocraticAgent()
critic_agent = CriticAgent()
curator_agent = CuratorAgent()

# 3. Define Nodes
def node_ingest(state: GraphState):
    if state.get("chunks"):
        logger.info("--- SKIPPING INGEST: Chunks already present in state ---")
        return {}
        
    logger.info(f"--- INGESTING: {state['file_path']} for Subject ID: {state['subject_id']} ---")
    chunks = ingestion_agent.process_document(state["file_path"], state["doc_id"], state["subject_id"])
    
    # Combine chunk text for curator
    full_text = "\n\n".join([c.page_content for c in chunks])
    
    return {
        "chunks": chunks,
        "full_text": full_text,
        "status_message": f"Successfully parsed document into {len(chunks)} sections."
    }

def node_curate(state: GraphState):
    if state.get("hierarchy") and state.get("doc_summary"):
        logger.info("--- SKIPPING CURATE: Hierarchy already present in state ---")
        return {}
        
    logger.info("--- CURATING STRUCTURE ---")
    result = curator_agent.curate_structure(state["subject_id"], state["doc_id"], state["full_text"])
    return {
        "hierarchy": result["hierarchy"],
        "doc_summary": result["doc_summary"],
        "status_message": f"Integrated content into {len(result['hierarchy'])} topics."
    }

def node_generate(state: GraphState):
    idx = state["current_chunk_index"]
    chunk = state["chunks"][idx]
    logger.info(f"--- GENERATING FLASHCARDS FOR CHUNK {idx+1}/{len(state['chunks'])} ---")
    
    # Find the best matching subtopic for this chunk
    subtopic_id = None
    subtopic_name = "General"
    
    if state["hierarchy"]:
        # Simple heuristic: find subtopic whose name/summary matches chunk text or use LLM
        # For better accuracy, we'll use a tiny LLM call to classify
        from core.models import get_llm
        from langchain_core.prompts import ChatPromptTemplate
        from pydantic import BaseModel, Field
        
        class Classification(BaseModel):
            subtopic_id: int = Field(description="The ID of the most relevant subtopic.")

        classifier = get_llm(purpose="routing").with_structured_output(Classification)
        
        all_subs = []
        for t in state["hierarchy"]:
            for s in t["subtopics"]:
                all_subs.append(f"ID {s['id']}: {s['name']} ({s['summary']})")
        
        mapping_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a classifier. Given a text chunk and a list of subtopics, return the ID of the ONE subtopic it most closely belongs to.\n\nSubtopics:\n{subtopics}"),
            ("user", "Chunk Text: {text}")
        ])
        
        try:
            mapping_result = classifier.invoke(
                mapping_prompt.format(
                    subtopics="\n".join(all_subs),
                    text=chunk.page_content[:1000]
                )
            )
            subtopic_id = mapping_result.subtopic_id
            
            # Find name for status message
            for t in state["hierarchy"]:
                for s in t["subtopics"]:
                    if s["id"] == subtopic_id:
                        subtopic_name = s["name"]
                        break
        except Exception as e:
            logger.warning(f"Classification failed: {e}. Defaulting to first subtopic.")
            subtopic_id = state["hierarchy"][0]["subtopics"][0]["id"]
            subtopic_name = state["hierarchy"][0]["subtopics"][0]["name"]
    
    flashcard = socratic_agent.generate_flashcard(state["doc_id"], chunk, subtopic_id=subtopic_id)
    
    return {
        "generated_flashcards": state.get("generated_flashcards", []) + [flashcard],
        "status_message": f"Generating Q&A for Subtopic: '{subtopic_name}' ({idx+1}/{len(state['chunks'])})"
    }

def node_critic(state: GraphState):
    flashcard = state["generated_flashcards"][-1]
    idx = state["current_chunk_index"]
    chunk = state["chunks"][idx]
    
    logger.info(f"--- EVALUATING FLASHCARD {idx+1} ---")
    # For our simple CriticAgent, we need the flashcard ID
    fc_id = flashcard.get("flashcard_id")
    if fc_id:
        critic_agent.evaluate_flashcard(
            flashcard_id=fc_id,
            source_text=chunk.page_content,
            question=flashcard["question"],
            answer=flashcard["answer"]
        )
    
    return {
        "status_message": f"Verified grounding for content {idx+1}. Ready for review."
    }

def should_continue(state: GraphState):
    if state["current_chunk_index"] < len(state["chunks"]) - 1:
        return "continue"
    return "end"

def node_increment(state: GraphState):
    return {"current_chunk_index": state["current_chunk_index"] + 1}

# 4. Build the Graph
workflow = StateGraph(GraphState)

workflow.add_node("ingest", node_ingest)
workflow.add_node("curate", node_curate)
workflow.add_node("generate", node_generate)
workflow.add_node("critic", node_critic)
workflow.add_node("increment", node_increment)

workflow.add_edge(START, "ingest")
workflow.add_edge("ingest", "curate")
workflow.add_edge("curate", "generate")
workflow.add_edge("generate", "critic")

workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "continue": "increment",
        "end": END
    }
)

workflow.add_edge("increment", "generate")

phase1_graph = workflow.compile()
