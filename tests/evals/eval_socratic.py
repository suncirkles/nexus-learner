"""
tests/evals/eval_socratic.py
-----------------------------
Evaluation harness for SocraticAgent using a curated dataset.
Runs code-based graders and an LLM-as-a-judge to evaluate generated flashcards.
"""

import json
import os
import sys
import argparse
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.socratic import SocraticAgent, FlashcardDraft
from core.models import get_llm
from langchain_core.prompts import ChatPromptTemplate

from agents.critic import CriticAgent

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please run scripts/create_eval_dataset.py first.")
        return []
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

def code_based_graders(flashcards: List[FlashcardDraft], expected_type: str) -> Dict[str, bool]:
    """Run deterministic checks on the generated flashcards."""
    summary = {
        "has_flashcards": len(flashcards) > 0,
        "valid_count": 1 <= len(flashcards) <= 3,
        "correct_type": True,
        "valid_rubric": True
    }
    
    if not flashcards:
        return summary
        
    for fc in flashcards:
        if fc.question_type != expected_type:
             summary["correct_type"] = False
             
        try:
            rubric = json.loads(fc.rubric_json)
            if len(rubric) != 3:
                summary["valid_rubric"] = False
        except Exception:
            summary["valid_rubric"] = False
            
    return summary

from pydantic import BaseModel, Field

class JudgeOutput(BaseModel):
    is_grounded: bool = Field(description="Is the answer fully supported by the source text without relying on outside knowledge?")
    is_self_contained: bool = Field(description="Is the question entirely self-contained? It must NOT reference external tables, figures, or say 'Based on the text above'.")

def llm_based_graders(llm_judge, source_text: str, flashcards: List[FlashcardDraft]) -> Dict[str, bool]:
    """Use an LLM to judge qualitative aspects like groundedness and self-containment."""
    if not flashcards:
        return {"is_grounded": False, "is_self_contained": False}
        
    # We evaluate just the first flashcard for simplicity in this baseline
    fc = flashcards[0]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an impartial evaluator grading an AI-generated flashcard against a source text.
        You must evaluate two criteria:
        1. Groundedness: Is the answer fully supported by the source text without relying on outside knowledge?
        2. Self-containment: Is the question entirely self-contained? It must NOT reference external tables, figures, or say 'Based on the text above'."""),
        ("user", "Source Text:\n{source_text}\n\nGenerated Question: {question}\nGenerated Answer: {answer}")
    ])
    
    chain = prompt | llm_judge.with_structured_output(JudgeOutput)
    
    try:
         result = chain.invoke({
             "source_text": source_text,
             "question": fc.question,
             "answer": fc.answer
         })
         return {
             "is_grounded": result.is_grounded,
             "is_self_contained": result.is_self_contained
         }
    except Exception as e:
         print(f"LLM judge failed: {e}")
         return {"is_grounded": False, "is_self_contained": False}
         
def run_evals(limit: int = None, dataset_file: str = "dataset.json", report_file: str = "detailed_eval_results.md"):
    dataset_path = os.path.join(os.path.dirname(__file__), dataset_file)
    dataset = load_dataset(dataset_path)
    if limit:
        dataset = dataset[:limit]
        
    print(f"Running evals on {len(dataset)} examples...")
    
    agent = SocraticAgent()
    judge_llm = get_llm(purpose="primary", temperature=0.0) # Low temp for judge
    critic_agent = CriticAgent(llm=judge_llm)
    
    results = []
    
    for i, item in enumerate(dataset):
        is_bad_data = item.get("is_bad_data", False)
        print(f"Processing item {i+1}/{len(dataset)} (Type: {item['target_question_type']}, Bad Data: {is_bad_data})...")
        
        try:
             # Run the generative agent
             flashcards = agent.generate_flashcard(
                  source_text=item["source_text"],
                  question_type=item["target_question_type"],
                  topic=item.get("topic", "General Knowledge"),
                  subtopic=item.get("subtopic", "General Knowledge")
             )
             
             # Grade it
             code_scores = code_based_graders(flashcards, item["target_question_type"])
             llm_scores = llm_based_graders(judge_llm, item["source_text"], flashcards)
             
             # Run the CriticAgent
             critic_scores = None
             fc_text = None
             if flashcards:
                 fc = flashcards[0]
                 fc_text = {"question": fc.question, "answer": fc.answer}
                 critic_res = critic_agent.evaluate_flashcard(
                     source_text=item["source_text"], 
                     question=fc.question, 
                     answer=fc.answer,
                     topic=item.get("topic", "General Knowledge"),
                     subtopic=item.get("subtopic", "General Knowledge")
                 )
                 critic_scores = {
                     "aggregate": critic_res.aggregate_score,
                     "accuracy": critic_res.rubric_scores.get("accuracy", 0),
                     "logic": critic_res.rubric_scores.get("logic", 0),
                     "grounding": critic_res.rubric_scores.get("grounding", 0),
                     "clarity": critic_res.rubric_scores.get("clarity", 0),
                     "should_reject": critic_res.should_reject
                 }
             
             results.append({
                 "id": item.get("id", str(i)),
                 "is_bad_data": is_bad_data,
                 "code_scores": code_scores,
                 "llm_scores": llm_scores,
                 "critic_scores": critic_scores,
                 "flashcard": fc_text
             })
             
        except Exception as e:
             print(f"Error evaluating item {i}: {e}")
             
    # Aggregate and print report
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    
    total = len(results)
    if total == 0:
        print("No results to report.")
        return
        
    metrics = {
        "Has Output": sum(1 for r in results if r["code_scores"]["has_flashcards"]) / total,
        "Valid Count (1-3)": sum(1 for r in results if r["code_scores"]["valid_count"]) / total,
        "Valid Rubric (Exactly 3)": sum(1 for r in results if r["code_scores"]["valid_rubric"]) / total,
        "Correct Type": sum(1 for r in results if r["code_scores"]["correct_type"]) / total,
        "Grounded Answer (Simple LLM)": sum(1 for r in results if r["llm_scores"]["is_grounded"]) / total,
        "Self-contained (Simple LLM)": sum(1 for r in results if r["llm_scores"]["is_self_contained"]) / total,
    }
    
    for name, score in metrics.items():
        print(f"{name:35} : {score*100:.1f}%")
        
    print("\n--- CriticAgent Metrics (Averages) ---")
    valid_critic = [r["critic_scores"] for r in results if r.get("critic_scores")]
    if valid_critic:
        avg_aggregate = sum(c["aggregate"] for c in valid_critic) / len(valid_critic)
        avg_acc = sum(c["accuracy"] for c in valid_critic) / len(valid_critic)
        avg_logic = sum(c["logic"] for c in valid_critic) / len(valid_critic)
        avg_ground = sum(c["grounding"] for c in valid_critic) / len(valid_critic)
        avg_clarity = sum(c["clarity"] for c in valid_critic) / len(valid_critic)
        reject_rate = sum(1 for c in valid_critic if c["should_reject"]) / len(valid_critic)
        
        print(f"{'Average Aggregate Score':35} : {avg_aggregate:.2f} / 4.0")
        print(f"{'Average Accuracy':35} : {avg_acc:.2f} / 4.0")
        print(f"{'Average Logic':35} : {avg_logic:.2f} / 4.0")
        print(f"{'Average Grounding':35} : {avg_ground:.2f} / 4.0")
        print(f"{'Average Clarity':35} : {avg_clarity:.2f} / 4.0")
        print(f"{'Rejection Rate':35} : {reject_rate*100:.1f}%")
    else:
        print("No CriticAgent scores collected.")
        
    print("\n--- Performance on Bad Data ---")
    bad_data_results = [r for r in results if r.get("is_bad_data")]
    if bad_data_results:
        bd_total = len(bad_data_results)
        bd_no_output = sum(1 for r in bad_data_results if not r["code_scores"]["has_flashcards"])
        bd_rejected = sum(1 for r in bad_data_results if r.get("critic_scores") and r["critic_scores"]["should_reject"])
        print(f"Total Bad Data Items: {bd_total}")
        print(f"Successfully refused to generate: {bd_no_output} ({bd_no_output/bd_total*100:.1f}%)")
        gen_total = bd_total - bd_no_output
    else:
        print("No bad data items in dataset.")
        
    # Write detailed results to markdown
    md_path = os.path.join(os.path.dirname(__file__), report_file)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Detailed Evaluation Results\n\n")
        for idx, (item, r) in enumerate(zip(dataset, results)):
            f.write(f"## Item {idx+1}: {item['target_question_type'].capitalize()}\n")
            f.write(f"**Bad Data:** {r.get('is_bad_data', False)}\n")
            f.write(f"**Topic:** {item.get('topic', 'N/A')} > {item.get('subtopic', 'N/A')}\n\n")
            f.write("### Source Text\n```text\n" + item["source_text"] + "\n```\n\n")
            
            if not r["code_scores"]["has_flashcards"]:
                f.write("> **Result:** Agent refused to generate flashcards.\n\n")
            else:
                fc = r.get("flashcard")
                if fc:
                    f.write(f"**Question:** {fc['question']}\n\n")
                    f.write(f"**Answer:** {fc['answer']}\n\n")
                f.write("Evaluation scores attached below.\n\n")
                
            f.write("### Scores\n")
            f.write(f"- Code Graders: {r['code_scores']}\n")
            f.write(f"- Basic LLM Judge: {r['llm_scores']}\n")
            c_scores = r.get("critic_scores")
            if c_scores:
                f.write(f"- Critic Agent: {c_scores}\n")
            f.write("---\n\n")
            
    print(f"Detailed results saved to {md_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SocraticAgent Evals")
    parser.add_argument("--limit", type=int, help="Limit the number of examples to evaluate")
    parser.add_argument("--dataset", type=str, default="dataset.json", help="Dataset json filename in tests/evals")
    parser.add_argument("--report", type=str, default="detailed_eval_results.md", help="Output md filename in tests/evals")
    args = parser.parse_args()
    
    run_evals(limit=args.limit, dataset_file=args.dataset, report_file=args.report)
