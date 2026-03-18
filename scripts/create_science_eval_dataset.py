"""
scripts/create_science_eval_dataset.py
--------------------------------------
Extracts text chunks from Class 10 Science 'Acids, Bases and Salts' PDFs to create a curated eval dataset.
"""

import json
import os
import fitz  # PyMuPDF
import random

PDF_PATHS = [
    r"D:\projects\Gen-AI\Nexus Learner\documents\class 10 - chapter 2 science - acids and bases.pdf",
    r"D:\projects\Gen-AI\Nexus Learner\documents\class 10 science acids and bases - q &a .pdf"
]

OUT_OF_TOPIC_PDF = r"D:\projects\Gen-AI\Nexus Learner\documents\Chemical Reactions and Equa.pdf"

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "evals", "dataset_science.json")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def extract_chunks(pdf_path, chunks_per_pdf=20, chunk_size=1000):
    chunks = []
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return chunks
        
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            text = page.get_text()
            text = " ".join(text.split())
            
            if len(text) > 150:
                # Extract multiple chunks per page to ensure we get enough from just 2 PDFs
                for _ in range(5):
                    start_idx = random.randint(0, max(0, len(text) - min(chunk_size, len(text))))
                    chunk = text[start_idx:start_idx + chunk_size]
                    chunks.append(chunk.strip())
                
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        
    random.shuffle(chunks)
    return chunks[:chunks_per_pdf]

def categorize_chunk(chunk: str):
    """Heuristic for science chunks"""
    chunk_lower = chunk.lower()
    if "+" in chunk and "->" in chunk or "→" in chunk:
        return "equation"
    if "experiment" in chunk_lower or "activity" in chunk_lower or "take" in chunk_lower and "test tube" in chunk_lower:
        return "scenario"
    return "text"

def main():
    print("Extracting text from Science PDFs...")
    all_chunks = []
    
    for pdf_path in PDF_PATHS:
        chunks = extract_chunks(pdf_path, chunks_per_pdf=40)
        all_chunks.extend(chunks)
        
    equations = [c for c in all_chunks if categorize_chunk(c) == "equation"]
    scenarios = [c for c in all_chunks if categorize_chunk(c) == "scenario"]
    texts = [c for c in all_chunks if categorize_chunk(c) == "text"]
    
    dataset = []
    
    # 5 Simple Recall
    for i in range(5):
        dataset.append({
            "id": f"sci_{len(dataset)+1}", 
            "source_text": texts.pop() if texts else "Acids turn blue litmus red.", 
            "target_question_type": "active_recall", 
            "expected_complexity": "simple", 
            "topic": "Acids, Bases and Salts",
            "subtopic": "Properties and Indicators"
        })
        
    # 5 Medium Analytical (Fill blank or short answer based on text/equations)
    for i in range(5):
        dataset.append({
            "id": f"sci_{len(dataset)+1}", 
            "source_text": equations.pop() if equations else (texts.pop() if texts else "Acid + Base -> Salt + Water"),
            "target_question_type": "fill_blank", 
            "expected_complexity": "medium", 
            "topic": "Acids, Bases and Salts",
            "subtopic": "Chemical Reactions"
        })
        
    # 5 Complex Scenario/Application 
    for i in range(5):
        dataset.append({
            "id": f"sci_{len(dataset)+1}", 
            "source_text": scenarios.pop() if scenarios else (texts.pop() if texts else "A student is given three test tubes containing distilled water, acid, and base respectively."), 
            "target_question_type": "scenario", 
            "expected_complexity": "complex", 
            "topic": "Acids, Bases and Salts",
            "subtopic": "Experimental Activities"
        })
        
    # 5 Invalid/Bad Data
    bad_data = [
        "page 12 \n \n \n", 
        "The", 
        "Contents Chapter 1 Chapter 2", 
        "xyz "*20,
        "Figure 1.1 shows a graph."
    ]
    for i, bad in enumerate(bad_data):
        dataset.append({
            "id": f"sci_{len(dataset)+1}", 
            "source_text": bad, 
            "target_question_type": "active_recall", 
            "expected_complexity": "simple", 
            "topic": "Unknown Document",
            "subtopic": "Unclassified Data",
            "is_bad_data": True
        })

    # 2 Out of Topic Data (Valid science text but wrong topic constraint to test CriticAgent intersection)
    out_of_topic_chunks = extract_chunks(OUT_OF_TOPIC_PDF, chunks_per_pdf=10, chunk_size=800)
    for i in range(2):
        dataset.append({
            "id": f"sci_{len(dataset)+1}_out_of_topic", 
            "source_text": out_of_topic_chunks.pop() if out_of_topic_chunks else "Zinc reacts with copper sulphate to form zinc sulphate and copper.", 
            "target_question_type": "active_recall", 
            "expected_complexity": "medium", 
            "topic": "Acids, Bases and Salts",  # INTENTIONALLY MISMATCHED
            "subtopic": "Properties and Indicators", # INTENTIONALLY MISMATCHED
            "is_bad_data": False # It's valid text, just wrong topic context
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Science Dataset generated with {len(dataset)} items and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
