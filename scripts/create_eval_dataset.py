"""
scripts/create_eval_dataset.py
------------------------------
Extracts text chunks from Sheldon Probability PDFs to create a curated eval dataset
for testing the SocraticAgent. Curates 20 specific items of varying types, complexities,
and includes invalid data.
"""

import json
import os
import fitz  # PyMuPDF
import random

PDF_PATHS = [
    r"D:\cse\Probability\sheldon-chapter-1.pdf",
    r"D:\cse\Probability\sheldon-chapter2.pdf",
    r"D:\cse\Probability\sheldon solutions 1-2.pdf"
]

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "evals", "dataset.json")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def extract_chunks(pdf_path, chunks_per_pdf=20, chunk_size=1500):
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
                start_idx = random.randint(0, max(0, len(text) - min(chunk_size, len(text))))
                chunk = text[start_idx:start_idx + chunk_size]
                chunks.append(chunk.strip())
                
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        
    random.shuffle(chunks)
    return chunks[:chunks_per_pdf]

def categorize_chunk(chunk: str):
    """Simple heuristic to categorize chunks"""
    if "=" in chunk and ("+" in chunk or "-" in chunk):
        return "equation"
    if sum(c.isdigit() for c in chunk) > len(chunk) * 0.05:
        return "numerical"
    return "text"

def main():
    print("Extracting text from PDFs...")
    all_chunks = []
    
    for pdf_path in PDF_PATHS:
        chunks = extract_chunks(pdf_path, chunks_per_pdf=20)
        all_chunks.extend(chunks)
        
    # Pool chunks by category
    equations = [c for c in all_chunks if categorize_chunk(c) == "equation"]
    numericals = [c for c in all_chunks if categorize_chunk(c) == "numerical"]
    texts = [c for c in all_chunks if categorize_chunk(c) == "text"]
    
    dataset = []
    
    # Let's build exactly 20 diverse items
    # 4 Simple Recall (Text)
    for i in range(4):
        dataset.append({
            "id": f"item_{len(dataset)+1}", 
            "source_text": texts.pop() if texts else "Basic probability definition.", 
            "target_question_type": "active_recall", 
            "expected_complexity": "simple", 
            "topic": "Fundamentals of Probability",
            "subtopic": "Definitions and Terminology"
        })
        
    # 4 Medium Fill Blank (Text)
    for i in range(4):
        dataset.append({
            "id": f"item_{len(dataset)+1}", 
            "source_text": texts.pop() if texts else "The sample space of an experiment is the set of all possible outcomes.", 
            "target_question_type": "fill_blank", 
            "expected_complexity": "medium", 
            "topic": "Fundamentals of Probability",
            "subtopic": "Sample Spaces and Events"
        })
        
    # 4 Complex Numerical (Equations/Numericals)
    for i in range(4):
        src = equations.pop() if equations else (numericals.pop() if numericals else "P(A|B) = P(A AND B) / P(B). Find P(A) if P(B)=0.5 and P(A|B)=0.2")
        dataset.append({
            "id": f"item_{len(dataset)+1}", 
            "source_text": src, 
            "target_question_type": "numerical", 
            "expected_complexity": "complex", 
            "topic": "Conditional Probability and Independence",
            "subtopic": "Bayes' Theorem and Conditional Logic"
        })
        
    # 4 Scenario/Application (Texts/Numericals)
    for i in range(4):
        dataset.append({
            "id": f"item_{len(dataset)+1}", 
            "source_text": texts.pop() if texts else "Consider a coin tossing game.", 
            "target_question_type": "scenario", 
            "expected_complexity": "medium", 
            "topic": "Applied Probability Models",
            "subtopic": "Independent Trials and Games"
        })
        
    # 4 Invalid/Bad Data (OCR errors, too short, irrelevant text)
    bad_data = [
        "x % $ & * ( ) ) \n  table 1 1 2 3 \n", 
        "The", 
        "This page intentionally left blank.", 
        "abcd1234"*10
    ]
    for i, bad in enumerate(bad_data):
        dataset.append({
            "id": f"item_{len(dataset)+1}", 
            "source_text": bad, 
            "target_question_type": "active_recall", 
            "expected_complexity": "simple", 
            "topic": "Unknown Document",
            "subtopic": "Unclassified Data",
            "is_bad_data": True
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Dataset generated with {len(dataset)} items and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    random.seed(42) # For reproducibility
    main()
