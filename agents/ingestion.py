"""
agents/ingestion.py
--------------------
Handles document ingestion: PDF text extraction (with OCR fallback),
content hashing for duplicate detection, text chunking, relational DB
persistence, and vector embedding into Qdrant.
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from core.config import settings
import hashlib
from core.database import SessionLocal, ContentChunk, Document as DBDocument

class IngestionAgent:
    def __init__(self):
        # Using OpenAI embeddings for standard text chunking
        # TODO: Fallback/Pluggable embeddings
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for Embeddings in MVP")
        self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extracts text from PDF, falling back to OCR if a page is purely image-based."""
        doc = fitz.open(file_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            
            # Simple heuristic: if less than 50 chars, assume it might be a scanned image
            if len(text.strip()) < 50:
                 try:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img)
                 except Exception as e:
                    # Gracefully skip OCR if tesseract is missing
                    print(f"OCR failed or Tesseract not found: {e}. Skipping OCR for page {page_num}.")
                 
            full_text += text + "\n\n"
            
        return full_text

    def extract_text_from_image(self, file_path: str) -> str:
        """Extracts text from an image directly using Tesseract."""
        try:
            img = Image.open(file_path)
            return pytesseract.image_to_string(img)
        except Exception as e:
            return f"Error: Tesseract OCR failed. Is it installed? Details: {e}"
        
    def get_content_hash(self, text: str) -> str:
        """Generates a SHA256 hash of the text content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def process_document(self, file_path: str, doc_id: str, subject_id: int) -> list[Document]:
        """Orchestrates extraction, chunking, DB persistence, and Vector DB embedding."""
        if file_path.lower().endswith('.pdf'):
            raw_text = self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            raw_text = self.extract_text_from_image(file_path)
        else:
             raise ValueError("Unsupported file type")
             
        # Duplicate Check
        content_hash = self.get_content_hash(raw_text)
        db = SessionLocal()
        try:
            existing_doc = db.query(DBDocument).filter(DBDocument.content_hash == content_hash).first()
            if existing_doc:
                raise ValueError(f"Duplicate content detected. This document has already been processed as '{existing_doc.filename}'.")
            
            uploaded_filename = os.path.basename(file_path)
            # Generate a meaningful title if possible
            title = uploaded_filename
            try:
                from core.models import get_llm
                from langchain_core.prompts import ChatPromptTemplate
                title_llm = get_llm(purpose="primary", temperature=0)
                title_prompt = ChatPromptTemplate.from_template(
                    "Generate a short, professional title (max 5 words) for a document with the following content summary:\n\n{summary}"
                )
                title_chain = title_prompt | title_llm
                # Use the first 2000 chars as a summary base if no better summary exists
                title_res = title_chain.invoke({"summary": raw_text[:2000]})
                title = title_res.content.strip().strip('"')
            except Exception as e:
                print(f"Title generation failed: {e}")

            # Save to DB
            new_doc = DBDocument(id=doc_id, subject_id=subject_id, filename=uploaded_filename, title=title, content_hash=content_hash)
            db.add(new_doc)
            db.commit()
        finally:
            db.close()

        # Create LangChain chunks
        chunks = self.text_splitter.split_text(raw_text)
        
        # 1. Save to Relational DB for source reference mapping
        db = SessionLocal()
        saved_chunk_ids = []
        try:
            for chunk_text in chunks:
                db_chunk = ContentChunk(document_id=doc_id, text=chunk_text)
                db.add(db_chunk)
                db.commit()
                db.refresh(db_chunk)
                saved_chunk_ids.append(db_chunk.id)
        finally:
             db.close()
             
        # 2. Prepare LangChain Document objects with Relational DB IDs as metadata
        lc_documents = []
        for i, chunk_text in enumerate(chunks):
            doc = Document(
                page_content=chunk_text,
                metadata={"document_id": doc_id, "db_chunk_id": saved_chunk_ids[i]}
            )
            lc_documents.append(doc)
            
        # 3. Ingest into Vector DB (Qdrant)
        QdrantVectorStore.from_documents(
            lc_documents,
            self.embeddings,
            url=settings.QDRANT_URL,
            collection_name=settings.QDRANT_COLLECTION_NAME
        )
        
        return lc_documents
