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
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from core.config import settings
import hashlib
import logging
from core.database import SessionLocal, ContentChunk, Document as DBDocument

logger = logging.getLogger(__name__)

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

    def get_page_count(self, file_path: str) -> int:
        """Returns the total number of pages in a PDF."""
        if not file_path.lower().endswith('.pdf'):
            return 1
        with fitz.open(file_path) as doc:
            count = len(doc)
        return count

    def extract_text_from_image(self, file_path: str) -> str:
        """Extracts text from an image file using Tesseract OCR."""
        try:
            img = Image.open(file_path)
            return pytesseract.image_to_string(img)
        except Exception as e:
            logger.warning(f"OCR failed for image {file_path}: {e}")
            return ""

    def load_page_text(self, file_path: str, page_num: int) -> str:
        """Extracts text from a single PDF page with OCR fallback."""
        if not file_path.lower().endswith('.pdf'):
            if page_num > 0: return ""
            return self.extract_text_from_image(file_path)
            
        with fitz.open(file_path) as doc:
            if page_num >= len(doc):
                return ""
                
            page = doc.load_page(page_num)
            text = page.get_text("text")
            
            if len(text.strip()) < 50:
                try:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img)
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num}: {e}")
            
        return text

    def get_content_hash(self, text: str) -> str:
        """Generates a SHA256 hash of the text content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def create_document_record(self, file_path: str, doc_id: str, subject_id: int = None) -> DBDocument:
        """Creates a document record in the database if it doesn't exist."""
        sample_text = self.load_page_text(file_path, 0)[:10000]
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        content_hash = self.get_content_hash(sample_text + str(file_size) + os.path.basename(file_path))
        
        db = SessionLocal()
        try:
            existing_doc = db.query(DBDocument).filter(DBDocument.content_hash == content_hash).first()
            if existing_doc:
                # If it exists, just return it
                return existing_doc
            
            uploaded_filename = os.path.basename(file_path)
            title = uploaded_filename
            try:
                from core.models import get_llm
                from langchain_core.prompts import ChatPromptTemplate
                title_llm = get_llm(purpose="primary", temperature=0)
                title_prompt = ChatPromptTemplate.from_template(
                    "Generate a short, professional title (max 5 words) for a document with the following content sample:\n\n{summary}"
                )
                title_res = (title_prompt | title_llm).invoke({"summary": sample_text[:2000]})
                title = title_res.content.strip().strip('"')
            except Exception as e:
                logger.warning(f"Title generation failed: {e}")

            new_doc = DBDocument(id=doc_id, filename=uploaded_filename, title=title, content_hash=content_hash)
            db.add(new_doc)
            
            # If a subject_id is provided, create an association
            if subject_id:
                from core.database import SubjectDocumentAssociation
                # Check for existing association
                existing_assoc = db.query(SubjectDocumentAssociation).filter(
                    SubjectDocumentAssociation.subject_id == subject_id,
                    SubjectDocumentAssociation.document_id == new_doc.id
                ).first()
                if not existing_assoc:
                    new_assoc = SubjectDocumentAssociation(subject_id=subject_id, document_id=new_doc.id)
                    db.add(new_assoc)

            db.commit()
            db.refresh(new_doc)
            return new_doc
        finally:
            db.close()

    def ingest_text_chunk(self, doc_id: str, text: str) -> List[Document]:
        """Processes a single chunk of text: splitting, DB save, and Vector DB embedding."""
        chunks = self.text_splitter.split_text(text)
        if not chunks:
            return []

        # 1. Save to Relational DB
        db = SessionLocal()
        lc_documents = []
        try:
            for chunk_text in chunks:
                db_chunk = ContentChunk(document_id=doc_id, text=chunk_text)
                db.add(db_chunk)
                db.commit()
                db.refresh(db_chunk)
                
                doc = Document(
                    page_content=chunk_text,
                    metadata={"document_id": doc_id, "db_chunk_id": db_chunk.id}
                )
                lc_documents.append(doc)
            
            # 2. Ingest into Vector DB
            QdrantVectorStore.from_documents(
                lc_documents,
                self.embeddings,
                url=settings.QDRANT_URL,
                collection_name=settings.QDRANT_COLLECTION_NAME
            )
        finally:
             db.close()
             
        return lc_documents
