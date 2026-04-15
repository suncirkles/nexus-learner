"""
agents/ingestion.py
--------------------
Responsibility: Extract raw text from documents (PDF via PyMuPDF, OCR fallback via
Tesseract), split into chunks, deduplicate via content hash, persist to the relational
DB, and embed chunks into Qdrant or PGVector.

Do Not:
- Classify, label, or assign topic/subtopic information to chunks (TopicAssignerAgent).
- Generate flashcards or evaluate content quality (SocraticAgent / CriticAgent).
- Scrape or fetch content from URLs (WebResearcherAgent).
- Make decisions about whether a chunk is relevant to a study topic (RelevanceAgent).
- Interpret the educational meaning of the text — treat it as opaque bytes to be stored.
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os
import hashlib
import logging
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import settings
from repositories.sql.document_repo import DocumentRepo
from repositories.vector.factory import get_vector_store

logger = logging.getLogger(__name__)

# Point pytesseract at the Windows installer path if tesseract isn't on PATH
_TESSERACT_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_TESSERACT_WIN):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_WIN


class IngestionAgent:
    def __init__(self):
        self._vector_store = get_vector_store()
        self.embeddings = self._vector_store.embeddings
        self.collection_name = self._vector_store.collection_name
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
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

    def load_page_text(self, file_path: str, page_number: int) -> str:
        """Loads text from a specific page of a PDF, with OCR fallback."""
        if not file_path.lower().endswith('.pdf'):
            return ""
        
        try:
            with fitz.open(file_path) as doc:
                page = doc.load_page(page_number)
                text = page.get_text()
                
                # If no text found, try OCR
                if not text.strip():
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img)
                    
            return text
        except Exception as e:
            logger.error(f"Failed to load text from page {page_number} of {file_path}: {e}")
            return ""

    def get_content_hash(self, text: str) -> str:
        """Returns a stable SHA-256 hash of the text content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def create_document_record(self, file_path: str, subject_id: Optional[int] = None) -> str:
        """Creates a Document record and associates it with a Subject."""
        # Use a sample of text + file size + filename for the hash
        sample_text = self.load_page_text(file_path, 0)[:10000]
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        raw_basename = os.path.basename(file_path)
        # Strip the UUID prefix added by upload_and_spawn: "{uuid4}_{original_name}"
        # UUID4 is exactly 36 chars with 4 hyphens; the separator is an underscore.
        _parts = raw_basename.split("_", 1)
        if len(_parts) == 2 and len(_parts[0]) == 36 and _parts[0].count("-") == 4:
            filename = _parts[1]
        else:
            filename = raw_basename

        content_hash = self.get_content_hash(sample_text + str(file_size) + filename)

        repo = DocumentRepo()
        doc = repo.get_by_content_hash(content_hash)

        if not doc:
            import uuid
            doc_id = str(uuid.uuid4())
            doc = repo.create(
                doc_id=doc_id,
                filename=filename,
                title=filename,
                content_hash=content_hash
            )
        
        if subject_id:
            repo.attach_to_subject(doc["id"], subject_id)
            
        return doc["id"]

    def process_and_store(self, text: str, document_id: str, page_number: Optional[int] = None) -> List[int]:
        """Chunks text, persists to SQL, and embeds into the vector store."""
        chunks = self.text_splitter.split_text(text)
        
        from core.database import SessionLocal, ContentChunk
        db = SessionLocal()
        chunk_ids = []
        vector_data = []

        try:
            for text_chunk in chunks:
                db_chunk = ContentChunk(
                    document_id=document_id,
                    text=text_chunk,
                    page_number=page_number
                )
                db.add(db_chunk)
                db.commit()
                db.refresh(db_chunk)
                chunk_ids.append(db_chunk.id)
                
                vector_data.append({
                    "text": text_chunk,
                    "metadata": {
                        "document_id": document_id,
                        "db_chunk_id": db_chunk.id,
                        "page_number": page_number
                    }
                })
            
            # Use unified vector store for upsert
            self._vector_store.upsert_chunks(vector_data)
            
            return chunk_ids
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to process and store chunks for doc {document_id}: {e}")
            raise
        finally:
            db.close()
