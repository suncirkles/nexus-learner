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
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from core.config import settings
import hashlib
import logging
from repositories.sql.document_repo import DocumentRepo

logger = logging.getLogger(__name__)

# Point pytesseract at the Windows installer path if tesseract isn't on PATH
_TESSERACT_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_TESSERACT_WIN):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_WIN


def _is_real_openai_key(key: str) -> bool:
    """True if the key looks like a real OpenAI key, not a placeholder."""
    return bool(key) and key.startswith("sk-") and len(key) > 20


def _make_embeddings():
    """Return (embeddings, collection_name).

    Provider is controlled by settings.EMBEDDING_PROVIDER:
    - "openai"      : OpenAIEmbeddings (requires a valid key, 1536-dim collection)
    - "huggingface" : local all-MiniLM-L6-v2 (no key, 384-dim, "_hf" collection suffix)

    If provider="openai" but the key is absent/placeholder, auto-falls back to
    HuggingFace with a warning so CI stays green without an OpenAI account.
    """
    use_hf = settings.EMBEDDING_PROVIDER.lower() == "huggingface"

    if not use_hf:
        if _is_real_openai_key(settings.OPENAI_API_KEY):
            return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY), settings.QDRANT_COLLECTION_NAME
        logger.warning(
            "EMBEDDING_PROVIDER=openai but key is absent/invalid — "
            "falling back to HuggingFace embeddings"
        )

    try:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]
        logger.info("Using HuggingFace all-MiniLM-L6-v2 embeddings (384 dims)")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        collection = settings.QDRANT_COLLECTION_NAME + "_hf"
        return embeddings, collection
    except Exception as e:
        raise ValueError(
            "HuggingFace embeddings unavailable. "
            "Install sentence-transformers: pip install sentence-transformers. "
            f"Original error: {e}"
        )


class IngestionAgent:
    def __init__(self):
        self.embeddings, self.collection_name = _make_embeddings()
        
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

    def create_document_record(self, file_path: str, doc_id: str, subject_id: Optional[int] = None) -> dict:
        """Creates a document record in the database if it doesn't exist.

        Returns a dict with document fields (id, filename, title, content_hash, …).
        Uses DocumentRepo — no direct SessionLocal import in the agent.
        """
        sample_text = self.load_page_text(file_path, 0)[:10000]
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        content_hash = self.get_content_hash(sample_text + str(file_size) + os.path.basename(file_path))

        doc_repo = DocumentRepo()
        existing = doc_repo.get_by_content_hash(content_hash)
        if existing:
            return existing

        uploaded_filename = os.path.basename(file_path)
        title = uploaded_filename
        try:
            from core.context import get_langchain_config
            title_llm = get_llm(purpose="primary", temperature=0)
            title_prompt = ChatPromptTemplate.from_template(
                "Generate a short, professional title (max 5 words) for a document with the following content sample:\n\n{summary}"
            )
            title_res = (title_prompt | title_llm).invoke(
                {"summary": sample_text[:2000]},
                config=get_langchain_config()
            )
            title = title_res.content.strip().strip('"')
        except Exception as e:
            logger.warning(f"Title generation failed: {e}")

        new_doc = doc_repo.create(
            doc_id=doc_id,
            filename=uploaded_filename,
            title=title,
            content_hash=content_hash,
        )
        if subject_id:
            doc_repo.attach_to_subject(new_doc["id"], subject_id)
        return new_doc

    def ingest_text_chunk(self, doc_id: str, text: str) -> List[str]:
        """Splits text into chunks. Pure — no DB writes, no Qdrant.

        Phase 2b: DB persistence is the caller's responsibility (use ChunkRepo.create_batch).
        Qdrant embedding is the caller's responsibility (use QdrantStore.upsert_chunks).
        """
        return self.text_splitter.split_text(text)
