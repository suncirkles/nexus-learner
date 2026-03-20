"""
agents/topic_parser.py
-----------------------
Parses user-supplied topic input into a clean, deduplicated list of topic
strings.  Accepts free-form text (from a textarea), as well as .txt, .pdf,
and .docx file uploads.

For PDF extraction the existing IngestionAgent.extract_text_from_pdf() is
reused so we don't duplicate that logic.
"""

import json
import logging
import os
from typing import List

from core.models import get_llm, invoke_with_retry

logger = logging.getLogger(__name__)

# LLM prompt for extracting topics from arbitrary text
_EXTRACT_PROMPT = (
    "Extract a clean list of topics/subjects from the following text.\n"
    "Return ONLY a valid JSON array of strings, deduplicated, max 50 items.\n"
    "PRESERVE the original casing of technically significant terms (e.g., 'ANN', 'ML').\n"
    "Normalize other names only for consistency if needed, but do not force Title Case on abbreviations.\n"
    "Text: {text}"
)


class TopicParserAgent:
    """Converts user-supplied topic text or files into a clean list of strings."""

    def __init__(self):
        self._llm = get_llm(purpose="routing", temperature=0.0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def parse_topics_from_text(self, text: str) -> List[str]:
        """Use an LLM to extract a clean, deduplicated list of topics.

        Handles bullet points, numbered lists, comma-separated values,
        newline-separated values, and paragraph form.
        Returns at most 50 topics.
        """
        if not text or not text.strip():
            return []

        # Truncate very long inputs to avoid token explosion
        truncated = text.strip()[:6000]
        from core.context import get_langchain_config
        from scripts.model_hop import is_quota_error
        for _attempt in range(2):
            try:
                self._llm = get_llm(purpose="routing", temperature=0.0)
                response = invoke_with_retry(self._llm.invoke, prompt, config=get_langchain_config())
                return self._parse_json_array(response.content)
            except Exception as exc:
                if is_quota_error(exc) and _attempt == 0:
                    continue
                logger.error("Topic extraction from text failed: %s", exc)
                return self._fallback_parse(text)

    def parse_topics_from_file(self, file_path: str, file_type: str) -> List[str]:
        """Extract topics from a .txt, .pdf, or .docx file.

        Args:
            file_path: Absolute path to the file on disk.
            file_type: One of "txt", "pdf", "docx".

        Returns:
            List of topic strings.
        """
        raw_text = self._extract_text(file_path, file_type)
        if not raw_text:
            return []
        return self.parse_topics_from_text(raw_text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_text(self, file_path: str, file_type: str) -> str:
        """Dispatch to the correct extraction method based on file type."""
        file_type_lower = file_type.lower().lstrip(".")

        if file_type_lower == "txt":
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                    return fh.read()
            except Exception as exc:
                logger.error("Failed to read text file '%s': %s", file_path, exc)
                return ""

        elif file_type_lower == "pdf":
            try:
                # Reuse the existing IngestionAgent PDF extractor to avoid duplication
                from agents.ingestion import IngestionAgent
                agent = IngestionAgent()
                return agent.extract_text_from_pdf(file_path)
            except Exception as exc:
                logger.error("Failed to extract text from PDF '%s': %s", file_path, exc)
                return ""

        elif file_type_lower == "docx":
            try:
                import docx  # python-docx
                doc = docx.Document(file_path)
                paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
                return "\n".join(paragraphs)
            except Exception as exc:
                logger.error("Failed to extract text from DOCX '%s': %s", file_path, exc)
                return ""

        else:
            logger.warning("Unsupported file type for topic parsing: %s", file_type)
            return ""

    @staticmethod
    def _parse_json_array(text: str) -> List[str]:
        """Parse the LLM's JSON array response, tolerating markdown fences."""
        cleaned = text.strip()
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        data = json.loads(cleaned)
        if isinstance(data, list):
            # Ensure all items are non-empty strings
            topics = [str(item).strip() for item in data if str(item).strip()]
            return topics[:50]
        return []

    @staticmethod
    def _fallback_parse(text: str) -> List[str]:
        """Simple heuristic fallback when LLM JSON parsing fails."""
        import re
        lines = re.split(r"[\n,;]+", text)
        topics = []
        seen = set()
        for line in lines:
            # Strip leading bullets, numbers, and whitespace
            cleaned = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                topics.append(cleaned)
            if len(topics) >= 50:
                break
        return topics
