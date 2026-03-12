"""
agents/safety.py
-----------------
Safety guardrail agent.  Screens subjects and scraped web content for
harmful, hateful, sexual, or dangerous material before any content is
ingested into the knowledge base.  All LLM calls use the fast routing
model (gpt-4o-mini) at temperature 0 to ensure consistent, deterministic
classification decisions.
"""

import json
import logging
from typing import Optional

from pydantic import BaseModel
from core.models import get_llm
from core.config import settings

logger = logging.getLogger(__name__)


class SafetyResult(BaseModel):
    """Structured output for every safety evaluation."""
    is_safe: bool
    reason: str
    filtered_text: str = ""


class SafetyAgent:
    """LLM-powered safety classifier used by the web ingestion pipeline."""

    # Prompt for subject-level screening
    _SUBJECT_PROMPT = (
        "You are a content safety classifier. Determine if the following subject is appropriate "
        "for an educational platform. Block subjects that involve: hate speech, violence, sexual content, "
        "illegal activities, self-harm, or dangerous/harmful material.\n"
        "Subject: {subject_name}\n"
        'Return ONLY valid JSON with this exact schema: {{"is_safe": bool, "reason": str}}'
    )

    # Prompt for per-page content screening
    _CONTENT_PROMPT = (
        "You are a content safety classifier for an educational platform. Analyze this text and:\n"
        "1. Determine if it contains harmful, hateful, sexual, violent, or illegal content\n"
        "2. Determine if it is relevant to educational/technical learning\n"
        "3. If safe and relevant, return the cleaned text; if not, return empty string\n"
        "Text: {text}\n"
        'Return ONLY valid JSON with this exact schema: {{"is_safe": bool, "reason": str, "filtered_text": str}}'
    )

    # Prompt for per-topic relevance check
    _RELEVANCE_PROMPT = (
        "You are a relevance classifier. Determine whether the following topic is directly "
        "related to the given subject for an educational course.\n"
        "Subject: {subject_name}\n"
        "Topic: {topic}\n"
        'Return ONLY valid JSON with this exact schema: {{"is_relevant": bool, "reason": str}}'
    )

    def __init__(self):
        self._llm = get_llm(purpose="routing", temperature=0.0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def check_subject_safety(self, subject_name: str) -> SafetyResult:
        """Determine whether the subject itself is safe to research.

        If ``CONTENT_SAFETY_ENABLED`` is False in settings the check is
        bypassed and every subject is treated as safe.
        """
        if not settings.CONTENT_SAFETY_ENABLED:
            return SafetyResult(is_safe=True, reason="Safety checks disabled.")

        prompt = self._SUBJECT_PROMPT.format(subject_name=subject_name)
        try:
            response = self._llm.invoke(prompt)
            data = self._parse_json(response.content)
            return SafetyResult(
                is_safe=bool(data.get("is_safe", False)),
                reason=str(data.get("reason", "")),
            )
        except Exception as exc:
            logger.error("Subject safety check failed for '%s': %s", subject_name, exc)
            # Default to SAFE on classifier error so we don't silently block every request
            return SafetyResult(is_safe=True, reason=f"Safety check error (defaulting to safe): {exc}")

    def check_content_safety(self, text: str, source_url: Optional[str] = None) -> SafetyResult:
        """Screen a scraped page for harmful content.

        Returns a ``SafetyResult`` where ``filtered_text`` holds the
        cleaned content when ``is_safe`` is True, or an empty string
        when the content is blocked.
        """
        if not settings.CONTENT_SAFETY_ENABLED:
            return SafetyResult(is_safe=True, reason="Safety checks disabled.", filtered_text=text)

        # Limit text sent to LLM to avoid huge token bills
        truncated = text[:4000]
        prompt = self._CONTENT_PROMPT.format(text=truncated)
        try:
            response = self._llm.invoke(prompt)
            data = self._parse_json(response.content)
            is_safe = bool(data.get("is_safe", False))
            return SafetyResult(
                is_safe=is_safe,
                reason=str(data.get("reason", "")),
                filtered_text=text if is_safe else "",
            )
        except Exception as exc:
            logger.error("Content safety check failed for '%s': %s", source_url or "unknown", exc)
            # Default to UNSAFE on classifier error — better to skip a page than ingest bad content
            return SafetyResult(
                is_safe=False,
                reason=f"Safety check error (defaulting to unsafe): {exc}",
                filtered_text="",
            )

    def check_topic_relevance(self, topic: str, subject_name: str) -> bool:
        """Quick relevance check: is this topic actually related to the subject?"""
        if not settings.CONTENT_SAFETY_ENABLED:
            return True

        prompt = self._RELEVANCE_PROMPT.format(subject_name=subject_name, topic=topic)
        try:
            response = self._llm.invoke(prompt)
            data = self._parse_json(response.content)
            return bool(data.get("is_relevant", True))
        except Exception as exc:
            logger.error("Topic relevance check failed for '%s': %s", topic, exc)
            # Default to relevant on error so we don't drop valid topics silently
            return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract JSON from the LLM response, tolerating markdown code fences."""
        cleaned = text.strip()
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove opening fence (```json or ```)
            lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return json.loads(cleaned)
