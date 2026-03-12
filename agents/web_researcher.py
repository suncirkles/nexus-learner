"""
agents/web_researcher.py
-------------------------
Web content research agent. Uses a multi-source strategy for reliability:

  1. Wikipedia REST API  — primary, free, no key, plain-text content via API
  2. ddgs (DuckDuckGo)  — broad search filtered to trusted domains
  3. Parallel scraping  — 4 concurrent workers for non-Wikipedia pages
  4. Inherently-safe tier — skips LLM safety check for official docs
"""

import hashlib
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from urllib.parse import urlparse, quote_plus

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

from agents.safety import SafetyAgent
from core.config import settings
from core.database import SessionLocal, Document as DBDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trusted educational sources (priority-ordered, max 10)
# ---------------------------------------------------------------------------
TRUSTED_SOURCES: List[str] = [
    "wikipedia.org",
    "geeksforgeeks.org",
    "freecodecamp.org",
    "developer.mozilla.org",
    "docs.python.org",
    "learn.microsoft.com",
    "docs.aws.amazon.com",
    "pytorch.org",
    "docs.docker.com",
    "scikit-learn.org",
]

# Domains that are inherently safe — skip the per-page LLM safety check
_INHERENTLY_SAFE: frozenset = frozenset([
    "wikipedia.org",
    "docs.python.org",
    "developer.mozilla.org",
    "docs.aws.amazon.com",
    "pytorch.org",
    "scikit-learn.org",
    "docs.docker.com",
    "learn.microsoft.com",
])

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_HEADERS = {"User-Agent": "NexusLearner/1.0 (educational app; contact: admin@example.com)"}


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    domain: str


class WebDocument(BaseModel):
    topic: str
    url: str
    title: str
    domain: str
    content: str
    content_hash: str
    is_safe: bool
    safety_reason: str


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class WebResearchAgent:
    """Searches trusted educational sources and scrapes clean content."""

    _SCRAPE_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self):
        self._safety = SafetyAgent()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def research_topics(
        self,
        topics: List[str],
        subject_name: str,
        subject_id: int,
        status_callback=None,
        stop_event: Optional[threading.Event] = None,
    ) -> List[WebDocument]:
        """Research topics and return scraped, screened WebDocument objects."""
        results: List[WebDocument] = []
        seen_hashes: set = self._load_existing_hashes(subject_id)
        total = len(topics)

        for i, topic in enumerate(topics):
            if stop_event is not None and stop_event.is_set():
                logger.info("Stop event — halting web research.")
                break

            if status_callback:
                status_callback(f"🔍 Topic {i + 1}/{total}: **{topic}**")

            # 1. Relevance check
            if not self._safety.check_topic_relevance(topic, subject_name):
                logger.info("Skipping irrelevant topic: %s", topic)
                if status_callback:
                    status_callback(f"⏭ Skipped (not related to '{subject_name}'): {topic}")
                continue

            query = f"{topic} {subject_name}"

            # 2. Find candidate pages (Wikipedia API + DDG)
            if status_callback:
                status_callback(f"🌐 Searching for: *{query}*")

            search_results = self._find_candidates(query)

            if not search_results:
                if status_callback:
                    status_callback(f"⚠ No results found for: **{topic}**")
                continue

            if status_callback:
                status_callback(
                    f"📄 Found {len(search_results)} candidate(s) — fetching content..."
                )

            # 3. Fetch content (Wikipedia via API, others via parallel scraping)
            wiki_results = [sr for sr in search_results if "wikipedia.org" in sr.domain]
            other_results = [sr for sr in search_results if "wikipedia.org" not in sr.domain]

            content_map: Dict[str, Optional[str]] = {}

            # Wikipedia: get plain text directly from API (fast, reliable)
            for sr in wiki_results:
                title = sr.title
                text = self._get_wikipedia_content(title)
                content_map[sr.url] = text

            # Others: parallel HTTP scraping
            if other_results:
                scraped = self._scrape_parallel(other_results)
                content_map.update(scraped)

            # 4. Process each result
            pages_collected = 0
            for sr in search_results:
                if pages_collected >= settings.WEB_MAX_PAGES_PER_TOPIC:
                    break

                raw_content = content_map.get(sr.url)
                if not raw_content:
                    continue

                content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    logger.info("Duplicate skipped: %s", sr.url)
                    continue

                # 5. Safety screening (skip for inherently safe domains)
                is_inherently_safe = any(safe in sr.domain for safe in _INHERENTLY_SAFE)
                if is_inherently_safe:
                    final_content = raw_content
                    safety_reason = "Trusted official source"
                else:
                    safety_result = self._safety.check_content_safety(
                        raw_content, source_url=sr.url
                    )
                    if not safety_result.is_safe:
                        if status_callback:
                            status_callback(f"🚫 Blocked (safety): {sr.domain}")
                        continue
                    final_content = safety_result.filtered_text or raw_content
                    safety_reason = safety_result.reason

                seen_hashes.add(content_hash)
                pages_collected += 1

                results.append(WebDocument(
                    topic=topic,
                    url=sr.url,
                    title=sr.title,
                    domain=sr.domain,
                    content=final_content[: settings.WEB_MAX_CONTENT_CHARS],
                    content_hash=content_hash,
                    is_safe=True,
                    safety_reason=safety_reason,
                ))

                if status_callback:
                    status_callback(f"✅ **{sr.domain}** — {sr.title[:70]}")

        return results

    # ------------------------------------------------------------------
    # Source 1: Wikipedia API
    # ------------------------------------------------------------------

    def _search_wikipedia(self, query: str, max_results: int = 3) -> List[SearchResult]:
        """Search Wikipedia articles via the free MediaWiki API."""
        try:
            resp = requests.get(
                _WIKI_API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": max_results,
                    "srnamespace": 0,
                },
                headers=_WIKI_HEADERS,
                timeout=8,
            )
            resp.raise_for_status()
            items = resp.json().get("query", {}).get("search", [])
            results = []
            for item in items:
                title = item["title"]
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                results.append(SearchResult(url=url, title=title, snippet=snippet, domain="wikipedia.org"))
            return results
        except Exception as exc:
            logger.warning("Wikipedia search API failed: %s", exc)
            return []

    def _get_wikipedia_content(self, title: str) -> Optional[str]:
        """Fetch Wikipedia article as plain text via the MediaWiki API — no scraping."""
        try:
            resp = requests.get(
                _WIKI_API,
                params={
                    "action": "query",
                    "titles": title,
                    "prop": "extracts",
                    "explaintext": True,
                    "exsectionformat": "plain",
                    "format": "json",
                },
                headers=_WIKI_HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                text = page.get("extract", "")
                if text and len(text) > 200:
                    # Trim overly long articles
                    return text[: settings.WEB_MAX_CONTENT_CHARS]
        except Exception as exc:
            logger.warning("Wikipedia content API failed for '%s': %s", title, exc)
        return None

    # ------------------------------------------------------------------
    # Source 2: DuckDuckGo (ddgs package)
    # ------------------------------------------------------------------

    def _search_ddgs(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Broad DuckDuckGo search filtered to trusted domains."""
        results: List[SearchResult] = []
        seen_urls: set = set()
        trusted_set = set(TRUSTED_SOURCES)

        try:
            from ddgs import DDGS
            ddgs = DDGS()

            # Step 1: broad search, filter to trusted domains
            try:
                raw = ddgs.text(query, max_results=20)
                for item in raw or []:
                    url = item.get("href") or item.get("url", "")
                    if not url or url in seen_urls:
                        continue
                    domain = urlparse(url).netloc.replace("www.", "")
                    if any(t in domain for t in trusted_set) and "wikipedia.org" not in domain:
                        seen_urls.add(url)
                        results.append(SearchResult(
                            url=url,
                            title=item.get("title", url),
                            snippet=item.get("body", ""),
                            domain=domain,
                        ))
                        if len(results) >= max_results:
                            break
            except Exception as exc:
                logger.debug("DDG broad search failed: %s", exc)

            # Step 2: if sparse, try site: on top 3 non-Wikipedia domains
            if len(results) < 2:
                for domain in [d for d in TRUSTED_SOURCES if "wikipedia" not in d][:3]:
                    if len(results) >= max_results:
                        break
                    try:
                        raw = ddgs.text(f"site:{domain} {query}", max_results=2)
                        for item in raw or []:
                            url = item.get("href") or item.get("url", "")
                            if not url or url in seen_urls:
                                continue
                            seen_urls.add(url)
                            parsed_domain = urlparse(url).netloc.replace("www.", "")
                            results.append(SearchResult(
                                url=url,
                                title=item.get("title", url),
                                snippet=item.get("body", ""),
                                domain=parsed_domain,
                            ))
                        time.sleep(0.15)
                    except Exception as exc:
                        logger.debug("DDG site: search failed for %s: %s", domain, exc)

        except ImportError:
            logger.error("ddgs not installed. Run: pip install ddgs")

        return results

    # ------------------------------------------------------------------
    # Combined candidate finder
    # ------------------------------------------------------------------

    def _find_candidates(self, query: str) -> List[SearchResult]:
        """Return candidates from Wikipedia API + DDG, deduped, up to WEB_SEARCH_MAX_RESULTS."""
        results: List[SearchResult] = []
        seen_urls: set = set()

        # Wikipedia first (most reliable)
        for sr in self._search_wikipedia(query, max_results=2):
            if sr.url not in seen_urls:
                seen_urls.add(sr.url)
                results.append(sr)

        # DDG for other trusted sources
        remaining = settings.WEB_SEARCH_MAX_RESULTS - len(results)
        if remaining > 0:
            for sr in self._search_ddgs(query, max_results=remaining + 3):
                if sr.url not in seen_urls and len(results) < settings.WEB_SEARCH_MAX_RESULTS:
                    seen_urls.add(sr.url)
                    results.append(sr)

        return results[: settings.WEB_SEARCH_MAX_RESULTS]

    # ------------------------------------------------------------------
    # Parallel scraping (for non-Wikipedia pages)
    # ------------------------------------------------------------------

    def _scrape_parallel(self, search_results: List[SearchResult]) -> Dict[str, Optional[str]]:
        """Scrape multiple URLs concurrently."""
        contents: Dict[str, Optional[str]] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            future_to_url = {
                pool.submit(self._scrape_page, sr.url): sr.url
                for sr in search_results
            }
            for future in as_completed(future_to_url, timeout=60):
                url = future_to_url[future]
                try:
                    contents[url] = future.result()
                except Exception as exc:
                    logger.debug("Scrape failed for %s: %s", url, exc)
                    contents[url] = None
        return contents

    def _scrape_page(self, url: str) -> Optional[str]:
        """Fetch a page and extract clean plaintext."""
        try:
            response = requests.get(
                url,
                headers=self._SCRAPE_HEADERS,
                timeout=settings.WEB_SCRAPE_TIMEOUT,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.warning("Timeout: %s", url)
            return None
        except requests.exceptions.RequestException as exc:
            logger.warning("HTTP error '%s': %s", url, exc)
            return None

        try:
            soup = BeautifulSoup(response.text, "lxml")
        except Exception:
            try:
                soup = BeautifulSoup(response.text, "html.parser")
            except Exception as exc:
                logger.warning("Parse error '%s': %s", url, exc)
                return None

        for tag in soup(["nav", "footer", "header", "aside", "script", "style", "noscript"]):
            tag.decompose()

        content_tag = self._select_content_element(soup, url)
        if content_tag is None:
            return None

        raw_text = content_tag.get_text(separator=" ", strip=True)
        raw_text = re.sub(r"\s{2,}", " ", raw_text).strip()

        return raw_text[: settings.WEB_MAX_CONTENT_CHARS] if len(raw_text) >= 200 else None

    @staticmethod
    def _select_content_element(soup: BeautifulSoup, url: str):
        hostname = urlparse(url).netloc.lower().replace("www.", "")

        if "geeksforgeeks.org" in hostname:
            for sel in [".article-body", ".entry-content", ".text"]:
                tag = soup.select_one(sel)
                if tag:
                    return tag

        if "freecodecamp.org" in hostname:
            for sel in [".post-content", "article"]:
                tag = soup.select_one(sel)
                if tag:
                    return tag

        if "developer.mozilla.org" in hostname:
            tag = soup.select_one("#content article") or soup.select_one("article")
            if tag:
                return tag

        for tag_name in ["main", "article"]:
            tag = soup.find(tag_name)
            if tag:
                return tag

        best_div, best_len = None, 0
        for div in soup.find_all("div"):
            text_len = len(div.get_text(strip=True))
            if text_len > best_len:
                best_len = text_len
                best_div = div
        return best_div

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_existing_hashes(subject_id: int) -> set:
        db = SessionLocal()
        try:
            rows = db.query(DBDocument.content_hash).filter(
                DBDocument.subject_id == subject_id,
                DBDocument.content_hash.isnot(None),
            ).all()
            return {row[0] for row in rows}
        except Exception as exc:
            logger.error("Failed to load existing hashes: %s", exc)
            return set()
        finally:
            db.close()
