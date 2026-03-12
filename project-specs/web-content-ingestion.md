# Feature Spec: Web Content Ingestion for Nexus Learner

## Overview

Extend the subject creation flow to allow users to specify topics (via PDF/doc upload, text file, or rich text box) and automatically gather high-quality content from authoritative web sources. The gathered content must follow the exact same pipeline as PDF uploads (chunking → topic/subtopic extraction → flashcard generation → critic evaluation). All content must be traceable to its source. Guardrails must block harmful, hateful, sexual, or irrelevant material — and prevent content gathering entirely if the subject itself is flagged.

---

## Tech Stack (existing — do not change)

- **Frontend**: Streamlit (`app.py`)
- **Backend**: Python, LangChain, LangGraph
- **LLM**: OpenAI `gpt-4o` (primary), `gpt-4o-mini` (routing), Anthropic fallback
- **Database**: SQLite via SQLAlchemy ORM (`core/database.py`)
- **Vector DB**: Qdrant (`nexus_chunks` collection)
- **Workflow**: LangGraph (`workflows/phase1_ingestion.py`)
- **Background tasks**: Python threading (`core/background.py`)
- **Config**: Pydantic Settings (`core/config.py`)
- **Agents**: `agents/ingestion.py`, `agents/curator.py`, `agents/socratic.py`, `agents/critic.py`

---

## Database Changes (`core/database.py`)

### Modify `Document` table

Add two new columns to the existing `Document` model:

```python
source_type: str  # "pdf" | "image" | "web" | "text"  (default: "pdf")
source_url: Optional[str]  # URL if source_type == "web", None otherwise
```

Also add `source_type` and `source_url` to the existing `ContentChunk` model so individual chunks are traceable to their origin.

### No new tables needed — source traceability lives on Document + ContentChunk.

---

## New Files to Create

### 1. `agents/safety.py` — Safety Guardrail Agent

```
Class: SafetyAgent

Methods:
  - check_subject_safety(subject_name: str) -> SafetyResult
      Uses LLM (routing model, low temperature) to determine if the subject itself is
      harmful/hateful/sexual/dangerous. Returns {is_safe: bool, reason: str}.
      If not safe, ALL content gathering is blocked for this subject.

  - check_content_safety(text: str, source_url: str = None) -> SafetyResult
      Screens raw web-scraped or uploaded text for harmful/hateful/sexual/irrelevant content.
      Returns {is_safe: bool, reason: str, filtered_text: str}.
      filtered_text is the cleaned content if is_safe=True, empty string if False.

  - check_topic_relevance(topic: str, subject_name: str) -> bool
      Quick relevance check: is this topic actually related to the subject?

Pydantic model:
  class SafetyResult(BaseModel):
      is_safe: bool
      reason: str
      filtered_text: str = ""

LLM Prompt for subject check:
  "You are a content safety classifier. Determine if the following subject is appropriate
  for an educational platform. Block subjects that involve: hate speech, violence, sexual content,
  illegal activities, self-harm, or dangerous/harmful material.
  Subject: {subject_name}
  Return JSON: {is_safe: bool, reason: str}"

LLM Prompt for content check:
  "You are a content safety classifier for an educational platform. Analyze this text and:
  1. Determine if it contains harmful, hateful, sexual, violent, or illegal content
  2. Determine if it is relevant to educational/technical learning
  3. If safe and relevant, return the cleaned text; if not, return empty string
  Text: {text}
  Return JSON: {is_safe: bool, reason: str, filtered_text: str}"
```

### 2. `agents/web_researcher.py` — Web Content Research Agent

```
Class: WebResearchAgent

Dependencies: requests, beautifulsoup4, duckduckgo-search (or serpapi)

TRUSTED_SOURCES (priority-ordered list):
  - "geeksforgeeks.org"
  - "freecodecamp.org"
  - "wikipedia.org"
  - "docs.python.org"
  - "developer.mozilla.org"
  - "learn.microsoft.com"
  - "docs.aws.amazon.com"
  - "cloud.google.com/docs"
  - "kubernetes.io/docs"
  - "docs.docker.com"
  - "reactjs.org"
  - "docs.djangoproject.com"
  - "pytorch.org/docs"
  - "tensorflow.org/api_docs"
  - "scikit-learn.org"

Methods:

  - research_topics(topics: List[str], subject_name: str, subject_id: int) -> List[WebDocument]
      For each topic:
        1. Call SafetyAgent.check_topic_relevance(topic, subject_name)
        2. Generate smart search queries using LLM
        3. Search trusted sources (prefer site: operator per domain)
        4. Scrape & clean content from top results
        5. Run SafetyAgent.check_content_safety() on each page
        6. Return WebDocument list with content + metadata

  - _generate_search_queries(topic: str, subject_name: str) -> List[str]
      Use LLM (routing model) to generate 2-3 diverse search queries for a topic.
      Example: topic="binary search trees", subject="Data Structures"
      → ["binary search tree implementation tutorial",
         "BST traversal algorithms explained",
         "binary search tree operations complexity"]

  - _search_trusted_sources(query: str) -> List[SearchResult]
      Uses DuckDuckGo search (ddg-search library) with site: operator
      to find relevant pages from TRUSTED_SOURCES list.
      Returns top 3 results per query (URL + title + snippet).

  - _scrape_page(url: str) -> Optional[str]
      HTTP GET with 10s timeout, User-Agent header.
      Parse with BeautifulSoup4: extract main content area only
      (remove nav, footer, ads, scripts, styles).
      For Wikipedia: extract only the article body (id="mw-content-text")
      For GeeksForGeeks: extract .article-body
      For freeCodeCamp: extract .post-content
      For generic: extract <main>, <article>, or largest <div> with most text.
      Return cleaned plaintext (no HTML, no code blocks stripped unless relevant).
      Limit to 8000 chars max per page to avoid token explosion.

  - _deduplicate_content(documents: List[WebDocument]) -> List[WebDocument]
      SHA-256 hash of content to remove duplicate pages.
      Also check against existing Document.content_hash in DB.

Pydantic models:
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
```

### 3. `agents/topic_parser.py` — Topic Input Parser

```
Class: TopicParserAgent

Methods:
  - parse_topics_from_text(text: str) -> List[str]
      Use LLM to extract a clean list of topics from free-form text.
      Handles:
        - Bullet points
        - Numbered lists
        - Comma-separated
        - Newline-separated
        - Paragraph form
      Returns deduplicated list of topic strings, max 50 topics.

  - parse_topics_from_file(file_path: str, file_type: str) -> List[str]
      For .txt: read raw text → parse_topics_from_text()
      For .pdf: use existing IngestionAgent.extract_text_from_pdf() → parse_topics_from_text()
      For .docx: use python-docx to extract text → parse_topics_from_text()
      Returns list of topic strings.

LLM Prompt:
  "Extract a clean list of topics/subjects from the following text.
  Return only the topic names as a JSON array of strings, deduplicated, max 50 items.
  Normalize topic names (title case, remove duplicates/near-duplicates).
  Text: {text}"
```

### 4. `workflows/phase2_web_ingestion.py` — New LangGraph Workflow

Mirrors `phase1_ingestion.py` exactly but for web-sourced content.

```python
GraphState (TypedDict):
    subject_id: int
    topics: List[str]               # Input topics from user
    subject_name: str
    web_documents: List[dict]       # WebDocument dicts from WebResearchAgent
    current_doc_index: int          # Which web document being processed
    doc_id: str                     # UUID for current document
    full_text: str                  # Content of current web document
    chunks: List[Any]               # LangChain Document objects
    hierarchy: List[Dict]           # Topic/Subtopic structure
    doc_summary: str
    current_chunk_index: int
    generated_flashcards: List[Dict]
    status_message: str
    safety_blocked: bool            # True if subject is unsafe
    safety_reason: str
    processed_urls: List[str]       # For deduplication tracking

Nodes (in order):
  1. node_safety_check()
     - Run SafetyAgent.check_subject_safety(subject_name)
     - If not safe: set safety_blocked=True, go to END
     - If safe: continue to node_research

  2. node_research()
     - Run WebResearchAgent.research_topics(topics, subject_name, subject_id)
     - Populate web_documents list
     - Set current_doc_index = 0
     - If no documents found: set status_message, go to END

  3. node_ingest_web_document()
     - Take web_documents[current_doc_index]
     - Create Document record in DB with:
         source_type = "web"
         source_url = document.url
         filename = document.title
         content_hash = document.content_hash
     - Check for duplicate content_hash (skip if exists)
     - Chunk content using RecursiveCharacterTextSplitter (same 1000/200 settings)
     - Save ContentChunk records with source_type="web", source_url=document.url
     - Embed chunks into Qdrant with metadata including source_url

  4. node_curate() — REUSE existing CuratorAgent unchanged

  5. node_generate() — REUSE existing SocraticAgent unchanged

  6. node_critic() — REUSE existing CriticAgent unchanged

  7. node_increment_chunk() — move to next chunk within current document

  8. node_next_document() — move to next web document, reset chunk index

Edges:
  START → safety_check
  safety_check → research (if safe) | END (if blocked)
  research → ingest_web_document (if docs found) | END (if no docs)
  ingest_web_document → curate → generate → critic → [chunk_loop or next_doc or END]

Conditional edges:
  after_critic:
    - if more chunks in current doc → increment_chunk → generate
    - elif more documents → next_document → ingest_web_document
    - else → END
```

---

## Modified Files

### `core/database.py` — Add source fields

Add to `Document` model:
```python
source_type: Mapped[str] = mapped_column(String(20), nullable=False, default="pdf")
source_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
```

Add to `ContentChunk` model:
```python
source_type: Mapped[str] = mapped_column(String(20), nullable=False, default="pdf")
source_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
```

Add migration: when app starts, check if columns exist (using `PRAGMA table_info`) and ALTER TABLE if missing (SQLite migration pattern).

### `core/config.py` — Add new settings

```python
WEB_SCRAPE_TIMEOUT: int = 10           # HTTP request timeout seconds
WEB_MAX_PAGES_PER_TOPIC: int = 3       # Max pages to scrape per topic
WEB_MAX_CONTENT_CHARS: int = 8000      # Max chars per scraped page
WEB_SEARCH_MAX_RESULTS: int = 5        # DuckDuckGo results per query
CONTENT_SAFETY_ENABLED: bool = True    # Enable safety guardrails
```

### `requirements.txt` — Add dependencies

```
beautifulsoup4>=4.12.0
duckduckgo-search>=6.0.0
python-docx>=1.1.0
lxml>=5.0.0
```

### `app.py` — UI Changes

**In `render_study_materials()` view:**

After the subject selector, add a **tabbed interface** with two tabs:
- Tab 1: "📄 Upload Document" (existing PDF/image upload — no changes)
- Tab 2: "🌐 Web Research" (new)

**Tab 2 — Web Research UI:**

1. **Topic Input Section** with three sub-options (radio buttons):
   - "✏️ Type topics" → `st.text_area("Enter topics (one per line, or separated by commas)", height=150)`
   - "📄 Upload topic file" → `st.file_uploader("Upload .txt, .pdf, or .docx", type=["txt","pdf","docx"])`
   - (Both options feed into TopicParserAgent)

2. **Parsed Topics Preview:**
   After input, show a `st.expander("📋 Topics identified (click to review)")` listing the parsed topics as chips/bullets. Allow user to remove individual topics before proceeding.

3. **Safety Status Indicator:**
   Before launch, show a small inline safety check result for the subject (green checkmark = safe, red = blocked).

4. **"🔍 Start Web Research" button:**
   - Disabled if no topics parsed or subject is safety-blocked
   - Shows a progress section when clicked

5. **Transparent Progress Display:**
   During processing, show:
   ```
   🔍 Researching: "Binary Search Trees"
     ✅ Found: GeeksForGeeks - "Binary Search Tree | Set 1 (Search and Insertion)"
     ✅ Found: Wikipedia - "Binary search tree"
     ⚠️  Skipped: [url] - content safety filter
   📥 Ingesting content from: geeksforgeeks.org (chunk 3/12)
   🧠 Extracting topics and subtopics...
   ✨ Generating flashcards... (8 generated so far)
   ```

6. **Results Summary:**
   After completion, show:
   - Sources used (expandable list with URLs, domain, title)
   - Topics/subtopics created
   - Flashcards generated
   - Any URLs skipped (with reason)

**In `render_mentor_review()` view — show source badge on flashcards:**

On each flashcard card, add a small source indicator below the critic score:
- If source_type == "web": show 🌐 + domain name + clickable URL
- If source_type == "pdf": show 📄 + filename
- If source_type == "image": show 🖼️ + filename

**In `render_learner()` view — show source on flashcard:**

Below each flashcard's answer, show a small "Source: [📄 filename]" or "Source: [🌐 domain.com]" attribution.

---

## Implementation Notes

### Web Scraping Strategy

Use `duckduckgo-search` library (no API key required) with this query pattern:
```python
f'site:{domain} {query}'
```

Iterate through TRUSTED_SOURCES list. For each topic, collect up to `WEB_MAX_PAGES_PER_TOPIC` unique pages across all trusted sources.

### Content Extraction (BeautifulSoup4)

Priority selectors per domain:
- `wikipedia.org`: `#mw-content-text .mw-parser-output`
- `geeksforgeeks.org`: `.article-body`, `.entry-content`
- `freecodecamp.org`: `.post-content`, `article`
- Generic fallback: `<main>`, `<article>`, or largest `<div>` by text length

Always:
- Remove `<nav>`, `<footer>`, `<header>`, `<aside>`, `<script>`, `<style>`
- Extract text only (no HTML tags in output)
- Normalize whitespace
- Truncate to `WEB_MAX_CONTENT_CHARS`

### Error Handling

- HTTP errors (4xx, 5xx): log and skip URL, continue with next
- Timeout: log and skip, continue with next
- Parse error: log and skip, continue with next
- Safety block on content: log reason, skip document, add to skipped_urls list
- Zero results after all searches: show clear user message "No content found for topic X on trusted sources"

### Background Processing

Web research (HTTP requests) runs synchronously in the UI for the first topic for immediate feedback, then continues in background thread for remaining topics — same pattern as existing PDF background processing via `core/background.py`.

### Deduplication

- Use SHA-256 hash of scraped content
- Check against `Document.content_hash` before creating new Document record
- Skip if already ingested (same message as PDF deduplication)

### Existing Pipeline Reuse

The curator, socratic, and critic agents are **completely unchanged**. The web ingestion workflow creates `Document` and `ContentChunk` records with `source_type="web"` — everything downstream treats them identically to PDF-sourced chunks.

---

## Guardrails Summary

| Check | When | Action if Fails |
|-------|------|----------------|
| Subject safety | Before any research starts | Block entire web research for this subject, show clear message |
| Topic relevance | Per topic before searching | Skip topic, show in UI as skipped |
| Content safety per page | After scraping, before chunking | Skip page, add to skipped_urls |
| Content relevance per page | Same as above | Skip page if off-topic |

---

## Files to Create/Modify Summary

| File | Action | Description |
|------|--------|-------------|
| `agents/safety.py` | **CREATE** | SafetyAgent for subject + content screening |
| `agents/web_researcher.py` | **CREATE** | WebResearchAgent for searching & scraping |
| `agents/topic_parser.py` | **CREATE** | TopicParserAgent for extracting topic lists |
| `workflows/phase2_web_ingestion.py` | **CREATE** | LangGraph workflow for web ingestion |
| `core/database.py` | **MODIFY** | Add source_type, source_url to Document + ContentChunk |
| `core/config.py` | **MODIFY** | Add web scraping + safety config settings |
| `requirements.txt` | **MODIFY** | Add beautifulsoup4, duckduckgo-search, python-docx, lxml |
| `app.py` | **MODIFY** | Add Web Research tab, progress UI, source badges |

---

## Acceptance Criteria

1. User can enter topics via text area, or upload .txt/.pdf/.docx file
2. Parsed topics are shown for review before research starts
3. Subject safety check runs before any web requests; harmful subjects are blocked with clear message
4. Web research searches GeeksForGeeks, freeCodeCamp, Wikipedia, and relevant vendor docs
5. Each scraped page passes content safety filter before ingestion
6. Scraped content goes through identical pipeline: chunking → curator → socratic → critic
7. Every Document and ContentChunk records source_type and source_url
8. Flashcards in Mentor Review show source badge (🌐 domain or 📄 filename)
9. Flashcards in Learner view show source attribution
10. Progress is transparent to user during web research (which topic, which site, how many flashcards)
11. Skipped URLs (safety + errors) are reported to user
12. No changes to existing PDF upload flow
13. All existing tests pass
