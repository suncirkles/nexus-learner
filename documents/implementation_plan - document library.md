# Knowledge Library & Decoupled Generation

Refactor the architecture to support a **Common Knowledge Library**. Documents are indexed (chunked and topic-mapped) once globally. Subjects then "subscribe" to these documents and generate subject-specific flashcards.

## User Review Required

> [!IMPORTANT]
> **Data Reset**: To facilitate this major architectural shift, we will perform a clean database reset. All existing subjects and cards will be cleared.
>
> **Shared Knowledge, Private Cards**: Topics and Subtopics are discovered once per document and shared across all subjects. However, the **Flashcards** generated for those topics remain isolated within the specific Subject where they were created.
> 
> **Similarity Matching**: When you provide topics to filter, the system will use semantic similarity to map your request to the document's pre-indexed subtopics.

## Proposed Changes

### Database Layer

#### [MODIFY] [database.py](file:///d:/projects/Gen-AI/Nexus%20Learner/core/database.py)
*   **[NEW]** `SubjectDocumentAssociation`: Bridge table linking `Subject` to `Document`.
*   **[MODIFY]** `Topic`: Change `subject_id` (ForeignKey) to `document_id`. A topic now belongs to a document.
*   **[MODIFY]** `Flashcard`: Add `subject_id` (ForeignKey). Flashcards now belong to both a Subject and a Subtopic.
*   **[MODIFY]** `ContentChunk`: Ensure `subtopic_id` is robustly linked.

---

### Agent Layer

#### [MODIFY] [topic_assigner.py](file:///d:/projects/Gen-AI/Nexus%20Learner/agents/topic_assigner.py)
*   Update to handle global document indexing context.
*   Ensure every chunk is assigned a descriptive subtopic (No "General Content").

#### [NEW] [topic_matcher.py](file:///d:/projects/Gen-AI/Nexus%20Learner/agents/topic_matcher.py)
*   Agent to perform semantic similarity matching between user-provided topics and a document's indexed subtopics.

---

### Workflow Layer

#### [MODIFY] [phase1_ingestion.py](file:///d:/projects/Gen-AI/Nexus%20Learner/workflows/phase1_ingestion.py)
*   **Mode: INDEXING**: A pure Document-level workflow. No subject context required. Extracts all chunks and topics.
*   **Mode: GENERATION**: Subject-level workflow. Takes `subject_id`, `doc_id`, and `target_topics`. Matches topics to doc subtopics and generates cards for that subject.

---

### UI Layer

#### [MODIFY] [app.py](file:///d:/projects/Gen-AI/Nexus%20Learner/app.py)
*   **Knowledge Library**: Add a dedicated section or tab to manage global documents (Upload & Index).
*   **Document Attachment**: Within a Subject, provide a UI to "Attach" documents from the library.
*   **Topic-Based Processing**:
    *   Show "Available Topics" from all attached documents.
    *   Allow text input for filtering (with similarity matching).
    *   Trigger subject-specific generation.

## Verification Plan

### Automated Tests
*   `tests/test_global_indexing.py`: Verify a document can be fully indexed without a subject.
*   `tests/test_subject_association.py`: Verify that attaching a document to two subjects allows independent flashcard generation.

### Manual Verification
1. Upload a PDF to the Library.
2. Create "Subject A" and "Subject B".
3. Attach the PDF to both.
4. Generate cards for "Topic 1" in Subject A.
5. Verify Subject B still shows 0 cards for that topic until generated specifically for B.
