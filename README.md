# 🎓 Nexus Learner

An agentic AI platform that transforms static educational materials (PDFs, scanned images) into dynamic Active Recall flashcards using multi-agent workflows.

## ✨ Features

- **Smart Ingestion** – Upload PDFs or images; text is extracted with OCR fallback and duplicate detection.
- **Auto-Curation** – AI builds a Topic → Subtopic hierarchy from your documents.
- **Flashcard Generation** – High-quality Active Recall Q&A pairs, grounded and fact-checked by a Critic Agent.
- **Mentor Review (HITL)** – Approve, reject, or regenerate flashcards with bulk actions at topic level.
- **Active Recall Study Room** – Study approved flashcards with show/hide answer format.
- **Background Processing** – Initial burst of cards generated instantly; the rest processed in the background.

## 📋 Prerequisites

| Dependency | Version | Notes |
| :--- | :--- | :--- |
| Python | 3.11+ | |
| Docker | Latest | For Qdrant vector database |
| Tesseract OCR | Latest | Optional – for scanned PDF/image support |
| OpenAI API Key | – | Required for LLM and embeddings |

### Installing Tesseract (Optional)

- **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd nexus-learner
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example `.env` file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key        # Optional
LANGCHAIN_API_KEY=your-langsmith-api-key         # Optional, for tracing
LANGCHAIN_TRACING_V2=false                       # Set to "true" to enable LangSmith
LANGCHAIN_PROJECT=nexus_learner_mvp
DB_URL=sqlite:///./nexus_v3.db
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

### 5. Start Qdrant (Docker)

```bash
docker-compose up -d
```

This starts Qdrant on `http://localhost:6333`.

### 6. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 🧪 Running Tests

```bash
# Set PYTHONPATH and run all tests
# Windows PowerShell
$env:PYTHONPATH="."; pytest tests/

# macOS/Linux
PYTHONPATH=. pytest tests/
```

## 📁 Project Structure

```
nexus-learner/
├── app.py                          # Streamlit frontend (all views)
├── agents/
│   ├── ingestion.py                # Document parsing, chunking, embedding
│   ├── curator.py                  # Topic/Subtopic hierarchy extraction
│   ├── socratic.py                 # Flashcard generation & recreation
│   └── critic.py                   # Grounding evaluation (1-5 scoring)
├── core/
│   ├── config.py                   # Pydantic Settings (.env loading)
│   ├── models.py                   # LLM factory (get_llm)
│   ├── database.py                 # SQLAlchemy ORM models
│   └── background.py              # Background thread manager
├── workflows/
│   └── phase1_ingestion.py         # LangGraph pipeline definition
├── tests/                          # Pytest test suite
├── documents/
│   ├── prd.md                      # Product Requirements Document
│   └── hld.md                      # High Level Design
├── docker-compose.yml              # Qdrant service
├── requirements.txt                # Python dependencies
├── .env.example                    # Template for environment variables
└── README.md                       # This file
```

## 📖 Documentation

- [Product Requirements Document (PRD)](documents/prd.md)
- [High Level Design (HLD)](documents/hld.md)

## 🛠️ Usage Guide

1. **Create a Subject** – Go to "📚 Study Materials" and create a subject (e.g., "Machine Learning").
2. **Upload a Document** – Select the subject, upload a PDF, and click "🚀 Process & Generate Hierarchy".
3. **Review Flashcards** – Go to "👨‍🏫 Mentor Review" to approve/reject AI-generated cards.
4. **Study** – Head to "🧠 Learner" to study approved flashcards with Active Recall.

## ⚙️ Configuration

| Setting | Environment Variable | Default |
| :--- | :--- | :--- |
| LLM Provider | `DEFAULT_LLM_PROVIDER` | `openai` |
| Primary Model | `PRIMARY_MODEL` | `gpt-4o` |
| Routing Model | `ROUTING_MODEL` | `gpt-4o-mini` |
| Auto-Accept Content | `AUTO_ACCEPT_CONTENT` | `false` |

## 📄 License

This project is for educational purposes.
