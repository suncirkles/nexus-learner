"""
modal_app.py
-------------
Unified Modal entry point for the Nexus Learner platform.
Hosts:
1. FastAPI Backend (@modal.asgi_app)
2. Streamlit Frontend (@modal.web_server)
3. Background Workers (Ingestion tasks)
"""

import modal
import os
import subprocess
from pathlib import Path
from typing import Optional

# 1. Define the system image and Python environment
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libtesseract-dev", "tesseract-ocr", "build-essential", "g++", "libpq-dev")
    .pip_install(
        "langchain-core",
        "langchain-text-splitters",
        "langchain-openai",
        "langchain-anthropic",
        "langchain-groq",
        "langchain-google-genai",
        "langchain-litellm",
        "langchain-qdrant",
        "langgraph",
        "langsmith",
        "litellm",
        "qdrant-client[fastembed]",
        "sqlalchemy",
        "psycopg2-binary",
        "pgvector",
        "langchain-postgres",
        "redis",
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "httpx",
        "pydantic",
        "pydantic-settings",
        "python-dotenv",
        "streamlit",
        "pymupdf",
        "pytesseract",
        "Pillow",
        "python-docx",
        "lxml",
        "beautifulsoup4",
        "ddgs",
        "requests"
    )
    # Pre-download the fastembed ONNX model so cold starts don't hit HuggingFace.
    # This bakes the model weights into the image layer — no network call at runtime.
    .run_commands(
        "python -c \""
        "from fastembed import TextEmbedding; "
        "list(TextEmbedding('sentence-transformers/all-MiniLM-L6-v2').embed(['warmup']))"
        "\""
    )
    # Mirror the local source code into the container, excluding large/unnecessary folders
    .add_local_dir(
        ".",
        remote_path="/root/nexus-learner",
        ignore=["agentic", ".git", "logs", "__pycache__", "page_cache", ".gemini", "venv", ".pytest_cache", "scratch", "tmp"]
    )
)

app = modal.App("nexus-learner", image=image)

# Persistent volume for logs and temporary uploads
volume = modal.Volume.from_name("nexus-learner-data", create_if_missing=True)

# Shared secrets (contains Supabase, LLM, and other API keys)
secrets = [modal.Secret.from_name("nexus-learner-secrets")]


# ---------------------------------------------------------------------------
# 1. FastAPI Backend
# ---------------------------------------------------------------------------

@app.function(
    volumes={"/data": volume},
    secrets=secrets,
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    import sys
    import os
    sys.path.append("/root/nexus-learner")
    os.environ["MODAL_RUN"] = "true"
    os.environ["DATA_DIR"] = "/data"
    from api.app import app as _app
    return _app


# ---------------------------------------------------------------------------
# 2. Streamlit Frontend
# ---------------------------------------------------------------------------

@app.function(
    volumes={"/data": volume},
    secrets=secrets,
    timeout=86400, # 24 hours: Prevent Modal from killing the Streamlit GUI
    max_containers=1, # REQUIRED FOR STREAMLIT: Forces all traffic to a single container so session state (uploaded files) isn't split across load-balanced replicas.
    min_containers=1, # Keep UI container alive
)
@modal.concurrent(max_inputs=100)
@modal.web_server(8501)
def streamlit_app():
    import sys
    import os
    import subprocess

    sys.path.append("/root/nexus-learner")

    # 1. Create robust Streamlit configuration in BOTH home and project root
    config_content = """
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false
enableWebsocketCompression = false
maxUploadSize = 200
headless = true

[browser]
gatherUsageStats = false
serverAddress = "rajeshrkt55--nexus-learner-streamlit-app.modal.run"
serverPort = 443
"""
    for path in [Path("/root/.streamlit"), Path("/root/nexus-learner/.streamlit")]:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.toml", "w") as f:
            f.write(config_content)

    # 2. Environmental context
    os.environ["MODAL_RUN"] = "true"
    os.environ["DATA_DIR"] = "/data"
    os.environ["API_BASE_URL"] = "https://rajeshrkt55--nexus-learner-fastapi-app.modal.run"
    
    # 3. Explicitly create the upload directory on the Volume
    upload_dir = Path("/data/temp_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Run Streamlit, binding to the port Modal expects
    target_path = "/root/nexus-learner/app.py"
    cmd = f"streamlit run {target_path} --server.port 8501 --server.address 0.0.0.0"
    
    # Must NOT block. Modal's @web_server decorator intercepts the return to mark the container as ready.
    subprocess.Popen(cmd, shell=True)


# ---------------------------------------------------------------------------
# 3. Background Ingestion Worker
# ---------------------------------------------------------------------------

@app.function(
    volumes={"/data": volume},
    secrets=secrets,
    timeout=1800,  # 30 minutes
)
def run_ingestion_background(job_id: str, doc_id: str, file_path: Optional[str], subject_id: Optional[int], mode: str, target_topics: list = [], question_type: str = "active_recall"):
    """
    Serverless background worker.
    Executes the LangGraph ingestion flow and updates Supabase.
    """
    import sys
    sys.path.append("/root/nexus-learner")

    # Reload the volume so files written by the FastAPI container are visible here.
    # vol.commit() in the API container marks writes as durable; vol.reload() here
    # pulls those writes into this container's filesystem before we open the file.
    try:
        import modal as _modal
        _vol = _modal.Volume.from_name("nexus-learner-data")
        _vol.reload()
    except Exception as _ve:
        print(f"Warning: volume reload failed (non-fatal): {_ve}")

    from datetime import datetime, timezone
    from core.database import SessionLocal, BatchJob
    from workflows.phase1_ingestion import phase1_graph

    db = SessionLocal()
    job = db.query(BatchJob).filter(BatchJob.id == job_id).first()
    if not job:
        print(f"Error: Job {job_id} not found in database.")
        return

    try:
        current_state = {
            "mode": mode,
            "file_path": file_path,
            "doc_id": doc_id,
            "subject_id": subject_id,
            "target_topics": target_topics,
            "question_type": question_type,
            "total_pages": 0,
            "current_page": 0,
            "chunks": [],
            "current_chunk_index": 0,
            "hierarchy": [],
            "pending_qdrant_docs": [],
            "matched_subtopic_ids": None,
            "current_new_cards": [],
            "subtopic_embeddings": [],
            "generated_flashcards": [],
            "status_message": "Starting...",
        }

        job.status = "generating" if mode == "GENERATION" else "indexing"
        job.status_message = "Starting..."
        db.commit()

        # Stream the graph, writing granular progress to the DB after every node.
        # The UI polls /ingestion/status/{job_id} every 5 s to read these fields.
        _COMMIT_EVERY = 3   # commit at most once every N node events (reduces DB round-trips)
        _event_count = 0

        for event in phase1_graph.stream(current_state):
            for node_name, node_update in event.items():
                if not isinstance(node_update, dict):
                    continue
                current_state.update(node_update)

                # Map graph state keys → BatchJob columns
                if "status_message" in node_update:
                    job.status_message = node_update["status_message"]
                if "total_pages" in node_update:
                    job.total_pages = node_update["total_pages"]
                if "current_page" in node_update:
                    job.current_page = node_update["current_page"]
                if "current_chunk_index" in node_update:
                    job.current_chunk_index = node_update["current_chunk_index"]
                if "chunks" in node_update:
                    job.total_chunks = len(node_update["chunks"])
                if "generated_flashcards" in node_update:
                    job.flashcards_count = len(node_update["generated_flashcards"])

                _event_count += 1
                if _event_count % _COMMIT_EVERY == 0:
                    db.commit()

        job.status = "completed"
        job.status_message = "Done!"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

        # Commit the volume so page-cache PNGs written during indexing are visible
        # to other containers (the API container that serves chunk-page-image).
        try:
            import modal as _modal
            _modal.Volume.from_name("nexus-learner-data").commit()
            print(f"Volume committed after job {job_id}")
        except Exception as _ve:
            print(f"Warning: volume commit failed (non-fatal): {_ve}")

        print(f"Job {job_id} completed successfully.")
    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        import traceback; traceback.print_exc()
        job.status = "failed"
        job.error = str(e)
        job.status_message = f"Error: {e}"
        db.commit()
    finally:
        db.close()
