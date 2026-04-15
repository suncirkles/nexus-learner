# Deployment Guide — Nexus Learner on Modal

## Architecture

Three Modal functions share a single App (`nexus-learner`):

| Function | Type | Purpose |
|---|---|---|
| `fastapi_app` | `@modal.asgi_app` | REST API — all business logic |
| `streamlit_app` | `@modal.web_server(8501)` | Streamlit UI — proxied by Modal |
| `run_ingestion_background` | `@modal.function` | Long-running ingestion worker (spawned per job) |

All three mount the same **Volume** (`nexus-learner-data`) at `/data` for shared file storage (uploads, logs).  
All three read the same **Secret** (`nexus-learner-secrets`) for API keys and DB credentials.

### Why two containers for UI + API?

Streamlit needs `min_containers=1` and `max_containers=1` (session state must not be split across replicas).  
FastAPI scales independently. The Streamlit container talks to FastAPI over the public Modal URL — not localhost.

---

## Prerequisites

1. **Modal account** — `pip install modal && modal setup`
2. **Secrets** — create a `nexus-learner-secrets` secret in the Modal dashboard containing all keys from `.env.example`
3. **Volume** — created automatically on first deploy (`create_if_missing=True`)

---

## Deploy

```bash
# One-time: authenticate
modal setup

# Deploy all three functions
modal deploy modal_app.py

# Output includes the live URLs:
#   https://<username>--nexus-learner-fastapi-app.modal.run
#   https://<username>--nexus-learner-streamlit-app.modal.run
```

After deploy, update `API_BASE_URL` in Modal secrets to the FastAPI URL, and update `serverAddress` in `modal_app.py`'s Streamlit config block to the Streamlit URL.

---

## Secrets (`nexus-learner-secrets`)

All variables from `.env.example` must be present. Key ones for Modal specifically:

| Variable | Value |
|---|---|
| `DB_URL` | Supabase connection string (pooler URL, port 5432) |
| `API_BASE_URL` | `https://<username>--nexus-learner-fastapi-app.modal.run` |
| `VECTOR_STORE_TYPE` | `qdrant` (Qdrant Cloud) or `pgvector` (Supabase) |
| `QDRANT_URL` | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant Cloud API key |

### Sync secrets from local `.env`

```bash
python scripts/sync_secrets.py
```

---

## Checking logs

```bash
# Stream live logs from both functions
modal app logs nexus-learner

# Or filter by function
modal app logs nexus-learner --function fastapi_app
modal app logs nexus-learner --function streamlit_app
modal app logs nexus-learner --function run_ingestion_background
```

Alternatively: Modal dashboard → Apps → nexus-learner → Functions → select function → Logs tab.

---

## Local development vs Modal

| Concern | Local | Modal |
|---|---|---|
| API URL | `http://127.0.0.1:8000` | `https://<username>--nexus-learner-fastapi-app.modal.run` |
| Vector DB | Qdrant Docker (`localhost:6333`) | Qdrant Cloud |
| Postgres | Supabase (same) | Supabase (same) |
| File uploads | Written to `./temp_uploads/` | Written to `/data/temp_uploads/` (Modal Volume) |
| Ingestion worker | Background thread in FastAPI | Separate Modal container (`run_ingestion_background`) |

---

## Gotchas & Lessons Learned

### 1. Deploy and test after every significant change

The most painful incidents came from batching multiple commits before deploying. Regressions are hard to attribute when 3-5 changes land at once. Rule: **deploy to Modal before merging any PR that touches `api/`, `workflows/`, `ui/`, or `modal_app.py`**.

### 2. `add_local_dir` ignore list must stay current

Modal's image build fails with `"file was modified during build process"` if any local directory contains files that change during the build (test caches, temp files, log files). Always exclude:

```python
ignore=["agentic", ".git", "logs", "__pycache__", "page_cache",
        ".gemini", "venv", ".pytest_cache", "scratch", "tmp"]
```

Add new volatile directories here before deploying.

### 3. Volume writes must be committed before spawning a worker

The FastAPI container and the ingestion worker run in **separate containers**. A file written by FastAPI is not visible to the worker unless the volume is explicitly committed first:

```python
# In FastAPI (after writing the upload):
vol = modal.Volume.from_name("nexus-learner-data")
vol.commit()

# In the worker (before reading the file):
vol.reload()
```

The ingestion router (`api/routers/ingestion.py`) handles both sides.

### 4. `max_containers=1` is required for Streamlit

Streamlit stores uploaded files and job state in-process. Without `max_containers=1`, Modal can load-balance requests across multiple containers, splitting session state. Symptoms: uploaded files disappear, progress monitors show stale data.

### 5. CORS `allow_origins` requires exact URLs — no wildcards

Starlette's `CORSMiddleware` does not support glob patterns like `*.modal.run`. Use exact origins:

```python
ALLOWED_ORIGINS: list[str] = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "https://<username>--nexus-learner-streamlit-app.modal.run",
]
```

Note: Streamlit→FastAPI calls are server-side (Python httpx), so CORS doesn't block them. This matters for browser-initiated requests only.

### 6. Streamlit `serverAddress` must match the deployed URL

The `[browser]` section of `config.toml` inside `modal_app.py` must reflect the actual Streamlit Modal URL. Mismatch causes WebSocket reconnection loops:

```toml
[browser]
serverAddress = "<username>--nexus-learner-streamlit-app.modal.run"
serverPort = 443
```

### 7. Cold starts — FastAPI

FastAPI containers can take 60–120 s to cold-start (image pull + dependency load + fastembed model warm-up). The Dashboard page has automatic retry logic (15 × 6 s) to handle this transparently. `min_containers=0` for FastAPI is acceptable since retries handle the delay.

### 8. Cold starts — Streamlit

`min_containers=1` keeps the Streamlit container alive permanently. If it gets replaced (new deploy), expect a 30–60 s gap. The Modal `@web_server` decorator will return 502 until the subprocess reports healthy on port 8501.

---

## Rollback

```bash
# List recent deployments
modal app history nexus-learner

# Re-deploy a specific git commit (no Modal-native rollback)
git checkout <commit>
modal deploy modal_app.py
git checkout -
```

---

## DB migrations

Migrations run automatically at import time via `core/database.py::_run_migrations()`. No manual steps needed on deploy. The function uses `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`-style logic — safe to run against an existing Supabase schema.

To reset the entire DB (destructive — dev only):

```
POST /system/reset
```
