# Streamlit + Modal in Production: Hard-Won Lessons

**Project:** Nexus Learner — a document-to-flashcard pipeline built on FastAPI, LangGraph, Streamlit, and Modal  
**Audience:** Python developers deploying their first production Streamlit + Modal app  
**Nature:** Incident-driven. Every section is a real failure from this codebase, not a hypothetical.

---

## Executive Summary

Modal's serverless container model and Streamlit's widget lifecycle each have sharp edges that are not well-documented and not obvious until you hit them in production. This paper documents eleven such edges, organized by subsystem. The pattern is consistent: behaviors that work locally or pass mock tests silently fail on Modal, and Streamlit exceptions that are cryptic in isolation become obvious once you understand the framework's ownership model over session state.

Read this before your first `modal deploy`.

---

## Section 1: Modal Volumes — The Commit/Reload Contract

### Problem

The ingestion pipeline had three actors:

1. A FastAPI container accepted a PDF upload, wrote it to `/data/temp_uploads/{doc_id}_filename.pdf`, and called `vol.commit()`.
2. A background worker was spawned, called `vol.reload()`, ran indexing, and wrote page-cache PNGs to `/data/page_cache/`. It did **not** call `vol.commit()`.
3. A different FastAPI container instance handled `GET /flashcards/chunk-page-image/{chunk_id}`. It never called `vol.reload()`. It scanned the upload directory for `{doc_id}_*`, found nothing, and returned 422.

### Root Cause

Modal volumes are not a distributed filesystem. They behave like a versioned snapshot store. `vol.commit()` publishes the current state of the filesystem under the mount point. `vol.reload()` pulls the latest published snapshot into the current container's view. Without an explicit commit, writes exist only in the writing container's local view and are invisible to all other containers.

### Fix

In `api/routers/flashcards.py`, on a cache miss, reload the volume before scanning:

```python
if os.environ.get("MODAL_RUN") == "true":
    try:
        import modal as _modal
        _modal.Volume.from_name("nexus-learner-data").reload()
    except Exception as _ve:
        logging.getLogger(__name__).warning("vol.reload() failed: %s", _ve)
    if os.path.exists(cached_path):
        with open(cached_path, "rb") as _f:
            png_bytes = _f.read()
        return {"image_b64": base64.b64encode(png_bytes).decode(), "page_number": page_number}
```

In `modal_app.py`, after the indexing graph stream completes and all PNGs are written:

```python
try:
    import modal as _modal
    _modal.Volume.from_name("nexus-learner-data").commit()
except Exception as _ve:
    print(f"Warning: volume commit failed: {_ve}")
```

### The Rule

Every container that writes persistent data must call `vol.commit()`. Every container that reads data written by another container must call `vol.reload()` before reading.

---

## Section 2: Modal Volumes — Stale Containers with min_containers=1

### Problem

The Streamlit app was deployed with:

```python
@app.function(
    volumes={"/data": volume},
    max_containers=1,
    min_containers=1,
    timeout=86400,
)
@modal.web_server(8501)
def streamlit_app(): ...
```

After `modal deploy`, the running container continued serving the old code. New code changes had no effect until the container was force-cycled.

### Root Cause

`min_containers=1` keeps exactly one container perpetually alive. When `max_containers=1` is also set, Modal cannot spin up a new container alongside the old one to do a zero-downtime swap. It must drain the old container first. If the old container is handling a long-lived Streamlit WebSocket connection (which it always is), it will not drain until that session ends. Modal does not forcibly terminate the old container on deploy.

### Fix

```bash
modal app stop nexus-learner
modal deploy modal_app.py
```

This terminates the live container, allowing the deploy to take effect immediately. Accept that there is a brief downtime window — the alternative is running stale code indefinitely.

### The Rule

`min_containers=1` plus `max_containers=1` means deployments require a manual stop. If you need zero-downtime rolling deploys, remove `min_containers` or raise `max_containers` to at least 2.

---

## Section 3: Modal Volumes — UUID Mismatch Between Router and Agent

### Problem

The upload router generated its own UUID:

```python
router_uuid = str(uuid.uuid4())
file_path = f"/data/temp_uploads/{router_uuid}_{filename}"
```

It then called `IngestionAgent.create_document_record(file_path)`, which internally computed a content-hash deduplication UUID — call it `canonical_uuid` — and stored that in the database. The database recorded all chunks with `document_id = canonical_uuid`.

Later, the page-image endpoint scanned `/data/temp_uploads/` for `{canonical_uuid}_*`. No such file existed. The file was named `{router_uuid}_*`. 422.

### Root Cause

Two services each generated an ID for the same logical entity at different points in the pipeline. The file was named with the first ID. The database was populated with the second. No reconciliation step existed.

### Fix

In `api/routers/ingestion.py`, resolve the canonical ID immediately after document record creation and rename the file before any further work proceeds:

```python
canonical_doc_id = _agent.create_document_record(file_path, subject_id=subject_id)
if canonical_doc_id != doc_id:
    canonical_path = os.path.join(upload_dir, f"{canonical_doc_id}_{safe_name}")
    if not os.path.exists(canonical_path):
        os.rename(file_path, canonical_path)
    else:
        os.remove(file_path)  # dedup: canonical copy already exists
    file_path = canonical_path
    _commit_volume()
doc_id = canonical_doc_id
```

### The Rule

Any service that generates its own ID for a resource must be called before that resource is named. Or: rename resources immediately after the authoritative ID is resolved. Never let two IDs for the same entity coexist across filesystem and database.

---

## Section 4: Modal — Relative Paths Write to the Ephemeral Container Filesystem

### Problem

The indexing worker wrote page-cache PNGs using a relative path:

```python
cache_dir = "page_cache"
img_path = os.path.join(cache_dir, f"{doc_id}_p{page:04d}.png")
```

The working directory inside a Modal container is `/root/nexus-learner/` (the project root). So this resolved to `/root/nexus-learner/page_cache/` — the container's local ephemeral filesystem, not the mounted volume at `/data/`. When the container terminated, all PNGs were gone.

### Root Cause

Modal mounts volumes at explicit mount points. Relative paths resolve against the container working directory, which is never under a volume mount point. There is no warning — writes succeed, the files exist for the lifetime of that container instance, and then they vanish.

### Fix

Introduce a `DATA_DIR` environment variable that defaults to `.` locally and is set to `/data` in the Modal deployment:

```python
# core/config.py
DATA_DIR: str = os.getenv("DATA_DIR", ".")
PAGE_CACHE_DIR: str = "page_cache"

@property
def abs_page_cache_dir(self) -> str:
    path = os.path.join(self.DATA_DIR, self.PAGE_CACHE_DIR)
    os.makedirs(path, exist_ok=True)
    return path
```

```python
# modal_app.py — set before any project imports
os.environ["DATA_DIR"] = "/data"
```

All code that writes or reads persistent files uses `settings.abs_page_cache_dir` or the equivalent `settings.abs_upload_dir`. No relative paths for persistent data anywhere in the codebase.

### The Rule

Every path that must survive a container restart must be rooted under the volume mount point. Use a `DATA_DIR` env var to switch between local development (`.`) and Modal (`/data`). Audit every `open()`, `os.path.join()`, and file write in your codebase before deploying.

---

## Section 5: Streamlit — Widget-Bound Session State Cannot Be Mutated After Instantiation

### Problem

A dashboard button tried to navigate to the Learner view by writing to the sidebar radio's session state key directly:

```python
if st.button("Start Learning"):
    st.session_state.sidebar_nav = "🧠 Learner"
    st.rerun()
```

This raised:

```
StreamlitAPIException: st.session_state.sidebar_nav cannot be modified after widget instantiation.
```

### Root Cause

When you pass `key="sidebar_nav"` to `st.sidebar.radio()`, Streamlit takes ownership of that session state key for the duration of the script run. After the widget has been instantiated in the current run, no code may write to that key — including callbacks and button handlers. Streamlit enforces this to prevent races between widget state and user input.

### Fix

Use an intermediary "pending navigation" key. Consume it before the radio widget renders:

```python
# app.py — runs before the radio widget is instantiated
if "pending_nav" in st.session_state:
    st.session_state.sidebar_nav = st.session_state.pop("pending_nav")

active_nav = st.sidebar.radio("Navigation", nav_options, key="sidebar_nav")
```

```python
# dashboard.py — writes to staging key, not to the widget key
if st.button("Start Learning"):
    st.session_state.study_subject_id = subj_id
    st.session_state.pending_nav = "🧠 Learner"
    st.rerun()
```

On the next script run triggered by `st.rerun()`, `pending_nav` is consumed before the radio renders. The radio initializes with the correct value.

### The Rule

Never write to a session state key that is also a widget `key=`. Use a staging key and consume it before the widget renders.

---

## Section 6: Streamlit — @st.fragment with run_every Manages Its Own Rerun Cycle

### Problem

A background-monitoring fragment used `run_every=5` and also called `st.rerun()` inside its body. This caused `WebSocketClosedError` in server logs and `ForwardMsg MISS` warnings. Navigation clicks from the sidebar appeared to have no effect — the displayed view was not updating even when session state changed.

### Root Cause

`@st.fragment(run_every=N)` re-executes only the fragment's code on its timer. It does not re-execute the full script. When `st.rerun()` is called inside a fragment, it triggers a full script rerun, which conflicts with the fragment's own scheduled rerun already in the framework's queue. This double-rerun corrupts the `ForwardMsg` delivery sequence, which is why the client drops messages.

The navigation breakage was a secondary effect: the sidebar radio (`key="sidebar_nav"`) was rendered by the full script, but the fragment's timer-driven partial reruns were not re-rendering the sidebar. The displayed state diverged from session state.

### Fix

Remove all `st.rerun()` calls from `run_every` fragments. Let the fragment timer drive the rerun. Keep `run_every` at 5 seconds or higher — faster intervals generate enough WebSocket traffic to degrade responsiveness for other interactions.

### The Rule

`run_every` fragments manage their own rerun cycle. Calling `st.rerun()` inside one conflicts with the framework's rerun queue. Fragment reruns update only the fragment's scope; they do not re-render the sidebar or other script-level widgets.

---

## Section 7: Streamlit — Image Display and the MediaFileStore Persistence Trap

### Problem

Three approaches to displaying PDF page images were tried. Two had production-breaking failure modes.

**Approach 2 (`st.image(bytes)`) — the trap:**

```python
st.image(png_bytes, use_container_width=True)
```

This worked in development. On Modal, it produced broken images after container restarts. Streamlit's `MediaFileStore` assigns each image a hash-based URL (`/_stcore/media/{hash}`) and serves it from in-memory storage. When the container restarted — which Modal does regularly — the in-memory store was empty. The browser had cached the URL from the previous session and requested it; the server returned 404. The image element rendered as a broken icon.

Additionally, `use_container_width=True` stretched low-resolution scanned pages until they were visibly pixelated.

**Approach 3 (`st.expander` + `st.image`) — also broken:**

Same `MediaFileStore` problem, plus `st.expander`'s default indentation broke the card layout alignment.

### Fix

Approach 1: embed the base64 string directly in an HTML `<details>` element:

```python
st.markdown(
    f"<details><summary style='cursor:pointer; color:#58a6ff;'>📄 Source Page {n}</summary>"
    f"<img src='data:image/png;base64,{b64}' style='max-width:100%; height:auto; display:block; margin-top:8px;'/>"
    f"</details>",
    unsafe_allow_html=True,
)
```

The base64 string is inlined in the HTML. There is no server-side media store, no hash URL, no in-memory state. The image survives container restarts because it is transmitted fresh with every response.

Use `max-width:100%; height:auto`, not `width:100%`. The former constrains oversized images while preserving the natural size of small ones. `width:100%` stretches everything to container width and pixelates low-DPI scans.

### The Rule

`st.image(bytes)` uses an in-memory `MediaFileStore` that does not survive container restarts. On Modal, where containers restart between sessions, this produces broken images. Use inline base64 in `st.markdown` for images that must be reliably visible after any container restart.

---

## Section 8: Streamlit — CSS Properties That Collapse Containers

### Problem

To remove whitespace around image elements, this CSS was applied to a markdown container:

```python
f"<div style='margin:0;padding:0;line-height:0;font-size:0;'>..."
```

The entire div and all its contents became invisible. Not broken — invisible. No error. Nothing in the browser console.

### Root Cause

Streamlit uses the computed `font-size` and `line-height` of its wrapper elements to calculate the rendered height of a markdown component. Setting either to `0` collapses the container to zero height. The content is technically present in the DOM but occupies no space and renders as invisible.

### Fix

Remove `font-size:0` and `line-height:0`. To eliminate whitespace between image and border, use negative margin or adjust padding on the `<img>` element itself. Setting `display:block` on the `<img>` is usually sufficient to eliminate the inline image baseline gap without touching font metrics.

### The Rule

Never set `font-size:0` or `line-height:0` on a Streamlit markdown container. Streamlit uses these properties to calculate rendered component height.

---

## Section 9: Testing — Patching uuid.uuid4 Globally Breaks Transitive Imports

### Problem

A test patched `uuid.uuid4` in the ingestion router to control document ID generation:

```python
uuid_seq = iter([ROUTER_UUID, str(uuid.uuid4())])
with patch("api.routers.ingestion.uuid.uuid4", side_effect=uuid_seq):
    ...
```

The test raised `AttributeError: 'str' object has no attribute 'hex'` — inside `transformers/utils/hub.py`, which is not test code.

### Root Cause

`patch("api.routers.ingestion.uuid.uuid4")` does not patch `uuid4` only within `api.routers.ingestion`. It patches the `uuid4` attribute on the `uuid` module object in `sys.modules`, which is a singleton shared by all code in the process. When the patch context was entered, it triggered the first import of `agents.ingestion` → `langchain_text_splitters` → `transformers`. At module level, `transformers/utils/hub.py` executes:

```python
SESSION_ID = uuid4().hex
```

With the mock active, `uuid4()` returned a plain string (the next value from the iterator). Calling `.hex` on a string raised `AttributeError`.

### Fix

Pre-import all transitive dependencies before entering the uuid patch, so all module-level code that calls `uuid4()` runs against the real function:

```python
import agents.ingestion  # force transitive imports before uuid patch

_real_uuids = [str(uuid.uuid4()) for _ in range(8)]  # generated before patch
uuid_seq = iter([ROUTER_UUID] + _real_uuids)
```

Generate the surplus UUIDs outside the patch context so the iterator always has real `UUID` objects with a valid `.hex` attribute. Inside the patch, the iterator should only return values for call sites you control.

### The Rule

`patch("module.uuid.uuid4")` patches globally. Pre-import all transitive dependencies before entering uuid patches. Always give uuid iterators a surplus of values beyond the call sites you know about — module-level code in third-party libraries will consume values you did not account for.

---

## Section 10: Testing — Patching @property Settings Requires patch.object + PropertyMock

### Problem

A test attempted to override the upload directory by patching the settings object directly:

```python
with patch("api.routers.ingestion.settings.abs_upload_dir", "/tmp/test"):
    ...
```

The patch silently failed — `settings.abs_upload_dir` still returned the real path.

### Root Cause

`abs_upload_dir` is a `@property` defined on the `Settings` class. `patch("module.settings.prop")` treats `prop` as an instance attribute and tries to set it on the instance. This does not work for descriptors — Python resolves `@property` through the class `__dict__`, not the instance. Setting an instance attribute with the same name shadows the descriptor only if the property has no `__set__` implementation, and Pydantic settings properties do not.

### Fix

```python
from unittest.mock import PropertyMock, patch
from core.config import Settings

with patch.object(Settings, "abs_upload_dir",
                  new_callable=PropertyMock,
                  return_value=str(upload_dir)):
    ...
```

`patch.object` with `new_callable=PropertyMock` replaces the descriptor on the class itself for the duration of the context manager. All accesses to `settings.abs_upload_dir` within that scope return the mock value.

### The Rule

`patch("module.settings.prop")` cannot replace a `@property`. Use `patch.object(SettingsClass, "prop", new_callable=PropertyMock)` to patch it on the class descriptor.

---

## Section 11: Testing — Mock Tests Cannot Verify Modal Volume Consistency

### Problem

The page-image pipeline had complete unit test coverage. All tests passed. The production 422 persisted across two deploys. The root cause (Section 1) was not caught by any test.

The mock tests patched `_commit_volume` to a no-op and ran within a single process. From the test's perspective, the filesystem was consistent because there was only one process, one filesystem view, and no container boundaries. The test was structurally incapable of observing the failure mode.

### Fix

The only test that proved the behavior was an end-to-end test against the live Modal deployment:

```python
# tests/test_modal_e2e_page_image.py
@pytest.mark.modal_e2e
def test_chunk_page_image_returns_png_not_422(indexed_document):
    client = _http()  # hits live rajeshrkt55--nexus-learner-fastapi-app.modal.run
    for chunk_id in indexed_document["chunk_ids"]:
        r = client.get(f"/flashcards/chunk-page-image/{chunk_id}")
        assert r.status_code == 200
        assert base64.b64decode(r.json()["image_b64"])[:4] == b"\x89PNG"
```

Run with:

```bash
pytest tests/test_modal_e2e_page_image.py -v -s -m modal_e2e
```

This test is gated behind a `modal_e2e` marker so it does not run in CI against local infrastructure. It runs manually after every deploy that touches the ingestion or page-image path.

### The Rule

Modal volume consistency is infrastructure behavior that cannot be replicated in a single-process test. Any test that mocks `vol.commit()` or `vol.reload()` as no-ops tells you nothing about cross-container file visibility. After every deploy that touches the ingestion pipeline, run an end-to-end test against the live deployment.

---

## Appendix: Pre-Deployment Checklist

Before every `modal deploy`:

- [ ] All paths that write persistent data use `settings.abs_*` properties rooted under `DATA_DIR`, not relative paths
- [ ] Every function that writes to the volume calls `vol.commit()` before returning
- [ ] Every function that reads data written by another container calls `vol.reload()` on cache miss
- [ ] No session state key used as a widget `key=` is written to directly outside of `pending_*` staging pattern
- [ ] No `st.rerun()` calls inside `@st.fragment(run_every=...)` bodies
- [ ] No `st.image(bytes)` used for images that must survive container restarts
- [ ] After deploy: run `pytest tests/ -m modal_e2e` against the live URL
- [ ] If `min_containers=1`: run `modal app stop <name>` before `modal deploy` to ensure new code takes effect
