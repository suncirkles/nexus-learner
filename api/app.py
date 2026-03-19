"""
api/app.py
-----------
FastAPI application factory for Nexus Learner.
Mounts all routers and configures CORS so the Streamlit UI (running on a
different port) can call it without browser-level blocks.

Start the server:
    uvicorn api.app:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import subjects, flashcards, topics, library, system
from api.middleware import ObservabilityMiddleware


def create_app() -> FastAPI:
    app = FastAPI(
        title="Nexus Learner API",
        version="2.0.0",
        description="REST API for the Nexus Learner multi-agent flashcard platform.",
    )

    # Middleware order: Observability first to capture everything
    app.add_middleware(ObservabilityMiddleware)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(subjects.router)
    app.include_router(flashcards.router)
    app.include_router(topics.router)
    app.include_router(library.router)
    app.include_router(system.router)

    return app


app = create_app()
