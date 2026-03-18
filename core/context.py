"""
core/context.py
---------------
Context management for tracing and observability.
Uses ``contextvars`` to maintain request-scoped data (request_id, session_id, user_id)
across asynchronous and synchronous call stacks.
"""

import contextvars
from typing import Optional

# contextvars for request tracing
request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
session_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("session_id", default=None)
user_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("user_id", default=None)

def get_request_id() -> str:
    return request_id_ctx.get() or "no-req"

def set_request_id(value: str) -> None:
    request_id_ctx.set(value)

def get_session_id() -> str:
    return session_id_ctx.get() or "no-sess"

def set_session_id(value: str) -> None:
    session_id_ctx.set(value)

def get_user_id() -> str:
    return user_id_ctx.get() or "anonymous"

def set_user_id(value: str) -> None:
    user_id_ctx.set(value)

def get_langchain_config() -> dict:
    """Return a config dict for LangChain invoke/stream calls with trace context."""
    return {
        "metadata": {
            "request_id": get_request_id(),
            "session_id": get_session_id(),
            "user_id": get_user_id(),
        },
        "tags": [get_session_id()]
    }
