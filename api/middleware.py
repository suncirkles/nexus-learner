"""
api/middleware.py
-----------------
Observability middleware for FastAPI.
Extracts trace IDs from headers and populates the core context.
"""

import logging
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from core.context import set_request_id, set_session_id, set_user_id

logger = logging.getLogger(__name__)

class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1. Extract or generate Request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        # 2. Extract Session ID (passed from Streamlit)
        session_id = request.headers.get("X-Session-ID") or "no-session"
        
        # 3. Extract User ID (if available)
        user_id = request.headers.get("X-User-ID") or "anonymous"

        # 4. Set Context (per-request storage)
        set_request_id(request_id)
        set_session_id(session_id)
        set_user_id(user_id)

        logger.info("Processing request: %s %s", request.method, request.url.path)

        # 5. Process Request
        response = await call_next(request)

        # 6. Attach Request ID to response for troubleshooting
        response.headers["X-Request-ID"] = request_id
        
        return response
