"""
api/routers/system.py
-----------------------
System administration endpoints (global reset).
"""

from fastapi import APIRouter, Depends

from api.schemas import ResetResponse
from api.dependencies import get_system_service
from services.system_service import SystemService

router = APIRouter(prefix="/system", tags=["system"])


@router.post("/reset", response_model=ResetResponse)
def reset_system(svc: SystemService = Depends(get_system_service)):
    """Wipe all relational tables and the Qdrant collection."""
    svc.reset()
    return ResetResponse(status="ok", message="System reset complete.")
