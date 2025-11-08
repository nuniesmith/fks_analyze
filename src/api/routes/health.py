"""
Health check endpoints for FKS Analyze Service.
"""

from fastapi import APIRouter, status
from datetime import datetime
from typing import Dict, Any

router = APIRouter()


@router.get("", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Health status of the service
    """
    return {
        "status": "healthy",
        "service": "fks_analyze",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint.
    
    Returns:
        Readiness status of the service
    """
    return {
        "ready": True,
        "service": "fks_analyze",
        "timestamp": datetime.utcnow().isoformat()
    }
