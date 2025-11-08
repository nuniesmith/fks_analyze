"""
FKS Analyze Service - FastAPI Application
Provides API endpoints for repository analysis and code quality checks.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import logging

from src.api.routes import analysis, health
from src.core.config import get_settings
from src.services.analyzer import AnalyzerService

logger = logging.getLogger(__name__)

# Global analyzer service instance
analyzer_service: Optional[AnalyzerService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global analyzer_service
    
    # Startup
    logger.info("Starting FKS Analyze Service...")
    analyzer_service = AnalyzerService()
    
    yield
    
    # Shutdown
    logger.info("Shutting down FKS Analyze Service...")
    analyzer_service = None


# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title="FKS Analyze Service",
    description="Repository analysis and code quality service for FKS Trading Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "fks_analyze",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }


def get_analyzer() -> AnalyzerService:
    """Dependency to get the analyzer service instance."""
    if analyzer_service is None:
        raise HTTPException(status_code=503, detail="Analyzer service not initialized")
    return analyzer_service
