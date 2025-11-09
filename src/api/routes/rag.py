"""
RAG endpoints for FKS Analyze Service.
Provides endpoints for document ingestion and querying the RAG system.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

from src.services.rag_pipeline import RAGPipelineService

router = APIRouter()

# Global RAG service instance
rag_service: Optional[RAGPipelineService] = None


def get_rag_service() -> RAGPipelineService:
    """Dependency to get the RAG service instance."""
    global rag_service
    if rag_service is None:
        rag_service = RAGPipelineService()
    return rag_service


class RAGAnalyzeRequest(BaseModel):
    """Request model for RAG document analysis."""
    
    job_id: Optional[str] = Field(default=None, description="Optional job ID (auto-generated if not provided)")
    query: str = Field(..., description="Query or document description")
    analysis_type: str = Field(default="project_management", 
                              description="Type of analysis (project_management, standardization, documentation)")
    scope: str = Field(default="all", description="Scope of analysis (all, specific repo, or service)")
    document_content: Optional[str] = Field(default=None, description="Document content to analyze")
    document_path: Optional[str] = Field(default=None, description="Path to document")
    document_name: Optional[str] = Field(default=None, description="Document name")


class RAGQueryRequest(BaseModel):
    """Request model for RAG query."""
    
    job_id: Optional[str] = Field(default=None, description="Optional job ID (auto-generated if not provided)")
    query: str = Field(..., description="Query text")
    analysis_type: str = Field(default="project_management",
                              description="Type of analysis")
    scope: str = Field(default="all", description="Scope of data retrieval")


class RAGJobResponse(BaseModel):
    """Response model for RAG job submission."""
    
    job_id: str
    status: str
    message: str
    timestamp: datetime


@router.post("/analyze", response_model=RAGJobResponse)
async def analyze_document(
    request: RAGAnalyzeRequest,
    background_tasks: BackgroundTasks,
    rag: RAGPipelineService = Depends(get_rag_service)
) -> RAGJobResponse:
    """
    Analyze a document using the RAG system.
    
    Args:
        request: RAG analysis request parameters
        background_tasks: FastAPI background tasks
        rag: RAG pipeline service instance
    
    Returns:
        Job submission response with job_id
    """
    job_id = request.job_id or str(uuid.uuid4())
    
    # Prepare query
    query = request.query
    if request.document_content:
        query = f"{query}\n\nDocument Content:\n{request.document_content[:5000]}"  # Limit content
    
    # Add analysis task to background
    background_tasks.add_task(
        rag.run_rag_analysis,
        job_id=job_id,
        query=query,
        analysis_type=request.analysis_type,
        scope=request.scope,
        max_retrieved=10
    )
    
    return RAGJobResponse(
        job_id=job_id,
        status="queued",
        message=f"RAG analysis job {job_id} queued",
        timestamp=datetime.utcnow()
    )


@router.post("/query", response_model=RAGJobResponse)
async def query_rag(
    request: RAGQueryRequest,
    background_tasks: BackgroundTasks,
    rag: RAGPipelineService = Depends(get_rag_service)
) -> RAGJobResponse:
    """
    Query the RAG system with a natural language question.
    
    Args:
        request: RAG query request parameters
        background_tasks: FastAPI background tasks
        rag: RAG pipeline service instance
    
    Returns:
        Job submission response with job_id
    """
    job_id = request.job_id or str(uuid.uuid4())
    
    # Add query task to background
    background_tasks.add_task(
        rag.run_rag_analysis,
        job_id=job_id,
        query=request.query,
        analysis_type=request.analysis_type,
        scope=request.scope,
        max_retrieved=10
    )
    
    return RAGJobResponse(
        job_id=job_id,
        status="queued",
        message=f"RAG query job {job_id} queued",
        timestamp=datetime.utcnow()
    )


@router.get("/status/{job_id}")
async def get_rag_status(
    job_id: str,
    rag: RAGPipelineService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Get RAG job status.
    
    Args:
        job_id: Job identifier
        rag: RAG pipeline service instance
    
    Returns:
        Current status of the RAG job
    """
    status = rag.get_job_status(job_id)
    
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"RAG job {job_id} not found"
        )
    
    return status


@router.get("/results/{job_id}")
async def get_rag_results(
    job_id: str,
    rag: RAGPipelineService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Get RAG job results.
    
    Args:
        job_id: Job identifier
        rag: RAG pipeline service instance
    
    Returns:
        RAG analysis results
    """
    results = rag.get_job_results(job_id)
    
    if results is None:
        raise HTTPException(
            status_code=404,
            detail=f"Results for RAG job {job_id} not found"
        )
    
    return results


@router.get("/jobs")
async def list_rag_jobs(
    rag: RAGPipelineService = Depends(get_rag_service)
) -> List[Dict[str, Any]]:
    """
    List all RAG jobs.
    
    Args:
        rag: RAG pipeline service instance
    
    Returns:
        List of RAG jobs with their status
    """
    return rag.list_jobs()


@router.delete("/jobs/{job_id}")
async def delete_rag_job(
    job_id: str,
    rag: RAGPipelineService = Depends(get_rag_service)
) -> Dict[str, str]:
    """
    Delete a RAG job and its results.
    
    Args:
        job_id: Job identifier
        rag: RAG pipeline service instance
    
    Returns:
        Deletion confirmation
    """
    success = rag.delete_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"RAG job {job_id} not found"
        )
    
    return {"message": f"RAG job {job_id} deleted successfully"}

