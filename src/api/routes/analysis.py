"""
Analysis endpoints for FKS Analyze Service.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

from src.services.analyzer import AnalyzerService
# Import getter function to avoid circular import
def get_analyzer():
    """Get the global analyzer service instance."""
    from src.main import analyzer_service
    if analyzer_service is None:
        raise HTTPException(status_code=503, detail="Analyzer service not initialized")
    return analyzer_service

router = APIRouter()


class AnalysisRequest(BaseModel):
    """Request model for repository analysis."""
    
    repository_path: str = Field(..., description="Path to repository to analyze")
    include_mermaid: bool = Field(default=False, description="Generate Mermaid diagrams")
    include_lint: bool = Field(default=False, description="Run linting checks")
    exclude_patterns: Optional[List[str]] = Field(default=None, description="Additional exclude patterns")


class AnalysisJobResponse(BaseModel):
    """Response model for analysis job submission."""
    
    job_id: str
    status: str
    message: str
    timestamp: datetime


class AnalysisStatus(BaseModel):
    """Response model for analysis job status."""
    
    job_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@router.post("/run", response_model=AnalysisJobResponse)
async def run_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    analyzer: AnalyzerService = Depends(get_analyzer)
) -> AnalysisJobResponse:
    """
    Run repository analysis.
    
    Args:
        request: Analysis request parameters
        background_tasks: FastAPI background tasks
        analyzer: Analyzer service instance
    
    Returns:
        Job submission response with job_id
    """
    job_id = str(uuid.uuid4())
    
    # Add analysis task to background
    background_tasks.add_task(
        analyzer.run_analysis,
        job_id=job_id,
        repository_path=request.repository_path,
        include_mermaid=request.include_mermaid,
        include_lint=request.include_lint,
        exclude_patterns=request.exclude_patterns
    )
    
    return AnalysisJobResponse(
        job_id=job_id,
        status="queued",
        message=f"Analysis job {job_id} queued for {request.repository_path}",
        timestamp=datetime.utcnow()
    )


@router.get("/status/{job_id}", response_model=AnalysisStatus)
async def get_analysis_status(
    job_id: str,
    analyzer: AnalyzerService = Depends(get_analyzer)
) -> AnalysisStatus:
    """
    Get analysis job status.
    
    Args:
        job_id: Job identifier
        analyzer: Analyzer service instance
    
    Returns:
        Current status of the analysis job
    """
    status = analyzer.get_job_status(job_id)
    
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis job {job_id} not found"
        )
    
    return AnalysisStatus(**status)


@router.get("/results/{job_id}")
async def get_analysis_results(
    job_id: str,
    analyzer: AnalyzerService = Depends(get_analyzer)
) -> Dict[str, Any]:
    """
    Get analysis job results.
    
    Args:
        job_id: Job identifier
        analyzer: Analyzer service instance
    
    Returns:
        Analysis results
    """
    results = analyzer.get_job_results(job_id)
    
    if results is None:
        raise HTTPException(
            status_code=404,
            detail=f"Results for job {job_id} not found"
        )
    
    return results


@router.get("/jobs")
async def list_analysis_jobs(
    analyzer: AnalyzerService = Depends(get_analyzer)
) -> List[Dict[str, Any]]:
    """
    List all analysis jobs.
    
    Args:
        analyzer: Analyzer service instance
    
    Returns:
        List of analysis jobs with their status
    """
    return analyzer.list_jobs()


@router.delete("/jobs/{job_id}")
async def delete_analysis_job(
    job_id: str,
    analyzer: AnalyzerService = Depends(get_analyzer)
) -> Dict[str, str]:
    """
    Delete an analysis job and its results.
    
    Args:
        job_id: Job identifier
        analyzer: Analyzer service instance
    
    Returns:
        Deletion confirmation
    """
    success = analyzer.delete_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis job {job_id} not found"
        )
    
    return {"message": f"Job {job_id} deleted successfully"}
