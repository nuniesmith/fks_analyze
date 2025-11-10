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


# Request/Response Models
class RAGAnalyzeRequest(BaseModel):
    """Request model for RAG document analysis."""
    query: str = Field(..., description="Analysis query or focus area")
    analysis_type: str = Field(default="project_management", description="Type of analysis")
    scope: str = Field(default="all", description="Scope of data retrieval")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt for AI analysis")
    max_retrieved: int = Field(default=10, description="Maximum number of items to retrieve")


class RAGQueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., description="Natural language question")
    k: Optional[int] = Field(None, description="Number of documents to retrieve")
    filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter for retrieval")


class RAGIngestRequest(BaseModel):
    """Request model for RAG document ingestion."""
    root_dir: Optional[str] = Field(None, description="Root directory of FKS repos")
    include_code: bool = Field(default=False, description="Include code files")
    clear_existing: bool = Field(default=False, description="Clear existing vector store")


class RAGOptimizationRequest(BaseModel):
    """Request model for optimization suggestions."""
    query: str = Field(..., description="Query about optimizations")
    context: Optional[str] = Field(None, description="Additional context")


class RAGEvaluationRequest(BaseModel):
    """Request model for RAG evaluation."""
    queries: List[str] = Field(..., description="List of queries to evaluate")
    ground_truths: Optional[List[str]] = Field(None, description="Optional ground truth answers")


class RAGJobResponse(BaseModel):
    """Response model for RAG job submission."""
    job_id: str
    status: str
    message: str
    timestamp: datetime


@router.post("/analyze", response_model=RAGJobResponse)
async def analyze_rag(
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
        RAGJobResponse with job ID and status
    """
    job_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        rag.run_rag_analysis,
        job_id=job_id,
        query=request.query,
        analysis_type=request.analysis_type,
        scope=request.scope,
        custom_prompt=request.custom_prompt,
        max_retrieved=request.max_retrieved
    )
    
    return RAGJobResponse(
        job_id=job_id,
        status="queued",
        message=f"RAG analysis job {job_id} queued",
        timestamp=datetime.utcnow()
    )


@router.post("/query", response_model=Dict[str, Any])
async def query_rag(
    request: RAGQueryRequest,
    rag: RAGPipelineService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Query the RAG system with a natural language question.
    
    Args:
        request: RAG query request parameters
        rag: RAG pipeline service instance
    
    Returns:
        Dictionary with query results and generated response
    """
    try:
        result = rag.query_rag(
            query=request.query,
            k=request.k,
            filter=request.filter
        )
        return result
    except Exception as e:
        logger.error(f"Error querying RAG: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=Dict[str, Any])
async def ingest_documents(
    request: RAGIngestRequest,
    background_tasks: BackgroundTasks,
    rag: RAGPipelineService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Ingest documents into the RAG vector store.
    
    Args:
        request: RAG ingestion request parameters
        background_tasks: FastAPI background tasks
        rag: RAG pipeline service instance
    
    Returns:
        Dictionary with ingestion status
    """
    try:
        # Run ingestion in background
        result = rag.ingest_documents(
            root_dir=request.root_dir,
            include_code=request.include_code,
            clear_existing=request.clear_existing
        )
        return result
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/suggest-optimizations", response_model=Dict[str, Any])
async def suggest_optimizations(
    request: RAGOptimizationRequest,
    rag: RAGPipelineService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Suggest optimizations based on RAG query.
    
    Args:
        request: Optimization request parameters
        rag: RAG pipeline service instance
    
    Returns:
        Dictionary with optimization suggestions
    """
    try:
        result = rag.suggest_optimizations(
            query=request.query,
            context=request.context
        )
        return result
    except Exception as e:
        logger.error(f"Error generating optimizations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_rag_status(
    job_id: str,
    rag: RAGPipelineService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Get RAG job status.
    
    Args:
        job_id: RAG job identifier
        rag: RAG pipeline service instance
    
    Returns:
        Current status of the RAG job
    """
    status = rag.get_job_status(job_id)
    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"RAG job {job_id} not found"
        )
    return status


@router.get("/jobs/{job_id}/results", response_model=Dict[str, Any])
async def get_rag_results(
    job_id: str,
    rag: RAGPipelineService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Get RAG job results.
    
    Args:
        job_id: RAG job identifier
        rag: RAG pipeline service instance
    
    Returns:
        RAG analysis results
    """
    results = rag.get_job_results(job_id)
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Results for RAG job {job_id} not found"
        )
    return results


@router.get("/jobs", response_model=List[Dict[str, Any]])
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


@router.delete("/jobs/{job_id}", response_model=Dict[str, str])
async def delete_rag_job(
    job_id: str,
    rag: RAGPipelineService = Depends(get_rag_service)
) -> Dict[str, str]:
    """
    Delete a RAG job and its results.
    
    Args:
        job_id: RAG job identifier
        rag: RAG pipeline service instance
    
    Returns:
        Success message
    """
    success = rag.delete_job(job_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"RAG job {job_id} not found"
        )
    return {"message": f"RAG job {job_id} deleted successfully"}


@router.get("/stats", response_model=Dict[str, Any])
async def get_rag_stats(
    rag: RAGPipelineService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Get RAG system statistics.
    
    Args:
        rag: RAG pipeline service instance
    
    Returns:
        RAG system statistics including vector store stats and usage
    """
    return rag.get_rag_stats()


@router.get("/health", response_model=Dict[str, Any])
async def rag_health(
    rag: RAGPipelineService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Health check for RAG system.
    
    Args:
        rag: RAG pipeline service instance
    
    Returns:
        Health status of RAG system
    """
    stats = rag.get_rag_stats()
    return {
        "status": "healthy" if stats.get("status") != "not_available" else "degraded",
        "rag_available": stats.get("status") != "not_available",
        "stats": stats,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_rag(
    request: RAGEvaluationRequest,
    rag: RAGPipelineService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Evaluate RAG system using RAGAS metrics.
    
    Args:
        request: RAG evaluation request with queries and optional ground truths
        rag: RAG pipeline service instance
    
    Returns:
        Dictionary with evaluation scores and metrics
    """
    try:
        from src.rag.evaluation.ragas_eval import RAGASEvaluator
        from src.rag.query_service import RAGQueryService
        from src.rag.config import RAGConfig
        from src.rag.vector_store import VectorStoreManager
        
        # Get RAG components from pipeline service
        config = RAGConfig()
        vector_store = rag.vector_store if hasattr(rag, 'vector_store') else None
        
        if vector_store is None:
            # Initialize vector store if not available
            vector_store = VectorStoreManager(config)
        
        query_service = RAGQueryService(config, vector_store)
        evaluator = RAGASEvaluator(config, query_service)
        
        # Run evaluation
        results = evaluator.evaluate_rag(
            queries=request.queries,
            ground_truths=request.ground_truths
        )
        
        return {
            "evaluation": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    except ImportError as e:
        logger.warning(f"RAGAS evaluation not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="RAGAS evaluation framework not available. Install with: pip install ragas datasets"
        )
    except Exception as e:
        logger.error(f"Error evaluating RAG: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
