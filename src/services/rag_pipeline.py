"""
RAG Pipeline service for FKS analysis using Retrieval-Augmented Generation.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import json
from pathlib import Path

logger = logging.getLogger(__name__)

from src.services.retrieval import RetrievalService
from src.services.ai_analyzer import AIAnalyzerService

# Import new RAG components
try:
    from src.rag.config import RAGConfig
    from src.rag.vector_store import VectorStoreManager
    from src.rag.loaders import FKSDocumentLoader
    from src.rag.ingestion_service import RAGIngestionService
    from src.rag.query_service import RAGQueryService
    RAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG components not available: {e}. Install dependencies: pip install -r requirements.txt")
    RAG_AVAILABLE = False


class RAGPipelineService:
    """Service for running RAG pipeline on FKS data for various analysis tasks."""
    
    def __init__(self):
        """Initialize RAG pipeline service."""
        self.retrieval_service = RetrievalService()
        self.ai_analyzer = AIAnalyzerService()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        
        # Initialize new RAG components if available
        if RAG_AVAILABLE:
            try:
                self.rag_config = RAGConfig()
                self.vector_store = VectorStoreManager(self.rag_config)
                self.document_loader = FKSDocumentLoader(self.rag_config)
                self.ingestion_service = RAGIngestionService(self.rag_config)
                self.query_service = RAGQueryService(self.rag_config, self.vector_store)
                logger.info("RAG components initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
                RAG_AVAILABLE = False
        else:
            self.rag_config = None
            self.vector_store = None
            self.document_loader = None
            self.ingestion_service = None
            self.query_service = None

        logger.info("RAGPipelineService initialized")
    
    async def run_rag_analysis(
        self,
        job_id: str,
        query: str,
        analysis_type: str = "project_management",
        scope: str = "all",
        custom_prompt: Optional[str] = None,
        max_retrieved: int = 10
    ) -> None:
        """
        Run RAG analysis on FKS data.
        
        Args:
            job_id: Unique job identifier
            query: Analysis query or focus area
            analysis_type: Type of analysis (project_management, standardization, documentation)
            scope: Scope of data retrieval (all, specific repo, or service)
            custom_prompt: Custom prompt for AI analysis
            max_retrieved: Maximum number of items to retrieve
        """
        try:
            # Update job status
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "query": query,
                "analysis_type": analysis_type,
                "scope": scope,
                "started_at": datetime.utcnow().isoformat(),
                "progress": 0.0
            }
            
            logger.info(f"Starting RAG analysis for job {job_id}: {query} (type: {analysis_type})")
            
            # Use new RAG query service if available
            if RAG_AVAILABLE and self.query_service:
                # Step 1: Query RAG system (40%)
                self.jobs[job_id]["progress"] = 0.4
                rag_result = self.query_service.query(query=query, k=max_retrieved)
                
                # Step 2: Prepare data for AI analysis (60%)
                self.jobs[job_id]["progress"] = 0.6
                analysis_data = {
                    "query": query,
                    "rag_answer": rag_result.get("answer", ""),
                    "rag_sources": rag_result.get("sources", []),
                    "context": f"Analysis of FKS services focusing on {analysis_type}"
                }
                
                # Step 3: Run AI analysis with RAG context (80%)
                self.jobs[job_id]["progress"] = 0.8
                ai_job_id = f"{job_id}_ai"
                
                # Use run_ai_analysis method (matches AIAnalyzerService interface)
                await self.ai_analyzer.run_ai_analysis(
                    job_id=ai_job_id,
                    repository_data=analysis_data,
                    analysis_type=analysis_type,
                    prompt=custom_prompt or f"Based on the RAG results, provide analysis for: {query}"
                )
                
                # Step 4: Get AI results (90%)
                self.jobs[job_id]["progress"] = 0.9
                ai_results = self.ai_analyzer.get_job_results(ai_job_id)
                
                # Step 5: Combine results (100%)
                self.jobs[job_id]["progress"] = 1.0
                self.jobs[job_id]["status"] = "completed"
                self.jobs[job_id]["results"] = {
                    "rag_result": rag_result,
                    "ai_insights": ai_results.get("ai_response", {}) if ai_results else {},
                    "combined_analysis": {
                        "query": query,
                        "rag_answer": rag_result.get("answer", ""),
                        "ai_analysis": ai_results.get("ai_response", {}) if ai_results else {},
                        "sources": rag_result.get("sources", [])
                    }
                }
                self.jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
                
                logger.info(f"RAG analysis completed for job {job_id}")
            else:
                # Fallback to old retrieval method
                logger.warning("RAG components not available, using fallback retrieval")
                
                # Step 1: Retrieve relevant data (40%)
                self.jobs[job_id]["progress"] = 0.4
                retrieved_data = await self.retrieval_service.retrieve_data(
                    query=query,
                    scope=scope,
                    max_results=max_retrieved,
                    include_content=True
                )
                
                # Step 2: Prepare data for AI analysis (60%)
                self.jobs[job_id]["progress"] = 0.6
                analysis_data = {
                    "query": query,
                    "retrieved_data": retrieved_data,
                    "context": f"Analysis of FKS services focusing on {analysis_type}"
                }
                
                # Step 3: Run AI analysis with retrieved data (80%)
                self.jobs[job_id]["progress"] = 0.8
                ai_job_id = f"{job_id}_ai"
                
                # Use run_ai_analysis method (matches AIAnalyzerService interface)
                await self.ai_analyzer.run_ai_analysis(
                    job_id=ai_job_id,
                    repository_data=analysis_data,
                    analysis_type=analysis_type,
                    prompt=custom_prompt
                )
                
                # Step 4: Get AI results (90%)
                self.jobs[job_id]["progress"] = 0.9
                ai_results = self.ai_analyzer.get_job_results(ai_job_id)
                
                # Step 5: Combine results (100%)
                self.jobs[job_id]["progress"] = 1.0
                self.jobs[job_id]["status"] = "completed"
                self.jobs[job_id]["results"] = {
                    "retrieved_data": retrieved_data,
                    "ai_insights": ai_results.get("ai_response", {}) if ai_results else {}
                }
                self.jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
                
                logger.info(f"RAG analysis completed for job {job_id} (fallback mode)")
        except Exception as e:
            logger.error(f"RAG analysis failed for job {job_id}: {e}", exc_info=True)
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
    
    async def run_project_management_analysis(
        self,
        job_id: str,
        query: str,
        scope: str = "all"
    ) -> None:
        """Specialized RAG analysis for project management."""
        await self.run_rag_analysis(
            job_id=job_id,
            query=query,
            analysis_type="project_management",
            scope=scope
        )
    
    async def run_standardization_analysis(
        self,
        job_id: str,
        query: str,
        scope: str = "all"
    ) -> None:
        """Specialized RAG analysis for standardization."""
        await self.run_rag_analysis(
            job_id=job_id,
            query=query,
            analysis_type="standardization",
            scope=scope
        )
    
    async def run_documentation_analysis(
        self,
        job_id: str,
        query: str,
        scope: str = "all"
    ) -> None:
        """Specialized RAG analysis for documentation."""
        await self.run_rag_analysis(
            job_id=job_id,
            query=query,
            analysis_type="documentation",
            scope=scope
        )
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a RAG job."""
        return self.jobs.get(job_id)
    
    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get results of a completed RAG job."""
        job = self.jobs.get(job_id)
        if job and job.get("status") == "completed":
            return job.get("results")
        return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all RAG jobs."""
        return [
            {
                "job_id": job_id,
                "status": job.get("status"),
                "query": job.get("query"),
                "analysis_type": job.get("analysis_type"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "progress": job.get("progress", 0.0)
            }
            for job_id, job in self.jobs.items()
        ]
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a RAG job."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            return True
        return False
    
    # New methods for RAG ingestion and querying
    def ingest_documents(
        self,
        root_dir: str = "/home/jordan/Documents/code/fks",
        include_code: bool = False,
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """Ingest documents into RAG vector store."""
        if not RAG_AVAILABLE or not self.ingestion_service:
            return {
                "status": "error",
                "message": "RAG components not available. Install dependencies: pip install -r requirements.txt"
            }
        
        try:
            return self.ingestion_service.ingest_documents(
                root_dir=root_dir,
                include_code=include_code,
                clear_existing=clear_existing
            )
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }
    
    def query_rag(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query the RAG system."""
        if not RAG_AVAILABLE or not self.query_service:
            return {
                "query": query,
                "answer": "RAG components not available. Install dependencies: pip install -r requirements.txt",
                "sources": [],
                "retrieved_count": 0
            }
        
        try:
            return self.query_service.query(query=query, k=k, filter=filter)
        except Exception as e:
            logger.error(f"Error querying RAG: {e}", exc_info=True)
            return {
                "query": query,
                "answer": f"Error querying RAG: {str(e)}",
                "sources": [],
                "retrieved_count": 0,
                "error": str(e)
            }
    
    def suggest_optimizations(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Suggest optimizations based on RAG query."""
        if not RAG_AVAILABLE or not self.query_service:
            return {
                "query": query,
                "suggestions": "RAG components not available. Install dependencies: pip install -r requirements.txt",
                "sources": []
            }
        
        try:
            return self.query_service.suggest_optimizations(query=query, context=context)
        except Exception as e:
            logger.error(f"Error generating optimizations: {e}", exc_info=True)
            return {
                "query": query,
                "suggestions": f"Error generating suggestions: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        if not RAG_AVAILABLE or not self.ingestion_service:
            return {
                "status": "not_available",
                "message": "RAG components not available"
            }
        
        try:
            ingestion_stats = self.ingestion_service.get_ingestion_stats()
            usage_stats = self.rag_config.get_usage_stats() if self.rag_config else {}
            
            return {
                **ingestion_stats,
                "usage": usage_stats
            }
        except Exception as e:
            logger.error(f"Error getting RAG stats: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
