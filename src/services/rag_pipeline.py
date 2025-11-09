"""
RAG Pipeline service for FKS analysis using Retrieval-Augmented Generation.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import json

logger = logging.getLogger(__name__)


from src.services.retrieval import RetrievalService
from src.services.ai_analyzer import AIAnalyzerService


class RAGPipelineService:
    """Service for running RAG pipeline on FKS data for various analysis tasks."""
    
    def __init__(self):
        """Initialize RAG pipeline service."""
        self.retrieval_service = RetrievalService()
        self.ai_analyzer = AIAnalyzerService()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
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
                "started_at": datetime.utcnow(),
                "progress": 0.0
            }
            
            logger.info(f"Starting RAG analysis for job {job_id}: {query} (type: {analysis_type})")
            
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
            await self.ai_analyzer.run_ai_analysis(
                ai_job_id,
                analysis_data,
                analysis_type=analysis_type,
                prompt=custom_prompt
            )
            
            # Step 4: Collect results (90%)
            self.jobs[job_id]["progress"] = 0.9
            ai_results = self.ai_analyzer.get_job_results(ai_job_id)
            if not ai_results:
                raise ValueError(f"AI analysis failed for job {ai_job_id}")
            
            # Compile final results
            results = {
                "job_id": job_id,
                "query": query,
                "analysis_type": analysis_type,
                "scope": scope,
                "started_at": self.jobs[job_id]["started_at"].isoformat(),
                "retrieved_data": retrieved_data,
                "ai_insights": ai_results.get("ai_response", {"error": "No AI response available"}),
                "status": "success"
            }
            
            # Complete job
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 1.0
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
            results["completed_at"] = self.jobs[job_id]["completed_at"].isoformat()
            
            self.results[job_id] = results
            
            logger.info(f"RAG analysis completed for job {job_id}")
        except Exception as e:
            logger.error(f"RAG analysis failed for job {job_id}: {e}", exc_info=True)
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
    
    async def run_project_management_analysis(
        self,
        job_id: str,
        focus_area: str = "tasks and issues",
        scope: str = "all"
    ) -> None:
        """
        Specialized RAG analysis for project management.
        
        Args:
            job_id: Unique job identifier
            focus_area: Specific focus within project management (e.g., tasks, issues, status)
            scope: Scope of data retrieval (all, specific repo, or service)
        """
        query = f"project management {focus_area}"
        custom_prompt = f"Analyze the following FKS repository data for project management, focusing on {focus_area}. Identify tasks, issues, and project status. Provide actionable recommendations for project advancement. Data: {{data}}"
        await self.run_rag_analysis(
            job_id=job_id,
            query=query,
            analysis_type="project_management",
            scope=scope,
            custom_prompt=custom_prompt
        )
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        return self.jobs.get(job_id)
    
    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job results."""
        return self.results.get(job_id)
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs."""
        return list(self.jobs.values())
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its results."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            if job_id in self.results:
                del self.results[job_id]
            return True
        return False
