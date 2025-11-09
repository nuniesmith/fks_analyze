"""
Standardization service for FKS repositories to ensure consistency across services.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)


from src.services.retrieval import RetrievalService
from src.services.rag_pipeline import RAGPipelineService


class StandardizationService:
    """Service for standardizing FKS repositories and services."""
    
    def __init__(self, base_path: str = "/home/jordan/Documents/code/fks"):
        """Initialize standardization service."""
        self.base_path = Path(base_path)
        self.retrieval_service = RetrievalService(base_path=base_path)
        self.rag_pipeline = RAGPipelineService()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.exclude_patterns = [
            ".git", "venv", "__pycache__", ".idea", ".vscode", "node_modules", 
            "dist", "build", ".env", ".gitignore", "*.log", "*.md5", "*.sha1"
        ]
        logger.info(f"StandardizationService initialized with base path: {base_path}")
    
    async def run_standardization_analysis(
        self,
        job_id: str,
        scope: str = "all",
        focus_area: str = "code consistency",
        max_retrieved: int = 10
    ) -> None:
        """
        Run standardization analysis on FKS repositories.
        
        Args:
            job_id: Unique job identifier
            scope: Scope of analysis (all, specific repo, or service)
            focus_area: Specific focus within standardization (e.g., code consistency, structure)
            max_retrieved: Maximum number of items to retrieve for analysis
        """
        try:
            # Update job status
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "scope": scope,
                "focus_area": focus_area,
                "started_at": datetime.utcnow(),
                "progress": 0.0
            }
            
            logger.info(f"Starting standardization analysis for job {job_id}: {focus_area} in {scope}")
            
            # Step 1: Retrieve relevant data (40%)
            self.jobs[job_id]["progress"] = 0.4
            query = f"standardization {focus_area}"
            retrieved_data = await self.retrieval_service.retrieve_data(
                query=query,
                scope=scope,
                max_results=max_retrieved,
                include_content=True
            )
            
            # Step 2: Run RAG pipeline for standardization analysis (60%)
            self.jobs[job_id]["progress"] = 0.6
            rag_job_id = f"{job_id}_rag"
            await self.rag_pipeline.run_standardization_analysis(
                job_id=rag_job_id,
                focus_area=focus_area,
                scope=scope
            )
            
            # Step 3: Collect results (90%)
            self.jobs[job_id]["progress"] = 0.9
            rag_results = self.rag_pipeline.get_job_results(rag_job_id)
            if not rag_results:
                raise ValueError(f"RAG analysis failed for job {rag_job_id}")
            
            # Compile final results
            results = {
                "job_id": job_id,
                "scope": scope,
                "focus_area": focus_area,
                "started_at": self.jobs[job_id]["started_at"].isoformat(),
                "retrieved_data": retrieved_data,
                "standardization_insights": rag_results.get("ai_insights", {"error": "No AI insights available"}),
                "recommendations": self._generate_recommendations(rag_results.get("ai_insights", {})),
                "status": "success"
            }
            
            # Complete job
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 1.0
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
            results["completed_at"] = self.jobs[job_id]["completed_at"].isoformat()
            
            self.results[job_id] = results
            
            logger.info(f"Standardization analysis completed for job {job_id}")
        except Exception as e:
            logger.error(f"Standardization analysis failed for job {job_id}: {e}", exc_info=True)
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
    
    def _generate_recommendations(self, ai_insights: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on AI insights."""
        recommendations = []
        if "text" in ai_insights:
            insights_text = ai_insights.get("text", "")
            # Simple extraction of recommendations (can be enhanced based on AI response format)
            lines = insights_text.split("\n")
            for line in lines:
                if line.strip().startswith("- ") or line.strip().startswith("* ") or "recommend" in line.lower():
                    recommendations.append(line.strip())
        if not recommendations:
            recommendations.append("No specific recommendations provided by AI analysis.")
        return recommendations
    
    async def apply_standardization_fixes(
        self,
        job_id: str,
        recommendations: List[str],
        scope: str = "all"
    ) -> Dict[str, Any]:
        """
        Apply standardization fixes based on AI recommendations (placeholder for actual implementation).
        
        Args:
            job_id: Unique job identifier for tracking
            recommendations: List of recommendations to apply
            scope: Scope of fixes (all, specific repo, or service)
        
        Returns:
            Dictionary with status of applied fixes
        """
        try:
            # Update job status
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "action": "apply_standardization_fixes",
                "scope": scope,
                "started_at": datetime.utcnow(),
                "progress": 0.0
            }
            
            logger.info(f"Applying standardization fixes for job {job_id} in scope {scope}")
            
            # Placeholder for actual implementation of fixes
            # This would involve parsing recommendations and applying changes to files
            self.jobs[job_id]["progress"] = 0.5
            
            # Simulate applying fixes
            applied_fixes = []
            for rec in recommendations[:3]:  # Limit to 3 for demo
                applied_fixes.append({
                    "recommendation": rec,
                    "status": "applied",
                    "details": "Placeholder: Fix applied successfully"
                })
            
            # Complete job
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 1.0
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
            
            result = {
                "job_id": job_id,
                "action": "apply_standardization_fixes",
                "scope": scope,
                "applied_fixes": applied_fixes,
                "status": "success",
                "completed_at": self.jobs[job_id]["completed_at"].isoformat()
            }
            
            self.results[job_id] = result
            
            logger.info(f"Standardization fixes applied for job {job_id}")
            return result
        except Exception as e:
            logger.error(f"Applying standardization fixes failed for job {job_id}: {e}", exc_info=True)
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
            return {"job_id": job_id, "status": "failed", "error": str(e)}
    
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
