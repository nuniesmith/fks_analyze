"""
Documentation service for FKS repositories to generate MkDocs and Mermaid visuals.
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


class DocumentationService:
    """Service for generating documentation for FKS repositories."""
    
    def __init__(self, base_path: str = "/home/jordan/Documents/code/fks"):
        """Initialize documentation service."""
        self.base_path = Path(base_path)
        self.retrieval_service = RetrievalService(base_path=base_path)
        self.rag_pipeline = RAGPipelineService()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.exclude_patterns = [
            ".git", "venv", "__pycache__", ".idea", ".vscode", "node_modules", 
            "dist", "build", ".env", ".gitignore", "*.log", "*.md5", "*.sha1"
        ]
        logger.info(f"DocumentationService initialized with base path: {base_path}")
    
    async def run_documentation_analysis(
        self,
        job_id: str,
        scope: str = "all",
        focus_area: str = "mkdocs and diagrams",
        max_retrieved: int = 10
    ) -> None:
        """
        Run documentation analysis on FKS repositories.
        
        Args:
            job_id: Unique job identifier
            scope: Scope of analysis (all, specific repo, or service)
            focus_area: Specific focus within documentation (e.g., mkdocs, diagrams)
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
            
            logger.info(f"Starting documentation analysis for job {job_id}: {focus_area} in {scope}")
            
            # Step 1: Retrieve relevant data (40%)
            self.jobs[job_id]["progress"] = 0.4
            query = f"documentation {focus_area}"
            retrieved_data = await self.retrieval_service.retrieve_data(
                query=query,
                scope=scope,
                max_results=max_retrieved,
                include_content=True
            )
            
            # Step 2: Run RAG pipeline for documentation analysis (60%)
            self.jobs[job_id]["progress"] = 0.6
            rag_job_id = f"{job_id}_rag"
            await self.rag_pipeline.run_documentation_analysis(
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
                "documentation_content": rag_results.get("ai_insights", {"error": "No AI insights available"}),
                "mkdocs_structure": self._extract_mkdocs_structure(rag_results.get("ai_insights", {})),
                "mermaid_diagrams": self._extract_mermaid_diagrams(rag_results.get("ai_insights", {})),
                "status": "success"
            }
            
            # Complete job
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 1.0
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
            results["completed_at"] = self.jobs[job_id]["completed_at"].isoformat()
            
            self.results[job_id] = results
            
            logger.info(f"Documentation analysis completed for job {job_id}")
        except Exception as e:
            logger.error(f"Documentation analysis failed for job {job_id}: {e}", exc_info=True)
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
    
    def _extract_mkdocs_structure(self, ai_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Extract MkDocs structure from AI insights."""
        if "text" in ai_insights:
            insights_text = ai_insights.get("text", "")
            # Placeholder for extracting MkDocs structure
            # This could be improved based on AI response format
            return {"message": "MkDocs structure extraction not fully implemented", "structure": {}}
        return {"message": "No MkDocs structure provided", "structure": {}}
    
    def _extract_mermaid_diagrams(self, ai_insights: Dict[str, Any]) -> List[str]:
        """Extract Mermaid diagrams from AI insights."""
        diagrams = []
        if "text" in ai_insights:
            insights_text = ai_insights.get("text", "")
            # Simple extraction of Mermaid code blocks
            matches = re.findall(r"```mermaid\n(.*?)\n```", insights_text, re.DOTALL)
            diagrams.extend([match.strip() for match in matches if match.strip()])
        if not diagrams:
            diagrams.append("No Mermaid diagrams provided by AI analysis.")
        return diagrams
    
    async def generate_documentation(
        self,
        job_id: str,
        output_path: str,
        scope: str = "all",
        documentation_content: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate documentation files based on AI analysis (placeholder for actual implementation).
        
        Args:
            job_id: Unique job identifier for tracking
            output_path: Path to save generated documentation
            scope: Scope of documentation (all, specific repo, or service)
            documentation_content: Content from AI analysis to use for documentation
        
        Returns:
            Dictionary with status of documentation generation
        """
        try:
            # Update job status
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "action": "generate_documentation",
                "scope": scope,
                "output_path": output_path,
                "started_at": datetime.utcnow(),
                "progress": 0.0
            }
            
            logger.info(f"Generating documentation for job {job_id} in scope {scope}")
            
            # Placeholder for actual implementation of documentation generation
            self.jobs[job_id]["progress"] = 0.5
            
            # Simulate generating documentation
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # If content is provided, simulate writing some files
            generated_files = []
            if documentation_content and "text" in documentation_content:
                with open(output_dir / "index.md", "w") as f:
                    f.write("# FKS Documentation\n\nGenerated content based on AI analysis.")
                generated_files.append("index.md")
            
            # Complete job
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 1.0
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
            
            result = {
                "job_id": job_id,
                "action": "generate_documentation",
                "scope": scope,
                "output_path": output_path,
                "generated_files": generated_files,
                "status": "success",
                "completed_at": self.jobs[job_id]["completed_at"].isoformat()
            }
            
            self.results[job_id] = result
            
            logger.info(f"Documentation generated for job {job_id}")
            return result
        except Exception as e:
            logger.error(f"Generating documentation failed for job {job_id}: {e}", exc_info=True)
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
