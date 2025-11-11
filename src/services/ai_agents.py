"""
AI Agents service for FKS repositories to enable automated fixes using RAG outputs.
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


class AIAgentService:
    """Service for running AI agents to perform automated fixes on FKS repositories."""
    
    def __init__(self, base_path: str = "/home/jordan/Documents/code/fks"):
        """Initialize AI agent service."""
        self.base_path = Path(base_path)
        self.retrieval_service = RetrievalService(base_path=base_path)
        self.rag_pipeline = RAGPipelineService()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.exclude_patterns = [
            ".git", "venv", "__pycache__", ".idea", ".vscode", "node_modules", 
            "dist", "build", ".env", ".gitignore", "*.log", "*.md5", "*.sha1"
        ]
        logger.info(f"AIAgentService initialized with base path: {base_path}")
    
    async def run_ai_agent_fixes(
        self,
        job_id: str,
        issue_type: str = "code issues",
        scope: str = "all",
        max_retrieved: int = 10
    ) -> None:
        """
        Run AI agent to analyze and fix issues in FKS repositories.
        
        Args:
            job_id: Unique job identifier
            issue_type: Type of issue to fix (e.g., code issues, documentation gaps)
            scope: Scope of analysis (all, specific repo, or service)
            max_retrieved: Maximum number of items to retrieve for analysis
        """
        try:
            # Update job status
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "issue_type": issue_type,
                "scope": scope,
                "started_at": datetime.utcnow(),
                "progress": 0.0
            }
            
            logger.info(f"Starting AI agent fixes for job {job_id}: {issue_type} in {scope}")
            
            # Step 1: Retrieve relevant data (40%)
            self.jobs[job_id]["progress"] = 0.4
            query = f"fix {issue_type}"
            retrieved_data = await self.retrieval_service.retrieve_data(
                query=query,
                scope=scope,
                max_results=max_retrieved,
                include_content=True
            )
            
            # Step 2: Run RAG pipeline for issue analysis (60%)
            self.jobs[job_id]["progress"] = 0.6
            rag_job_id = f"{job_id}_rag"
            custom_prompt = f"Analyze the following FKS repository data to identify and suggest fixes for {issue_type}. Provide specific code changes or actions to resolve issues. Data: {{data}}"
            await self.rag_pipeline.run_rag_analysis(
                job_id=rag_job_id,
                query=query,
                analysis_type="issue_fixing",
                scope=scope,
                custom_prompt=custom_prompt
            )
            
            # Step 3: Collect results and generate fix proposals (90%)
            self.jobs[job_id]["progress"] = 0.9
            rag_results = self.rag_pipeline.get_job_results(rag_job_id)
            if not rag_results:
                raise ValueError(f"RAG analysis failed for job {rag_job_id}")
            
            # Compile final results with fix proposals
            results = {
                "job_id": job_id,
                "issue_type": issue_type,
                "scope": scope,
                "started_at": self.jobs[job_id]["started_at"].isoformat(),
                "retrieved_data": retrieved_data,
                "fix_proposals": self._extract_fix_proposals(rag_results.get("ai_insights", {})),
                "status": "success"
            }
            
            # Complete job
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 1.0
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
            results["completed_at"] = self.jobs[job_id]["completed_at"].isoformat()
            
            self.results[job_id] = results
            
            logger.info(f"AI agent fixes analysis completed for job {job_id}")
        except Exception as e:
            logger.error(f"AI agent fixes analysis failed for job {job_id}: {e}", exc_info=True)
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
    
    def _extract_fix_proposals(self, ai_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract fix proposals from AI insights."""
        proposals = []
        if "text" in ai_insights:
            insights_text = ai_insights.get("text", "")
            # Simple extraction of fix proposals (can be enhanced based on AI response format)
            lines = insights_text.split("\n")
            current_proposal = None
            for line in lines:
                line = line.strip()
                if line.startswith("- ") or line.startswith("* ") or "fix" in line.lower():
                    if current_proposal:
                        proposals.append(current_proposal)
                    current_proposal = {
                        "description": line,
                        "file": "",
                        "action": "",
                        "code_snippet": ""
                    }
                elif current_proposal and line.startswith("File: "):
                    current_proposal["file"] = line.replace("File: ", "").strip()
                elif current_proposal and line.startswith("Action: "):
                    current_proposal["action"] = line.replace("Action: ", "").strip()
                elif current_proposal and line.startswith("```"):
                    # Start of code snippet
                    current_proposal["code_snippet"] += line + "\n"
                elif current_proposal and line == "```":
                    # End of code snippet
                    current_proposal["code_snippet"] += line
                elif current_proposal and current_proposal.get("code_snippet"):
                    current_proposal["code_snippet"] += line + "\n"
            if current_proposal:
                proposals.append(current_proposal)
        if not proposals:
            proposals.append({
                "description": "No specific fix proposals provided by AI analysis.",
                "file": "",
                "action": "",
                "code_snippet": ""
            })
        return proposals
    
    async def apply_fixes(
        self,
        job_id: str,
        fix_proposals: List[Dict[str, Any]],
        scope: str = "all",
        apply_changes: bool = False
    ) -> Dict[str, Any]:
        """
        Apply fixes based on AI agent proposals (placeholder for actual implementation).
        
        Args:
            job_id: Unique job identifier for tracking
            fix_proposals: List of fix proposals to apply
            scope: Scope of fixes (all, specific repo, or service)
            apply_changes: If True, apply changes; if False, simulate only
        
        Returns:
            Dictionary with status of applied fixes
        """
        try:
            # Update job status
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "action": "apply_fixes",
                "scope": scope,
                "apply_changes": apply_changes,
                "started_at": datetime.utcnow(),
                "progress": 0.0
            }
            
            logger.info(f"Applying AI agent fixes for job {job_id} in scope {scope}, apply_changes: {apply_changes}")
            
            # Placeholder for actual implementation of fixes
            self.jobs[job_id]["progress"] = 0.5
            
            # Simulate applying fixes
            applied_fixes = []
            for proposal in fix_proposals[:3]:  # Limit to 3 for demo
                applied_fixes.append({
                    "description": proposal.get("description", "Unknown fix"),
                    "file": proposal.get("file", "Unknown file"),
                    "status": "applied" if apply_changes else "simulated",
                    "details": "Placeholder: Fix applied successfully" if apply_changes else "Placeholder: Fix simulation successful"
                })
            
            # Complete job
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 1.0
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
            
            result = {
                "job_id": job_id,
                "action": "apply_fixes",
                "scope": scope,
                "apply_changes": apply_changes,
                "applied_fixes": applied_fixes,
                "status": "success",
                "completed_at": self.jobs[job_id]["completed_at"].isoformat()
            }
            
            self.results[job_id] = result
            
            logger.info(f"AI agent fixes applied for job {job_id}")
            return result
        except Exception as e:
            logger.error(f"Applying AI agent fixes failed for job {job_id}: {e}", exc_info=True)
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
