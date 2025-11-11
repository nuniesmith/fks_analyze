"""
AI Analyzer service for repository analysis using Google AI API.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import json
from google.cloud import aiplatform
from google.cloud.aiplatform import Model

logger = logging.getLogger(__name__)


class AIAnalyzerService:
    """Service for running AI-powered repository analysis using Google AI API."""
    
    def __init__(self):
        """Initialize AI analyzer service."""
        self.api_key = os.getenv("GOOGLE_AI_API_KEY", "")
        if not self.api_key:
            logger.warning("Google AI API key not found in environment variables. Set GOOGLE_AI_API_KEY.")
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.model_name = os.getenv("GOOGLE_AI_MODEL", "gemini-1.0-pro")
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        try:
            aiplatform.init(project=self.project_id, location=self.location)
            logger.info("AIAnalyzerService initialized with Google AI API")
        except Exception as e:
            logger.error(f"Failed to initialize Google AI API: {e}")
    
    async def run_ai_analysis(
        self,
        job_id: str,
        repository_data: Dict[str, Any],
        analysis_type: str = "project_management",
        prompt: Optional[str] = None
    ) -> None:
        """
        Run AI-powered analysis on repository data.
        
        Args:
            job_id: Unique job identifier
            repository_data: Data from repository analysis
            analysis_type: Type of analysis (project_management, standardization, documentation)
            prompt: Custom prompt for the AI model
        """
        try:
            # Update job status
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "analysis_type": analysis_type,
                "started_at": datetime.utcnow(),
                "progress": 0.0
            }
            
            logger.info(f"Starting AI analysis for job {job_id}: {analysis_type}")
            
            # Default prompts based on analysis type
            if prompt is None:
                if analysis_type == "project_management":
                    prompt = "Analyze the following repository data for project management insights. Identify tasks, issues, and project status. Suggest next steps and priorities. Data: {data}"
                elif analysis_type == "standardization":
                    prompt = "Review the repository structure and code for standardization. Suggest improvements for consistency across FKS services. Data: {data}"
                elif analysis_type == "documentation":
                    prompt = "Generate documentation content based on the repository data. Include MkDocs structure and Mermaid diagrams if relevant. Data: {data}"
                else:
                    prompt = "Analyze the repository data and provide insights. Data: {data}"
            
            # Fill in the data
            data_str = json.dumps(repository_data, indent=2)
            final_prompt = prompt.format(data=data_str)
            
            # Simulate progress
            self.jobs[job_id]["progress"] = 0.3
            
            # Call Google AI API
            response = await self._call_google_ai(final_prompt)
            
            # Update job status
            self.jobs[job_id]["progress"] = 0.8
            
            # Process response
            results = {
                "job_id": job_id,
                "analysis_type": analysis_type,
                "started_at": self.jobs[job_id]["started_at"].isoformat(),
                "ai_response": response,
                "status": "success"
            }
            
            # Complete job
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 1.0
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
            results["completed_at"] = self.jobs[job_id]["completed_at"].isoformat()
            
            self.results[job_id] = results
            
            logger.info(f"AI analysis completed for job {job_id}")
        except Exception as e:
            logger.error(f"AI analysis failed for job {job_id}: {e}", exc_info=True)
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
    
    async def _call_google_ai(self, prompt: str) -> Dict[str, Any]:
        """Call Google AI API with the provided prompt."""
        try:
            # Use asyncio to run in a non-blocking way
            loop = asyncio.get_event_loop()
            from functools import partial
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor() as pool:
                response = await loop.run_in_executor(
                    pool, 
                    partial(self._make_api_call, prompt)
                )
            return response
        except Exception as e:
            logger.error(f"Google AI API call failed: {e}")
            raise
    
    def _make_api_call(self, prompt: str) -> Dict[str, Any]:
        """Make the actual API call to Google AI."""
        try:
            endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{self.project_id}/locations/{self.location}/endpoints/{self.model_name}")
            response = endpoint.predict(instances=[{"prompt": prompt}])
            return {
                "text": response.predictions[0].get("content", ""),
                "metadata": response.raw_response.metadata if response.raw_response else {}
            }
        except Exception as e:
            logger.error(f"Error in Google AI API call: {e}")
            raise
    
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
