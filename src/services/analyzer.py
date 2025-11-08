"""
Analyzer service for repository analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from collections import defaultdict

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class AnalyzerService:
    """Service for running repository analysis."""
    
    def __init__(self):
        """Initialize analyzer service."""
        self.settings = get_settings()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        logger.info("AnalyzerService initialized")
    
    async def run_analysis(
        self,
        job_id: str,
        repository_path: str,
        include_mermaid: bool = False,
        include_lint: bool = False,
        exclude_patterns: Optional[List[str]] = None
    ) -> None:
        """
        Run repository analysis (background task).
        
        Args:
            job_id: Unique job identifier
            repository_path: Path to repository
            include_mermaid: Generate Mermaid diagrams
            include_lint: Run linting
            exclude_patterns: Additional exclude patterns
        """
        try:
            # Update job status
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "repository_path": repository_path,
                "started_at": datetime.utcnow(),
                "progress": 0.0
            }
            
            logger.info(f"Starting analysis for job {job_id}: {repository_path}")
            
            # Convert to Path
            repo_path = Path(repository_path)
            
            if not repo_path.exists():
                raise ValueError(f"Repository path does not exist: {repository_path}")
            
            # Combine exclude patterns
            exclude = self.settings.EXCLUDE_PATTERNS.copy()
            if exclude_patterns:
                exclude.extend(exclude_patterns)
            
            # Run analysis steps
            results = {
                "job_id": job_id,
                "repository_path": str(repo_path),
                "started_at": self.jobs[job_id]["started_at"].isoformat(),
                "analysis": {}
            }
            
            # Step 1: File structure analysis (20%)
            self.jobs[job_id]["progress"] = 0.2
            results["analysis"]["file_structure"] = await self._analyze_file_structure(
                repo_path, exclude
            )
            
            # Step 2: Empty files and directories (40%)
            self.jobs[job_id]["progress"] = 0.4
            results["analysis"]["empty_items"] = await self._find_empty_items(
                repo_path, exclude
            )
            
            # Step 3: Broken files check (60%)
            self.jobs[job_id]["progress"] = 0.6
            results["analysis"]["broken_files"] = await self._check_broken_files(
                repo_path, exclude
            )
            
            # Step 4: Code metrics (80%)
            self.jobs[job_id]["progress"] = 0.8
            results["analysis"]["metrics"] = await self._calculate_metrics(
                repo_path, exclude
            )
            
            # Step 5: Optional mermaid diagrams (90%)
            if include_mermaid:
                self.jobs[job_id]["progress"] = 0.9
                results["analysis"]["diagrams"] = await self._generate_diagrams(
                    repo_path, exclude
                )
            
            # Complete
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 1.0
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
            
            results["completed_at"] = self.jobs[job_id]["completed_at"].isoformat()
            results["status"] = "success"
            
            self.results[job_id] = results
            
            logger.info(f"Analysis completed for job {job_id}")
            
        except Exception as e:
            logger.error(f"Analysis failed for job {job_id}: {e}", exc_info=True)
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["completed_at"] = datetime.utcnow()
    
    async def _analyze_file_structure(
        self, repo_path: Path, exclude: List[str]
    ) -> Dict[str, Any]:
        """Analyze file structure."""
        file_types = defaultdict(int)
        total_files = 0
        total_size = 0
        
        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                # Check if excluded
                if any(pattern in str(file_path) for pattern in exclude):
                    continue
                
                total_files += 1
                total_size += file_path.stat().st_size
                file_types[file_path.suffix or "no_extension"] += 1
        
        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_types": dict(file_types),
            "top_extensions": sorted(
                file_types.items(), key=lambda x: x[1], reverse=True
            )[:10]
        }
    
    async def _find_empty_items(
        self, repo_path: Path, exclude: List[str]
    ) -> Dict[str, List[str]]:
        """Find empty files and directories."""
        empty_files = []
        empty_dirs = []
        
        for path in repo_path.rglob("*"):
            rel_path = str(path.relative_to(repo_path))
            
            # Check if excluded
            if any(pattern in rel_path for pattern in exclude):
                continue
            
            if path.is_file() and path.stat().st_size == 0:
                empty_files.append(rel_path)
            elif path.is_dir() and not any(path.iterdir()):
                empty_dirs.append(rel_path)
        
        return {
            "empty_files": empty_files,
            "empty_directories": empty_dirs,
            "empty_file_count": len(empty_files),
            "empty_dir_count": len(empty_dirs)
        }
    
    async def _check_broken_files(
        self, repo_path: Path, exclude: List[str]
    ) -> Dict[str, List[Dict[str, str]]]:
        """Check for broken/invalid files."""
        broken_files = []
        
        # Check Python files for syntax errors
        for py_file in repo_path.rglob("*.py"):
            rel_path = str(py_file.relative_to(repo_path))
            
            if any(pattern in rel_path for pattern in exclude):
                continue
            
            try:
                compile(py_file.read_text(), str(py_file), 'exec')
            except SyntaxError as e:
                broken_files.append({
                    "file": rel_path,
                    "error": f"Syntax error: {e}",
                    "line": e.lineno or 0
                })
            except Exception as e:
                broken_files.append({
                    "file": rel_path,
                    "error": str(e),
                    "line": 0
                })
        
        return {
            "broken_files": broken_files,
            "broken_count": len(broken_files)
        }
    
    async def _calculate_metrics(
        self, repo_path: Path, exclude: List[str]
    ) -> Dict[str, Any]:
        """Calculate code metrics."""
        lines_by_extension = defaultdict(int)
        total_lines = 0
        
        # Count lines in code files
        code_extensions = {".py", ".rs", ".js", ".ts", ".java", ".go", ".cpp", ".c"}
        
        for file_path in repo_path.rglob("*"):
            if file_path.suffix in code_extensions:
                rel_path = str(file_path.relative_to(repo_path))
                
                if any(pattern in rel_path for pattern in exclude):
                    continue
                
                try:
                    lines = len(file_path.read_text().splitlines())
                    lines_by_extension[file_path.suffix] += lines
                    total_lines += lines
                except Exception:
                    pass
        
        return {
            "total_lines_of_code": total_lines,
            "lines_by_language": dict(lines_by_extension)
        }
    
    async def _generate_diagrams(
        self, repo_path: Path, exclude: List[str]
    ) -> Dict[str, str]:
        """Generate Mermaid diagrams (placeholder)."""
        return {
            "message": "Diagram generation not yet implemented",
            "supported": False
        }
    
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
