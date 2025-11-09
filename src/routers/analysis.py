"""
Analysis API routes
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List
import uuid
from datetime import datetime
import logging

from src.models import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisStatus,
    FileAnalysisRequest,
    TaskGenerationRequest,
    TaskGenerationResponse,
    MetricsResponse,
    AITask,
    TaskPriority,
    FileIssue
)

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory storage for analysis results (use Redis/DB in production)
analysis_store: Dict[str, AnalysisResult] = {}


@router.post("/analyze/repo", response_model=AnalysisResult)
async def analyze_repository(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze a repository and generate reports.
    
    This endpoint triggers a comprehensive repository analysis including:
    - File structure analysis
    - Empty file/directory detection
    - Syntax checking
    - Optional linting
    - Mermaid diagram generation
    """
    analysis_id = str(uuid.uuid4())
    
    # Create initial result
    result = AnalysisResult(
        analysis_id=analysis_id,
        status=AnalysisStatus.PENDING,
        repo_path=request.repo_path,
        started_at=datetime.utcnow()
    )
    
    analysis_store[analysis_id] = result
    
    # Start background analysis
    background_tasks.add_task(
        run_analysis,
        analysis_id,
        request
    )
    
    logger.info(f"Started analysis {analysis_id} for {request.repo_path}")
    
    return result


@router.get("/analyze/reports/{analysis_id}", response_model=AnalysisResult)
async def get_analysis_report(analysis_id: str):
    """
    Get analysis report by ID.
    """
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    
    return analysis_store[analysis_id]


@router.get("/analyze/reports", response_model=List[AnalysisResult])
async def list_analysis_reports(limit: int = 10):
    """
    List recent analysis reports.
    """
    results = list(analysis_store.values())
    results.sort(key=lambda x: x.started_at, reverse=True)
    return results[:limit]


@router.post("/analyze/files", response_model=Dict)
async def analyze_files(request: FileAnalysisRequest):
    """
    Analyze specific files.
    """
    from pathlib import Path
    import ast
    
    results = {
        "total_files": len(request.file_paths),
        "valid_files": [],
        "broken_files": []
    }
    
    for file_path in request.file_paths:
        try:
            path = Path(file_path)
            if not path.exists():
                results["broken_files"].append({
                    "file": file_path,
                    "error": "File not found"
                })
                continue
            
            if request.check_syntax and path.suffix == ".py":
                try:
                    ast.parse(path.read_text())
                    results["valid_files"].append(file_path)
                except SyntaxError as e:
                    results["broken_files"].append({
                        "file": file_path,
                        "error": f"Syntax error: {str(e)}"
                    })
        except Exception as e:
            results["broken_files"].append({
                "file": file_path,
                "error": str(e)
            })
    
    return results


@router.post("/analyze/generate-tasks", response_model=TaskGenerationResponse)
async def generate_tasks(request: TaskGenerationRequest):
    """
    Generate AI agent tasks from analysis results.
    """
    if request.analysis_id not in analysis_store:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {request.analysis_id} not found"
        )
    
    analysis = analysis_store[request.analysis_id]
    
    if analysis.status != AnalysisStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail="Analysis not completed yet"
        )
    
    tasks = []
    
    # Generate tasks for empty files
    for file_path in analysis.empty_files:
        tasks.append(AITask(
            task_id=str(uuid.uuid4()),
            priority=TaskPriority.LOW,
            category="Empty Files",
            title=f"Populate empty file: {file_path}",
            description=f"File {file_path} is empty. Review and add appropriate content.",
            file_path=file_path,
            suggested_action="Review file purpose and add implementation or delete if unused."
        ))
    
    # Generate tasks for broken files
    for issue in analysis.broken_files:
        priority = TaskPriority.HIGH if "syntax" in issue.message.lower() else TaskPriority.MEDIUM
        tasks.append(AITask(
            task_id=str(uuid.uuid4()),
            priority=priority,
            category="Broken Files",
            title=f"Fix {issue.issue_type}: {issue.file}",
            description=f"{issue.message}",
            file_path=issue.file,
            suggested_action=f"Fix the {issue.issue_type} error in {issue.file}"
        ))
    
    # Generate tasks for lint issues
    for issue in analysis.lint_issues:
        tasks.append(AITask(
            task_id=str(uuid.uuid4()),
            priority=TaskPriority.MEDIUM,
            category="Code Quality",
            title=f"Fix linting issue: {issue.file}",
            description=issue.message,
            file_path=issue.file,
            suggested_action="Apply suggested lint fixes"
        ))
    
    # Filter by priority
    priority_order = {
        TaskPriority.CRITICAL: 4,
        TaskPriority.HIGH: 3,
        TaskPriority.MEDIUM: 2,
        TaskPriority.LOW: 1
    }
    
    min_priority_value = priority_order[request.min_priority]
    filtered_tasks = [
        t for t in tasks
        if priority_order[t.priority] >= min_priority_value
    ]
    
    # Count by priority
    tasks_by_priority = {
        "critical": len([t for t in filtered_tasks if t.priority == TaskPriority.CRITICAL]),
        "high": len([t for t in filtered_tasks if t.priority == TaskPriority.HIGH]),
        "medium": len([t for t in filtered_tasks if t.priority == TaskPriority.MEDIUM]),
        "low": len([t for t in filtered_tasks if t.priority == TaskPriority.LOW])
    }
    
    return TaskGenerationResponse(
        analysis_id=request.analysis_id,
        total_tasks=len(filtered_tasks),
        tasks_by_priority=tasks_by_priority,
        tasks=filtered_tasks
    )


@router.get("/analyze/metrics", response_model=MetricsResponse)
async def get_metrics(repo_path: str):
    """
    Get code quality metrics for a repository.
    """
    from pathlib import Path
    
    path = Path(repo_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Repository not found")
    
    # Calculate metrics
    total_files = 0
    total_lines = 0
    total_size = 0
    language_dist = {}
    
    for file in path.rglob("*"):
        if file.is_file() and not any(p in str(file) for p in ["__pycache__", ".git", "node_modules"]):
            total_files += 1
            try:
                total_size += file.stat().st_size
                total_lines += len(file.read_text().splitlines())
                
                ext = file.suffix or "no_extension"
                language_dist[ext] = language_dist.get(ext, 0) + 1
            except:
                pass
    
    avg_file_size = (total_size / total_files / 1024) if total_files > 0 else 0
    
    # Calculate health score (simple heuristic)
    health_score = 100.0
    # Deduct for empty files, broken files, etc.
    
    return MetricsResponse(
        repo_path=repo_path,
        total_files=total_files,
        total_lines=total_lines,
        average_file_size=avg_file_size,
        language_distribution=language_dist,
        health_score=health_score
    )


async def run_analysis(analysis_id: str, request: AnalysisRequest):
    """
    Background task to run repository analysis.
    """
    from pathlib import Path
    import ast
    import yaml
    import json
    
    try:
        result = analysis_store[analysis_id]
        result.status = AnalysisStatus.IN_PROGRESS
        
        repo_path = Path(request.repo_path)
        if not repo_path.exists():
            result.status = AnalysisStatus.FAILED
            logger.error(f"Repository not found: {request.repo_path}")
            return
        
        # Collect files
        files = []
        for file in repo_path.rglob("*"):
            if file.is_file() and not any(p in str(file) for p in request.exclude_patterns):
                files.append(file)
        
        result.total_files = len(files)
        
        # Find empty files
        for file in files:
            try:
                if file.stat().st_size == 0:
                    result.empty_files.append(str(file.relative_to(repo_path)))
            except:
                pass
        
        # Check syntax for Python files
        if request.check_syntax:
            for file in files:
                if file.suffix == ".py":
                    try:
                        ast.parse(file.read_text())
                    except SyntaxError as e:
                        result.broken_files.append(FileIssue(
                            file=str(file.relative_to(repo_path)),
                            issue_type="syntax_error",
                            severity="high",
                            message=str(e),
                            line_number=e.lineno
                        ))
                    except Exception as e:
                        result.broken_files.append(FileIssue(
                            file=str(file.relative_to(repo_path)),
                            issue_type="parse_error",
                            severity="medium",
                            message=str(e)
                        ))
        
        # Generate Mermaid diagrams
        if request.generate_mermaid:
            result.mermaid_diagrams["project_structure"] = generate_project_mermaid(repo_path)
        
        # Complete analysis
        result.status = AnalysisStatus.COMPLETED
        result.completed_at = datetime.utcnow()
        result.duration_seconds = (result.completed_at - result.started_at).total_seconds()
        
        logger.info(f"Completed analysis {analysis_id}")
        
    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {str(e)}", exc_info=True)
        result.status = AnalysisStatus.FAILED


def generate_project_mermaid(repo_path) -> str:
    """Generate simple project structure Mermaid diagram"""
    mermaid = "graph TD\n"
    dirs = [d for d in repo_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    for i, dir_path in enumerate(dirs[:10]):  # Limit to 10 dirs
        mermaid += f"    A[Project] --> {chr(66+i)}[{dir_path.name}]\n"
    return mermaid
