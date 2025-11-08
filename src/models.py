"""
Pydantic models for FKS Analyze service
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="Timestamp of health check")


class AnalysisRequest(BaseModel):
    """Request to analyze a repository"""
    repo_path: str = Field(..., description="Path to repository to analyze")
    exclude_patterns: List[str] = Field(
        default=["__pycache__", ".git", "node_modules", ".pytest_cache", ".venv"],
        description="Patterns to exclude from analysis"
    )
    generate_mermaid: bool = Field(default=True, description="Generate Mermaid diagrams")
    check_syntax: bool = Field(default=True, description="Check file syntax")
    run_linting: bool = Field(default=False, description="Run linting tools")
    output_format: str = Field(default="json", description="Output format (json, markdown)")


class FileAnalysisRequest(BaseModel):
    """Request to analyze specific files"""
    file_paths: List[str] = Field(..., description="List of file paths to analyze")
    check_syntax: bool = Field(default=True, description="Check syntax")
    check_imports: bool = Field(default=True, description="Check imports")


class AnalysisStatus(str, Enum):
    """Analysis status enum"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class FileIssue(BaseModel):
    """Issue found in a file"""
    file: str = Field(..., description="File path")
    issue_type: str = Field(..., description="Type of issue")
    severity: str = Field(..., description="Severity (high, medium, low)")
    message: str = Field(..., description="Issue description")
    line_number: Optional[int] = Field(None, description="Line number if applicable")


class AnalysisResult(BaseModel):
    """Result of repository analysis"""
    analysis_id: str = Field(..., description="Unique analysis ID")
    status: AnalysisStatus = Field(..., description="Analysis status")
    repo_path: str = Field(..., description="Repository path analyzed")
    started_at: datetime = Field(..., description="Analysis start time")
    completed_at: Optional[datetime] = Field(None, description="Analysis completion time")
    duration_seconds: Optional[float] = Field(None, description="Analysis duration")
    
    # Results
    total_files: int = Field(0, description="Total files analyzed")
    total_lines: int = Field(0, description="Total lines of code")
    empty_files: List[str] = Field(default_factory=list, description="Empty files found")
    empty_dirs: List[str] = Field(default_factory=list, description="Empty directories")
    broken_files: List[FileIssue] = Field(default_factory=list, description="Broken files")
    lint_issues: List[FileIssue] = Field(default_factory=list, description="Linting issues")
    
    # Metrics
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metrics and statistics"
    )
    
    # Generated artifacts
    mermaid_diagrams: Dict[str, str] = Field(
        default_factory=dict,
        description="Generated Mermaid diagrams"
    )
    reports: Dict[str, str] = Field(
        default_factory=dict,
        description="Generated reports (structure, summary, etc.)"
    )


class TaskPriority(str, Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AITask(BaseModel):
    """AI agent task for code fixes"""
    task_id: str = Field(..., description="Unique task ID")
    priority: TaskPriority = Field(..., description="Task priority")
    category: str = Field(..., description="Task category")
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Detailed description")
    file_path: Optional[str] = Field(None, description="Related file path")
    suggested_action: str = Field(..., description="Suggested fix/action")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TaskGenerationRequest(BaseModel):
    """Request to generate AI agent tasks"""
    analysis_id: str = Field(..., description="Analysis ID to generate tasks from")
    min_priority: TaskPriority = Field(
        default=TaskPriority.LOW,
        description="Minimum priority level for tasks"
    )


class TaskGenerationResponse(BaseModel):
    """Response with generated tasks"""
    analysis_id: str = Field(..., description="Source analysis ID")
    total_tasks: int = Field(..., description="Total tasks generated")
    tasks_by_priority: Dict[str, int] = Field(..., description="Task count by priority")
    tasks: List[AITask] = Field(..., description="Generated tasks")


class MetricsResponse(BaseModel):
    """Code quality metrics response"""
    repo_path: str = Field(..., description="Repository path")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # File metrics
    total_files: int = Field(0, description="Total files")
    total_lines: int = Field(0, description="Total lines of code")
    average_file_size: float = Field(0, description="Average file size in KB")
    
    # Quality metrics
    empty_file_count: int = Field(0, description="Number of empty files")
    broken_file_count: int = Field(0, description="Number of broken files")
    lint_issue_count: int = Field(0, description="Number of linting issues")
    
    # Language breakdown
    language_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="File count by language"
    )
    
    # Health score (0-100)
    health_score: float = Field(0, description="Overall code health score")
