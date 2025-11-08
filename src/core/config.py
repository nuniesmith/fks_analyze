"""
Configuration management for FKS Analyze Service.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """Application settings."""
    
    # Service Configuration
    SERVICE_NAME: str = "fks_analyze"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8008
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://fkstrading.xyz",
        "https://api.fkstrading.xyz",
    ]
    
    # Analysis Configuration
    MAX_FILE_SIZE_MB: int = 10
    ANALYSIS_TIMEOUT_SECONDS: int = 300
    EXCLUDE_PATTERNS: List[str] = [
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        "venv",
        ".venv",
        "dist",
        "build",
        "*.egg-info",
    ]
    
    # Redis Configuration (optional, for caching)
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
