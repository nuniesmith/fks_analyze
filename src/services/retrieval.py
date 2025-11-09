"""
Retrieval service for FKS repository data to support RAG system.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for retrieving relevant data from FKS repositories for RAG system."""
    
    def __init__(self, base_path: str = "/home/jordan/Documents/code/fks"):
        """Initialize retrieval service with base path to FKS repositories."""
        self.base_path = Path(base_path)
        self.exclude_patterns = [
            ".git", "venv", "__pycache__", ".idea", ".vscode", "node_modules", 
            "dist", "build", ".env", ".gitignore", "*.log", "*.md5", "*.sha1"
        ]
        logger.info(f"RetrievalService initialized with base path: {base_path}")
    
    async def retrieve_data(
        self, 
        query: str, 
        scope: str = "all", 
        max_results: int = 10, 
        include_content: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant data based on query from FKS repositories.
        
        Args:
            query: Search query to find relevant files or data
            scope: Scope of search (all, specific repo, or service)
            max_results: Maximum number of results to return
            include_content: Include file contents in results (False by default to save tokens)
        
        Returns:
            List of dictionaries containing file metadata and optionally content
        """
        try:
            logger.info(f"Retrieving data for query: {query}, scope: {scope}")
            search_path = self.base_path
            if scope != "all" and scope:
                search_path = self.base_path / scope
                if not search_path.exists():
                    logger.warning(f"Scope path {search_path} does not exist, falling back to all")
                    search_path = self.base_path
            
            results = []
            query_lower = query.lower()
            
            for file_path in search_path.rglob("*"):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(self.base_path))
                    if any(pattern in rel_path for pattern in self.exclude_patterns):
                        continue
                    
                    file_name = file_path.name.lower()
                    if query_lower in file_name or query_lower in rel_path.lower():
                        result = {
                            "path": rel_path,
                            "name": file_path.name,
                            "size": file_path.stat().st_size,
                            "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                            "relevance_score": self._calculate_relevance(query_lower, file_name, rel_path)
                        }
                        if include_content:
                            try:
                                result["content"] = file_path.read_text()[:1000]  # Limit content to save tokens
                            except Exception as e:
                                result["content_error"] = str(e)
                        results.append(result)
                        if len(results) >= max_results:
                            break
            
            # Sort by relevance score
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results[:max_results]
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            return []
    
    def _calculate_relevance(self, query: str, file_name: str, full_path: str) -> float:
        """Calculate relevance score for a file based on query match."""
        score = 0.0
        if query in file_name:
            score += 2.0  # Higher weight for filename match
        if query in full_path:
            score += 1.0  # Additional weight for path match
        return score
    
    async def get_repository_structure(self, repo_path: str) -> Dict[str, Any]:
        """
        Get the structure of a specific repository or all repositories.
        
        Args:
            repo_path: Path to the repository or 'all' for full FKS structure
        
        Returns:
            Dictionary representing repository structure
        """
        try:
            if repo_path == "all":
                search_path = self.base_path
            else:
                search_path = self.base_path / repo_path
                if not search_path.exists():
                    logger.error(f"Repository path {search_path} does not exist")
                    return {"error": f"Repository {repo_path} not found"}
            
            structure = {
                "name": search_path.name,
                "path": str(search_path.relative_to(self.base_path)),
                "directories": [],
                "files": []
            }
            
            for item in search_path.iterdir():
                if any(pattern in str(item) for pattern in self.exclude_patterns):
                    continue
                if item.is_dir():
                    structure["directories"].append({
                        "name": item.name,
                        "path": str(item.relative_to(self.base_path))
                    })
                elif item.is_file():
                    structure["files"].append({
                        "name": item.name,
                        "path": str(item.relative_to(self.base_path)),
                        "size": item.stat().st_size,
                        "last_modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })
            
            return structure
        except Exception as e:
            logger.error(f"Error getting repository structure: {e}")
            return {"error": str(e)}
    
    async def get_service_data(self, service_name: str) -> Dict[str, Any]:
        """
        Get data specific to a named FKS service.
        
        Args:
            service_name: Name of the service (e.g., fks_monitor)
        
        Returns:
            Dictionary containing service-specific data
        """
        try:
            service_path = None
            for repo in self.base_path.iterdir():
                if repo.is_dir():
                    potential_path = repo / service_name
                    if potential_path.exists() and potential_path.is_dir():
                        service_path = potential_path
                        break
                    # Check under core or tools
                    for subdir in ["core", "tools"]:
                        potential_path = repo / subdir / service_name
                        if potential_path.exists() and potential_path.is_dir():
                            service_path = potential_path
                            break
                    if service_path:
                        break
            
            if not service_path:
                return {"error": f"Service {service_name} not found in FKS repositories"}
            
            return await self.get_repository_structure(str(service_path.relative_to(self.base_path)))
        except Exception as e:
            logger.error(f"Error getting service data for {service_name}: {e}")
            return {"error": str(e)}
