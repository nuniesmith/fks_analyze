"""
Enhanced tests for Retrieval Service
"""
import pytest
import asyncio
from unittest.mock import Mock, patch
from pathlib import Path
from src.services.retrieval import RetrievalService


class TestRetrievalServiceEnhanced:
    """Enhanced tests for Retrieval Service"""
    
    @pytest.fixture
    def retrieval_service(self, tmp_path):
        """Create retrieval service with temporary path"""
        # Create a test directory structure
        test_repo = tmp_path / "repo"
        test_repo.mkdir()
        (test_repo / "service1").mkdir()
        (test_repo / "service2").mkdir()
        
        # Create test files
        (test_repo / "service1" / "file1.py").write_text("def test_function(): pass")
        (test_repo / "service1" / "file2.md").write_text("# Test Document")
        (test_repo / "service2" / "config.py").write_text("CONFIG = {}")
        
        service = RetrievalService(base_path=str(tmp_path / "repo"))
        return service
    
    @pytest.mark.asyncio
    async def test_retrieve_data_basic(self, retrieval_service):
        """Test basic data retrieval"""
        results = await retrieval_service.retrieve_data(
            query="test",
            scope="all",
            max_results=5
        )
        
        assert isinstance(results, list)
        assert len(results) <= 5
        if results:
            assert "path" in results[0]
            assert "name" in results[0]
            assert "relevance_score" in results[0]
    
    @pytest.mark.asyncio
    async def test_retrieve_data_with_content(self, retrieval_service):
        """Test data retrieval with content"""
        results = await retrieval_service.retrieve_data(
            query="test",
            scope="all",
            max_results=5,
            include_content=True
        )
        
        assert isinstance(results, list)
        if results:
            assert "content" in results[0] or "content_error" in results[0]
    
    @pytest.mark.asyncio
    async def test_retrieve_data_scope_filtering(self, retrieval_service):
        """Test data retrieval with scope filtering"""
        results = await retrieval_service.retrieve_data(
            query="file",
            scope="service1",
            max_results=10
        )
        
        assert isinstance(results, list)
        # All results should be from service1
        for result in results:
            assert "service1" in result["path"]
    
    @pytest.mark.asyncio
    async def test_retrieve_data_empty_query(self, retrieval_service):
        """Test data retrieval with empty query"""
        results = await retrieval_service.retrieve_data(
            query="",
            scope="all",
            max_results=5
        )
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_retrieve_data_max_results(self, retrieval_service):
        """Test that max_results is respected"""
        results = await retrieval_service.retrieve_data(
            query="file",
            scope="all",
            max_results=2
        )
        
        assert len(results) <= 2
    
    @pytest.mark.asyncio
    async def test_get_repository_structure(self, retrieval_service):
        """Test getting repository structure"""
        structure = await retrieval_service.get_repository_structure(repo_path="all")
        
        assert isinstance(structure, dict)
        assert "name" in structure
        assert "directories" in structure
        assert "files" in structure
    
    @pytest.mark.asyncio
    async def test_get_repository_structure_specific_repo(self, retrieval_service):
        """Test getting structure for specific repository"""
        structure = await retrieval_service.get_repository_structure(repo_path="service1")
        
        assert isinstance(structure, dict)
        assert structure["name"] == "service1"
        assert "files" in structure
    
    @pytest.mark.asyncio
    async def test_get_service_data(self, retrieval_service):
        """Test getting service-specific data"""
        data = await retrieval_service.get_service_data(service_name="service1")
        
        assert isinstance(data, dict)
        assert "name" in data or "error" in data
    
    @pytest.mark.asyncio
    async def test_get_service_data_not_found(self, retrieval_service):
        """Test getting data for non-existent service"""
        data = await retrieval_service.get_service_data(service_name="nonexistent")
        
        assert isinstance(data, dict)
        assert "error" in data
    
    def test_calculate_relevance(self, retrieval_service):
        """Test relevance score calculation"""
        score = retrieval_service._calculate_relevance("test", "test_file.py", "path/to/test_file.py")
        assert score > 0
        
        # Filename match should have higher score
        score_filename = retrieval_service._calculate_relevance("test", "test_file.py", "path/to/file.py")
        score_path = retrieval_service._calculate_relevance("test", "file.py", "path/to/test_file.py")
        assert score_filename >= score_path
    
    @pytest.mark.asyncio
    async def test_exclude_patterns(self, retrieval_service, tmp_path):
        """Test that excluded patterns are filtered out"""
        # Create files that should be excluded
        test_repo = tmp_path / "repo"
        (test_repo / ".git" / "config").write_text("git config")
        (test_repo / "__pycache__" / "file.pyc").write_text("bytecode")
        (test_repo / "venv" / "lib").mkdir(parents=True)
        
        results = await retrieval_service.retrieve_data(
            query="config",
            scope="all",
            max_results=10
        )
        
        # Should not include .git or __pycache__ files
        for result in results:
            assert ".git" not in result["path"]
            assert "__pycache__" not in result["path"]
            assert "venv" not in result["path"]

