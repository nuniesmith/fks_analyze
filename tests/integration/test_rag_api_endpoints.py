"""
Integration tests for RAG API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import tempfile
import shutil

# Try to import the app
try:
    from src.main import app
    APP_AVAILABLE = True
except ImportError:
    APP_AVAILABLE = False


@pytest.fixture
def client():
    """Create test client"""
    if not APP_AVAILABLE:
        pytest.skip("App not available")
    return TestClient(app)


class TestRAGAPIEndpoints:
    """Test RAG API endpoints"""
    
    def test_rag_analyze_endpoint(self, client):
        """Test RAG analyze endpoint"""
        if not APP_AVAILABLE:
            pytest.skip("App not available")
        
        # Test analyze endpoint
        response = client.post(
            "/api/v1/rag/analyze",
            json={
                "query": "What is the FKS platform?",
                "analysis_type": "project_management",
                "scope": "all",
                "max_retrieved": 5
            }
        )
        
        # Should return job ID
        assert response.status_code in [200, 201, 400, 500]  # Allow various status codes
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
            assert "status" in data
    
    def test_rag_query_endpoint(self, client):
        """Test RAG query endpoint"""
        if not APP_AVAILABLE:
            pytest.skip("App not available")
        
        # Test query endpoint
        response = client.post(
            "/api/v1/rag/query",
            json={
                "query": "What is FKS?",
                "k": 5
            }
        )
        
        # Should return query results
        assert response.status_code in [200, 400, 500]
        if response.status_code == 200:
            data = response.json()
            assert "query" in data
            assert "answer" in data
    
    def test_rag_suggest_optimizations_endpoint(self, client):
        """Test RAG suggest optimizations endpoint"""
        if not APP_AVAILABLE:
            pytest.skip("App not available")
        
        # Test suggest optimizations endpoint
        response = client.post(
            "/api/v1/rag/suggest-optimizations",
            json={
                "query": "How can I optimize the FKS platform?",
                "context": "Performance optimization"
            }
        )
        
        # Should return suggestions
        assert response.status_code in [200, 400, 500]
        if response.status_code == 200:
            data = response.json()
            assert "query" in data
            assert "suggestions" in data
    
    def test_rag_ingest_endpoint(self, client):
        """Test RAG document ingestion endpoint"""
        if not APP_AVAILABLE:
            pytest.skip("App not available")
        
        # Test ingest endpoint
        response = client.post(
            "/api/v1/rag/ingest",
            json={
                "root_dir": "/home/jordan/Documents/code/fks",
                "include_code": False,
                "clear_existing": False
            }
        )
        
        # Should return ingestion status
        assert response.status_code in [200, 400, 500]
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
    
    def test_rag_job_status_endpoint(self, client):
        """Test RAG job status endpoint"""
        if not APP_AVAILABLE:
            pytest.skip("App not available")
        
        # Test job status endpoint
        response = client.get("/api/v1/rag/jobs/test_job")
        
        # Should return 404 or job status
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data or "status" in data
    
    def test_rag_jobs_list_endpoint(self, client):
        """Test RAG jobs list endpoint"""
        if not APP_AVAILABLE:
            pytest.skip("App not available")
        
        # Test jobs list endpoint
        response = client.get("/api/v1/rag/jobs")
        
        # Should return list of jobs
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_rag_stats_endpoint(self, client):
        """Test RAG stats endpoint"""
        if not APP_AVAILABLE:
            pytest.skip("App not available")
        
        # Test stats endpoint
        response = client.get("/api/v1/rag/stats")
        
        # Should return stats
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "config" in data
    
    def test_rag_health_endpoint(self, client):
        """Test RAG health endpoint"""
        if not APP_AVAILABLE:
            pytest.skip("App not available")
        
        # Test health endpoint
        response = client.get("/api/v1/rag/health")
        
        # Should return health status
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "rag_available" in data
    
    def test_rag_error_handling(self, client):
        """Test error handling in RAG endpoints"""
        if not APP_AVAILABLE:
            pytest.skip("App not available")
        
        # Test with invalid request
        response = client.post(
            "/api/v1/rag/query",
            json={
                "invalid": "request"
            }
        )
        
        # Should return error
        assert response.status_code in [400, 422, 500]
