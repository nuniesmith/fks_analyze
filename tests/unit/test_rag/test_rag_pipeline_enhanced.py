"""
Enhanced tests for RAG Pipeline Service
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.services.rag_pipeline import RAGPipelineService
from src.services.retrieval import RetrievalService


class TestRAGPipelineServiceEnhanced:
    """Enhanced tests for RAG Pipeline Service"""
    
    @pytest.fixture
    def rag_service(self):
        """Create RAG pipeline service instance"""
        return RAGPipelineService()
    
    @pytest.fixture
    def mock_retrieval_service(self):
        """Mock retrieval service"""
        mock = Mock(spec=RetrievalService)
        mock.retrieve_data = AsyncMock(return_value=[
            {
                "path": "test/file.py",
                "name": "file.py",
                "relevance_score": 0.9,
                "content": "test content"
            }
        ])
        return mock
    
    @pytest.fixture
    def mock_ai_analyzer(self):
        """Mock AI analyzer service"""
        mock = Mock()
        mock.run_ai_analysis = AsyncMock()
        mock.get_job_results = Mock(return_value={
            "ai_response": "Test AI response",
            "status": "completed"
        })
        return mock
    
    @pytest.mark.asyncio
    async def test_run_rag_analysis_success(self, rag_service, mock_retrieval_service, mock_ai_analyzer):
        """Test successful RAG analysis"""
        # Replace services with mocks
        rag_service.retrieval_service = mock_retrieval_service
        rag_service.ai_analyzer = mock_ai_analyzer
        
        job_id = "test_job_123"
        await rag_service.run_rag_analysis(
            job_id=job_id,
            query="test query",
            analysis_type="project_management",
            scope="all",
            max_retrieved=5
        )
        
        # Check job status
        status = rag_service.get_job_status(job_id)
        assert status is not None
        assert status["status"] == "completed"
        assert status["query"] == "test query"
        
        # Check results
        results = rag_service.get_job_results(job_id)
        assert results is not None
        assert results["status"] == "success"
        assert "retrieved_data" in results
        assert "ai_insights" in results
    
    @pytest.mark.asyncio
    async def test_run_rag_analysis_retrieval_failure(self, rag_service, mock_ai_analyzer):
        """Test RAG analysis with retrieval failure"""
        mock_retrieval = Mock(spec=RetrievalService)
        mock_retrieval.retrieve_data = AsyncMock(side_effect=Exception("Retrieval failed"))
        
        rag_service.retrieval_service = mock_retrieval
        rag_service.ai_analyzer = mock_ai_analyzer
        
        job_id = "test_job_fail"
        await rag_service.run_rag_analysis(
            job_id=job_id,
            query="test query",
            scope="all"
        )
        
        # Check job status (should be failed)
        status = rag_service.get_job_status(job_id)
        assert status is not None
        assert status["status"] == "failed"
        assert "error" in status
    
    @pytest.mark.asyncio
    async def test_run_project_management_analysis(self, rag_service, mock_retrieval_service, mock_ai_analyzer):
        """Test project management analysis"""
        rag_service.retrieval_service = mock_retrieval_service
        rag_service.ai_analyzer = mock_ai_analyzer
        
        job_id = "test_pm_job"
        await rag_service.run_project_management_analysis(
            job_id=job_id,
            focus_area="tasks and issues",
            scope="all"
        )
        
        status = rag_service.get_job_status(job_id)
        assert status is not None
        assert status["status"] == "completed"
    
    def test_get_job_status_not_found(self, rag_service):
        """Test getting status for non-existent job"""
        status = rag_service.get_job_status("nonexistent")
        assert status is None
    
    def test_get_job_results_not_found(self, rag_service):
        """Test getting results for non-existent job"""
        results = rag_service.get_job_results("nonexistent")
        assert results is None
    
    def test_list_jobs(self, rag_service):
        """Test listing all jobs"""
        # Create some test jobs
        rag_service.jobs = {
            "job1": {"job_id": "job1", "status": "completed"},
            "job2": {"job_id": "job2", "status": "running"}
        }
        
        jobs = rag_service.list_jobs()
        assert len(jobs) == 2
        assert any(job["job_id"] == "job1" for job in jobs)
        assert any(job["job_id"] == "job2" for job in jobs)
    
    def test_delete_job(self, rag_service):
        """Test deleting a job"""
        job_id = "test_delete_job"
        rag_service.jobs[job_id] = {"job_id": job_id, "status": "completed"}
        rag_service.results[job_id] = {"job_id": job_id, "result": "test"}
        
        # Delete job
        deleted = rag_service.delete_job(job_id)
        assert deleted is True
        assert job_id not in rag_service.jobs
        assert job_id not in rag_service.results
        
        # Try to delete non-existent job
        deleted = rag_service.delete_job("nonexistent")
        assert deleted is False

