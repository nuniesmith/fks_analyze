"""
Unit tests for RAGPipelineService in fks_analyze.
"""

import unittest
import os
import sys
import asyncio

# Adjust sys.path to include the parent directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.rag_pipeline import RAGPipelineService

class TestRAGPipelineService(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.service = RAGPipelineService()
        
    def test_run_rag_analysis_initialization(self):
        """Test if RAG analysis job initializes correctly (actual API calls might fail without mocking)."""
        job_id = "test_rag_job"
        loop = asyncio.get_event_loop()
        # We expect this to fail without proper API key, but it tests initialization
        try:
            loop.run_until_complete(
                self.service.run_rag_analysis(job_id=job_id, query="test query", scope="all", max_retrieved=1)
            )
        except Exception as e:
            # Check if job was at least initialized
            status = self.service.get_job_status(job_id)
            self.assertIsNotNone(status)
            self.assertEqual(status["job_id"], job_id)
            self.assertEqual(status["query"], "test query")

if __name__ == '__main__':
    unittest.main()
