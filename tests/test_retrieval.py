"""
Unit tests for RetrievalService in fks_analyze.
"""

import unittest
import os
import sys
import asyncio

# Adjust sys.path to include the parent directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.retrieval import RetrievalService

class TestRetrievalService(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Use a test-specific base path or mock data if needed
        self.service = RetrievalService(base_path="/home/jordan/Documents/code/fks")
        
    def test_retrieve_data_basic(self):
        """Test basic data retrieval with a simple query."""
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            self.service.retrieve_data(query="monitor", scope="all", max_results=5)
        )
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) <= 5)
        if results:
            self.assertIn("path", results[0])
            self.assertIn("name", results[0])
            self.assertIn("relevance_score", results[0])
    
    def test_retrieve_data_empty_query(self):
        """Test data retrieval with an empty query."""
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            self.service.retrieve_data(query="", scope="all", max_results=5)
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)
    
    def test_get_repository_structure(self):
        """Test getting repository structure."""
        loop = asyncio.get_event_loop()
        structure = loop.run_until_complete(
            self.service.get_repository_structure(repo_path="repo")
        )
        self.assertIsInstance(structure, dict)
        self.assertIn("name", structure)
        self.assertIn("directories", structure)
        self.assertIn("files", structure)

if __name__ == '__main__':
    unittest.main()
