"""
Tests for RAG Configuration
"""
import pytest
import os
from unittest.mock import patch
from datetime import datetime

from src.rag.config import RAGConfig


class TestRAGConfig:
    """Test RAG configuration with Gemini/Ollama hybrid support"""
    
    def test_config_initialization_defaults(self):
        """Test RAG config initialization with defaults"""
        config = RAGConfig()
        
        assert config.vector_store_type == "chroma"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.top_k == 5
        assert config.similarity_threshold == 0.7
        assert config.llm_temperature == 0.7
        assert config.max_tokens == 1000
        assert config.use_hyde is True
        assert config.use_raptor is False
        assert config.use_reranking is False
        assert config.evaluation_enabled is True
    
    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "test_key_123"})
    def test_config_gemini_api_key(self):
        """Test Gemini API key configuration"""
        config = RAGConfig()
        
        assert config.gemini_api_key == "test_key_123"
        assert config.gemini_embedding_model == "models/embedding-001"
        assert config.gemini_llm_model == "gemini-2.5-flash"
    
    @patch.dict(os.environ, {"OLLAMA_HOST": "http://localhost:11434", "OLLAMA_MODEL": "qwen2.5"})
    def test_config_ollama_endpoint(self):
        """Test Ollama endpoint configuration"""
        config = RAGConfig()
        
        assert config.ollama_endpoint == "http://localhost:11434"
        assert config.ollama_model == "qwen2.5"
    
    @patch.dict(os.environ, {"RAG_USE_HYBRID": "true", "RAG_HYBRID_THRESHOLD": "500"})
    def test_config_hybrid_routing(self):
        """Test hybrid routing logic (Gemini vs Ollama)"""
        config = RAGConfig()
        
        assert config.use_hybrid is True
        assert config.hybrid_threshold == 500
        
        # Simple query should use Ollama
        simple_query = "test"
        assert config.should_use_gemini(simple_query) is False
        
        # Complex query should use Gemini
        complex_query = "a" * 600  # 600 characters
        assert config.should_use_gemini(complex_query) is True
    
    def test_should_use_gemini_simple_query(self):
        """Test that simple queries use Ollama"""
        config = RAGConfig()
        config.use_hybrid = True
        config.hybrid_threshold = 500
        
        simple_query = "What is FKS?"
        assert config.should_use_gemini(simple_query) is False
    
    def test_should_use_gemini_complex_query(self):
        """Test that complex queries use Gemini"""
        config = RAGConfig()
        config.use_hybrid = True
        config.hybrid_threshold = 500
        config.gemini_api_key = "test_key"
        
        complex_query = "Explain in detail how the FKS trading platform works, including all components, services, and their interactions" * 10
        assert len(complex_query) > 500
        assert config.should_use_gemini(complex_query) is True
    
    def test_should_use_gemini_usage_limit(self):
        """Test that usage limits trigger Ollama fallback"""
        config = RAGConfig()
        config.use_hybrid = True
        config.hybrid_threshold = 500
        config.gemini_free_tier_limit = 10
        config.gemini_api_key = "test_key"
        
        # Simulate reaching the limit
        today = datetime.now().strftime("%Y-%m-%d")
        config.gemini_usage_tracker[today] = 10
        
        complex_query = "a" * 600
        assert config.should_use_gemini(complex_query) is False  # Should fall back to Ollama
    
    @patch.dict(os.environ, {
        "GOOGLE_AI_API_KEY": "test_key",
        "RAG_USE_HYBRID": "true",
        "RAG_HYBRID_THRESHOLD": "500",
        "RAG_CHUNK_SIZE": "2000",
        "RAG_CHUNK_OVERLAP": "400"
    })
    def test_config_env_variables(self):
        """Test configuration from environment variables"""
        config = RAGConfig()
        
        assert config.gemini_api_key == "test_key"
        assert config.use_hybrid is True
        assert config.hybrid_threshold == 500
        assert config.chunk_size == 2000
        assert config.chunk_overlap == 400
    
    def test_track_gemini_usage(self):
        """Test Gemini usage tracking"""
        config = RAGConfig()
        config.gemini_api_key = "test_key"
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Track usage
        config.track_gemini_usage("test query")
        
        assert config.gemini_usage_tracker[today] == 1
        
        # Track more usage
        config.track_gemini_usage("another query")
        assert config.gemini_usage_tracker[today] == 2
    
    def test_get_usage_stats(self):
        """Test usage statistics retrieval"""
        config = RAGConfig()
        config.gemini_api_key = "test_key"
        config.gemini_free_tier_limit = 100
        
        today = datetime.now().strftime("%Y-%m-%d")
        config.gemini_usage_tracker[today] = 25
        
        stats = config.get_usage_stats()
        
        assert stats["today"] == today
        assert stats["gemini_usage_today"] == 25
        assert stats["gemini_free_tier_limit"] == 100
        assert stats["usage_percentage"] == 25.0
        assert stats["embedding_provider"] == config.embedding_provider
        assert stats["use_hybrid"] == config.use_hybrid
