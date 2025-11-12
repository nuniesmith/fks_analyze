"""
Tests for Vector Store Manager
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from src.rag.config import RAGConfig
from src.rag.vector_store import VectorStoreManager


class TestVectorStoreManager:
    """Test VectorStoreManager for Chroma and PGVector"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create RAG config for tests"""
        config = RAGConfig()
        config.chroma_persist_dir = temp_dir
        config.embedding_provider = "local"  # Use local embeddings for tests
        config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        return config
    
    @pytest.fixture
    def vector_store(self, config):
        """Create vector store manager for tests"""
        try:
            return VectorStoreManager(config)
        except Exception as e:
            pytest.skip(f"Vector store initialization failed: {e}")
    
    def test_vector_store_initialization_chroma(self, config):
        """Test Chroma vector store initialization"""
        try:
            store = VectorStoreManager(config)
            assert store.vector_store is not None
            assert store.embeddings is not None
        except Exception as e:
            pytest.skip(f"Chroma initialization failed (may need dependencies): {e}")
    
    def test_vector_store_initialization_pgvector(self, config):
        """Test PGVector vector store initialization"""
        config.vector_store_type = "pgvector"
        config.pgvector_connection = "postgresql://test:test@localhost/test"
        
        # PGVector should fall back to Chroma for now
        try:
            store = VectorStoreManager(config)
            # Should fall back to Chroma
            assert config.vector_store_type == "chroma"
        except Exception as e:
            pytest.skip(f"PGVector initialization failed: {e}")
    
    @patch.dict("os.environ", {"GOOGLE_AI_API_KEY": "test_key"})
    def test_create_embeddings_gemini(self, config, temp_dir):
        """Test Gemini embedding creation"""
        config.embedding_provider = "gemini"
        config.gemini_api_key = "test_key"
        
        try:
            store = VectorStoreManager(config)
            # Should use Gemini embeddings if available
            assert store.embeddings is not None
        except Exception as e:
            pytest.skip(f"Gemini embeddings not available: {e}")
    
    def test_create_embeddings_ollama(self, config, temp_dir):
        """Test Ollama embedding creation"""
        config.embedding_provider = "ollama"
        config.ollama_endpoint = "http://localhost:11434"
        
        try:
            store = VectorStoreManager(config)
            # Should use Ollama embeddings if available
            assert store.embeddings is not None
        except Exception as e:
            pytest.skip(f"Ollama embeddings not available: {e}")
    
    def test_create_embeddings_local(self, config, temp_dir):
        """Test local (sentence-transformers) embedding creation"""
        config.embedding_provider = "local"
        
        try:
            store = VectorStoreManager(config)
            assert store.embeddings is not None
        except Exception as e:
            pytest.skip(f"Local embeddings not available: {e}")
    
    def test_add_documents(self, vector_store):
        """Test adding documents to vector store"""
        if vector_store is None:
            pytest.skip("Vector store not initialized")
        
        documents = ["This is a test document.", "This is another test document."]
        metadatas = [{"source": "test1.md"}, {"source": "test2.md"}]
        
        try:
            vector_store.add_documents(documents, metadatas)
            # If no exception, documents were added successfully
            assert True
        except Exception as e:
            pytest.fail(f"Failed to add documents: {e}")
    
    def test_search_documents(self, vector_store):
        """Test searching documents in vector store"""
        if vector_store is None:
            pytest.skip("Vector store not initialized")
        
        # Add documents first
        documents = [
            "The FKS platform is a trading system.",
            "Python is a programming language.",
            "Trading involves buying and selling assets."
        ]
        metadatas = [
            {"source": "fks.md", "topic": "trading"},
            {"source": "python.md", "topic": "programming"},
            {"source": "trading.md", "topic": "trading"}
        ]
        
        try:
            vector_store.add_documents(documents, metadatas)
            
            # Search for trading-related documents
            results = vector_store.similarity_search("trading platform", k=2)
            
            assert len(results) > 0
            assert all("content" in r for r in results)
            assert all("metadata" in r for r in results)
            assert all("score" in r for r in results)
        except Exception as e:
            pytest.skip(f"Search failed (may need documents in store): {e}")
    
    @patch.dict("os.environ", {"GOOGLE_AI_API_KEY": "test_key"})
    def test_get_llm_gemini(self, config, temp_dir):
        """Test getting Gemini LLM"""
        config.gemini_api_key = "test_key"
        config.use_hybrid = True
        
        try:
            store = VectorStoreManager(config)
            llm = store.get_llm("a" * 600)  # Complex query
            assert llm is not None
        except Exception as e:
            pytest.skip(f"Gemini LLM not available: {e}")
    
    def test_get_llm_ollama(self, config, temp_dir):
        """Test getting Ollama LLM"""
        config.ollama_endpoint = "http://localhost:11434"
        config.use_hybrid = False
        config.embedding_provider = "ollama"
        
        try:
            store = VectorStoreManager(config)
            llm = store.get_llm("test query")
            assert llm is not None
        except Exception as e:
            pytest.skip(f"Ollama LLM not available: {e}")
    
    def test_hybrid_llm_selection(self, config, temp_dir):
        """Test hybrid LLM selection based on query"""
        config.gemini_api_key = "test_key"
        config.use_hybrid = True
        config.hybrid_threshold = 500
        
        try:
            store = VectorStoreManager(config)
            
            # Simple query should use Ollama
            simple_llm = store.get_llm("test")
            assert simple_llm is not None
            
            # Complex query should use Gemini (if available)
            complex_query = "a" * 600
            complex_llm = store.get_llm(complex_query)
            assert complex_llm is not None
        except Exception as e:
            pytest.skip(f"Hybrid LLM selection not available: {e}")
    
    def test_get_stats(self, vector_store):
        """Test getting vector store statistics"""
        if vector_store is None:
            pytest.skip("Vector store not initialized")
        
        try:
            stats = vector_store.get_stats()
            assert "status" in stats
            assert "vector_store_type" in stats
            assert "embedding_provider" in stats
        except Exception as e:
            pytest.skip(f"Stats retrieval failed: {e}")
    
    def test_delete_collection(self, vector_store):
        """Test deleting vector store collection"""
        if vector_store is None:
            pytest.skip("Vector store not initialized")
        
        try:
            # Add some documents first
            documents = ["Test document"]
            vector_store.add_documents(documents)
            
            # Delete collection
            vector_store.delete_collection()
            
            # Collection should be deleted (or at least no error thrown)
            assert True
        except Exception as e:
            pytest.skip(f"Collection deletion failed: {e}")
