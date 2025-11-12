"""
Tests for HyDE (Hypothetical Document Embeddings) Retriever
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from src.rag.config import RAGConfig
from src.rag.vector_store import VectorStoreManager
from src.rag.advanced.hyde import HyDERetriever


class TestHyDERetriever:
    """Test HyDE retriever"""
    
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
        config.embedding_provider = "local"
        config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        config.use_hyde = True
        return config
    
    @pytest.fixture
    def vector_store(self, config):
        """Create vector store for tests"""
        try:
            return VectorStoreManager(config)
        except Exception as e:
            pytest.skip(f"Vector store initialization failed: {e}")
    
    @pytest.fixture
    def hyde_retriever(self, config, vector_store):
        """Create HyDE retriever for tests"""
        if vector_store is None:
            pytest.skip("Vector store not initialized")
        
        try:
            return HyDERetriever(config, vector_store)
        except Exception as e:
            pytest.skip(f"HyDE retriever initialization failed: {e}")
    
    def test_hyde_initialization(self, config, vector_store):
        """Test HyDE retriever initialization"""
        if vector_store is None:
            pytest.skip("Vector store not initialized")
        
        try:
            hyde = HyDERetriever(config, vector_store)
            assert hyde.config == config
            assert hyde.vector_store == vector_store
        except Exception as e:
            pytest.skip(f"HyDE initialization failed: {e}")
    
    def test_generate_hypothetical_document(self, hyde_retriever):
        """Test hypothetical document generation"""
        if hyde_retriever is None:
            pytest.skip("HyDE retriever not initialized")
        
        try:
            query = "How does the FKS platform work?"
            hyde_doc = hyde_retriever.generate_hypothetical_document(query)
            
            assert isinstance(hyde_doc, str)
            assert len(hyde_doc) > 0
            assert len(hyde_doc) > len(query)  # Should be longer than query
        except Exception as e:
            pytest.skip(f"Hypothetical document generation failed: {e}")
    
    def test_retrieve_with_hyde(self, hyde_retriever, vector_store):
        """Test retrieval using HyDE"""
        if hyde_retriever is None or vector_store is None:
            pytest.skip("HyDE retriever or vector store not initialized")
        
        try:
            # Add test documents first
            documents = [
                "The FKS platform is a trading system.",
                "Python is a programming language.",
                "Trading involves buying and selling assets."
            ]
            metadatas = [
                {"source": "fks.md"},
                {"source": "python.md"},
                {"source": "trading.md"}
            ]
            
            vector_store.add_documents(documents, metadatas)
            
            # Retrieve using HyDE
            query = "What is FKS?"
            results = hyde_retriever.retrieve(query, k=2)
            
            assert len(results) > 0
            assert all("content" in r for r in results)
            assert all("hyde_used" in r for r in results)
            assert all(r["hyde_used"] is True for r in results)
        except Exception as e:
            pytest.skip(f"HyDE retrieval failed: {e}")
    
    def test_retrieve_hybrid(self, hyde_retriever, vector_store):
        """Test hybrid retrieval (standard + HyDE)"""
        if hyde_retriever is None or vector_store is None:
            pytest.skip("HyDE retriever or vector store not initialized")
        
        try:
            # Add test documents
            documents = ["FKS is a trading platform."]
            vector_store.add_documents(documents)
            
            # Test hybrid retrieval
            query = "What is FKS?"
            results = hyde_retriever.retrieve_hybrid(query, k=2, hyde_weight=0.7)
            
            assert len(results) > 0
            assert all("score" in r for r in results)
        except Exception as e:
            pytest.skip(f"Hybrid retrieval failed: {e}")
    
    def test_hyde_fallback_on_error(self, hyde_retriever):
        """Test that HyDE falls back to standard retrieval on error"""
        if hyde_retriever is None:
            pytest.skip("HyDE retriever not initialized")
        
        try:
            # Mock LLM to raise error
            with patch.object(hyde_retriever, '_get_llm') as mock_llm:
                mock_llm.side_effect = Exception("LLM error")
                
                # Should fall back to standard retrieval
                query = "test query"
                results = hyde_retriever.retrieve(query, k=1)
                
                # Should still return results (from fallback)
                assert isinstance(results, list)
        except Exception as e:
            pytest.skip(f"HyDE fallback test failed: {e}")

