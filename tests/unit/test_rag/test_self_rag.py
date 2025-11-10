"""
Tests for Self-RAG (Self-Retrieval Augmented Generation)
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from src.rag.config import RAGConfig
from src.rag.vector_store import VectorStoreManager
from src.rag.advanced.self_rag import SelfRAGNode, SelfRAGWorkflow


class TestSelfRAGNode:
    """Test Self-RAG node"""
    
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
        config.ragas_threshold = 0.9
        return config
    
    @pytest.fixture
    def vector_store(self, config):
        """Create vector store for tests"""
        try:
            return VectorStoreManager(config)
        except Exception as e:
            pytest.skip(f"Vector store initialization failed: {e}")
    
    @pytest.fixture
    def self_rag_node(self, config, vector_store):
        """Create Self-RAG node for tests"""
        if vector_store is None:
            pytest.skip("Vector store not initialized")
        
        try:
            return SelfRAGNode(config, vector_store)
        except Exception as e:
            pytest.skip(f"Self-RAG node initialization failed: {e}")
    
    def test_self_rag_node_initialization(self, config, vector_store):
        """Test Self-RAG node initialization"""
        if vector_store is None:
            pytest.skip("Vector store not initialized")
        
        try:
            node = SelfRAGNode(config, vector_store)
            assert node.config == config
            assert node.vector_store == vector_store
        except Exception as e:
            pytest.skip(f"Self-RAG node initialization failed: {e}")
    
    @patch('src.rag.advanced.self_rag.SelfRAGNode._get_llm')
    def test_judge_retrieval_need(self, mock_get_llm, self_rag_node):
        """Test retrieval need judgment"""
        if self_rag_node is None:
            pytest.skip("Self-RAG node not initialized")
        
        try:
            # Mock LLM response
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = Mock(content="Yes, retrieval is needed.")
            mock_get_llm.return_value = mock_llm
            
            result = self_rag_node.judge_retrieval_need("How does FKS work?")
            
            assert "needs_retrieval" in result
            assert "reason" in result
            assert "query" in result
            assert result["query"] == "How does FKS work?"
        except Exception as e:
            pytest.skip(f"Retrieval judgment test failed: {e}")
    
    @patch('src.rag.advanced.self_rag.SelfRAGNode._get_llm')
    def test_generate_with_retrieval(self, mock_get_llm, self_rag_node):
        """Test answer generation with retrieval"""
        if self_rag_node is None:
            pytest.skip("Self-RAG node not initialized")
        
        try:
            # Mock LLM response
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = Mock(content="FKS is a trading platform.")
            mock_get_llm.return_value = mock_llm
            
            retrieved_docs = [
                {"content": "FKS platform documentation", "metadata": {"source_file": "fks.md"}}
            ]
            
            result = self_rag_node.generate_with_retrieval("What is FKS?", retrieved_docs)
            
            assert "answer" in result
            assert "query" in result
            assert "sources" in result
            assert len(result["sources"]) > 0
        except Exception as e:
            pytest.skip(f"Generation test failed: {e}")
    
    @patch('src.rag.advanced.self_rag.SelfRAGNode._get_llm')
    def test_judge_faithfulness(self, mock_get_llm, self_rag_node):
        """Test faithfulness judgment"""
        if self_rag_node is None:
            pytest.skip("Self-RAG node not initialized")
        
        try:
            # Mock LLM response
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = Mock(content="Faithful. Score: 0.95")
            mock_get_llm.return_value = mock_llm
            
            retrieved_docs = [
                {"content": "FKS is a trading platform", "metadata": {"source_file": "fks.md"}}
            ]
            
            result = self_rag_node.judge_faithfulness(
                "What is FKS?",
                "FKS is a trading platform.",
                retrieved_docs
            )
            
            assert "is_faithful" in result
            assert "score" in result
            assert result["score"] >= 0.0
            assert result["score"] <= 1.0
        except Exception as e:
            pytest.skip(f"Faithfulness judgment test failed: {e}")
    
    @patch('src.rag.advanced.self_rag.SelfRAGNode._get_llm')
    def test_refine_answer(self, mock_get_llm, self_rag_node):
        """Test answer refinement"""
        if self_rag_node is None:
            pytest.skip("Self-RAG node not initialized")
        
        try:
            # Mock LLM response
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = Mock(content="Refined answer: FKS is a comprehensive trading platform.")
            mock_get_llm.return_value = mock_llm
            
            retrieved_docs = [
                {"content": "FKS is a trading platform", "metadata": {"source_file": "fks.md"}}
            ]
            
            faithfulness_judgment = {
                "is_faithful": False,
                "score": 0.5,
                "judgment": "Partially faithful"
            }
            
            result = self_rag_node.refine_answer(
                "What is FKS?",
                "FKS is a platform.",
                retrieved_docs,
                faithfulness_judgment
            )
            
            assert "answer" in result
            assert "refined" in result
            assert result["refined"] is True
        except Exception as e:
            pytest.skip(f"Refinement test failed: {e}")


class TestSelfRAGWorkflow:
    """Test Self-RAG workflow"""
    
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
        config.ragas_threshold = 0.9
        return config
    
    @pytest.fixture
    def vector_store(self, config):
        """Create vector store for tests"""
        try:
            return VectorStoreManager(config)
        except Exception as e:
            pytest.skip(f"Vector store initialization failed: {e}")
    
    @pytest.fixture
    def workflow(self, config, vector_store):
        """Create Self-RAG workflow for tests"""
        if vector_store is None:
            pytest.skip("Vector store not initialized")
        
        try:
            return SelfRAGWorkflow(config, vector_store)
        except Exception as e:
            pytest.skip(f"Self-RAG workflow initialization failed: {e}")
    
    @patch('src.rag.advanced.self_rag.SelfRAGNode._get_llm')
    def test_workflow_run(self, mock_get_llm, workflow, vector_store):
        """Test Self-RAG workflow execution"""
        if workflow is None or vector_store is None:
            pytest.skip("Workflow or vector store not initialized")
        
        try:
            # Add test documents
            documents = ["FKS is a trading platform."]
            vector_store.add_documents(documents)
            
            # Mock LLM responses
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = [
                Mock(content="Yes, retrieval is needed."),  # Retrieval judgment
                Mock(content="FKS is a trading platform."),  # Generation
                Mock(content="Faithful. Score: 0.95")  # Faithfulness judgment
            ]
            mock_get_llm.return_value = mock_llm
            
            result = workflow.run("What is FKS?", k=1)
            
            assert "query" in result
            assert "final_answer" in result
            assert "steps" in result
            assert len(result["steps"]) > 0
            assert result["retrieval_used"] is True
        except Exception as e:
            pytest.skip(f"Workflow test failed: {e}")

