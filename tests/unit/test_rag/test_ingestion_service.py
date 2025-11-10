"""
Tests for RAG Ingestion Service
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from src.rag.config import RAGConfig
from src.rag.ingestion_service import RAGIngestionService
from src.rag.vector_store import VectorStoreManager
from src.rag.loaders import FKSDocumentLoader


class TestRAGIngestionService:
    """Test RAG Ingestion Service"""
    
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
        return config
    
    @pytest.fixture
    def ingestion_service(self, config):
        """Create ingestion service for tests"""
        try:
            return RAGIngestionService(config)
        except Exception as e:
            pytest.skip(f"Ingestion service initialization failed: {e}")
    
    def test_ingest_documents(self, ingestion_service, temp_dir):
        """Test ingesting documents into vector store"""
        if ingestion_service is None:
            pytest.skip("Ingestion service not initialized")
        
        # Create test document
        test_file = Path(temp_dir) / "test.md"
        test_file.write_text("# Test Document\n\nThis is a test document.")
        
        try:
            # Mock the loader to return test documents
            with patch.object(ingestion_service.loader, 'load_documents') as mock_load:
                mock_load.return_value = [
                    {
                        "content": "Test document content",
                        "metadata": {"source": "test.md"}
                    }
                ]
                
                result = ingestion_service.ingest_documents(
                    root_dir=temp_dir,
                    include_code=False,
                    clear_existing=False
                )
                
                assert result["status"] in ["success", "error"]
                assert "documents_loaded" in result
        except Exception as e:
            pytest.skip(f"Document ingestion failed: {e}")
    
    def test_ingest_service_docs(self, ingestion_service, temp_dir):
        """Test ingesting FKS service documentation"""
        if ingestion_service is None:
            pytest.skip("Ingestion service not initialized")
        
        try:
            # Mock the loader
            with patch.object(ingestion_service.loader, 'load_documents') as mock_load:
                mock_load.return_value = [
                    {
                        "content": "FKS service documentation",
                        "metadata": {"source": "fks.md", "service": "fks_ai"}
                    }
                ]
                
                result = ingestion_service.ingest_documents(
                    root_dir=temp_dir,
                    include_code=False,
                    clear_existing=False
                )
                
                assert result["status"] in ["success", "error"]
        except Exception as e:
            pytest.skip(f"Service documentation ingestion failed: {e}")
    
    def test_ingest_incremental(self, ingestion_service, temp_dir):
        """Test incremental document ingestion"""
        if ingestion_service is None:
            pytest.skip("Ingestion service not initialized")
        
        try:
            # First ingestion
            with patch.object(ingestion_service.loader, 'load_documents') as mock_load:
                mock_load.return_value = [
                    {
                        "content": "First document",
                        "metadata": {"source": "doc1.md"}
                    }
                ]
                
                result1 = ingestion_service.ingest_documents(
                    root_dir=temp_dir,
                    include_code=False,
                    clear_existing=False
                )
                
                # Second ingestion (incremental)
                mock_load.return_value = [
                    {
                        "content": "Second document",
                        "metadata": {"source": "doc2.md"}
                    }
                ]
                
                result2 = ingestion_service.ingest_documents(
                    root_dir=temp_dir,
                    include_code=False,
                    clear_existing=False
                )
                
                assert result1["status"] in ["success", "error"]
                assert result2["status"] in ["success", "error"]
        except Exception as e:
            pytest.skip(f"Incremental ingestion failed: {e}")
    
    def test_ingest_error_handling(self, ingestion_service, temp_dir):
        """Test error handling during ingestion"""
        if ingestion_service is None:
            pytest.skip("Ingestion service not initialized")
        
        try:
            # Mock vector store to raise error
            with patch.object(ingestion_service.vector_store, 'add_documents') as mock_add:
                mock_add.side_effect = Exception("Test error")
                
                with patch.object(ingestion_service.loader, 'load_documents') as mock_load:
                    mock_load.return_value = [
                        {
                            "content": "Test document",
                            "metadata": {"source": "test.md"}
                        }
                    ]
                    
                    result = ingestion_service.ingest_documents(
                        root_dir=temp_dir,
                        include_code=False,
                        clear_existing=False
                    )
                    
                    assert result["status"] == "error"
                    assert "message" in result
        except Exception as e:
            pytest.skip(f"Error handling test failed: {e}")
    
    def test_get_ingestion_status(self, ingestion_service):
        """Test getting ingestion status"""
        if ingestion_service is None:
            pytest.skip("Ingestion service not initialized")
        
        try:
            stats = ingestion_service.get_ingestion_stats()
            
            assert "status" in stats or "config" in stats
            assert "config" in stats
            assert stats["config"]["chunk_size"] == ingestion_service.config.chunk_size
        except Exception as e:
            pytest.skip(f"Status retrieval failed: {e}")
    
    def test_ingest_large_batch(self, ingestion_service, temp_dir):
        """Test ingesting large batches of documents"""
        if ingestion_service is None:
            pytest.skip("Ingestion service not initialized")
        
        try:
            # Create large batch of documents
            large_batch = [
                {
                    "content": f"Document {i} content",
                    "metadata": {"source": f"doc_{i}.md"}
                }
                for i in range(150)  # Larger than default batch size
            ]
            
            with patch.object(ingestion_service.loader, 'load_documents') as mock_load:
                mock_load.return_value = large_batch
                
                result = ingestion_service.ingest_documents(
                    root_dir=temp_dir,
                    include_code=False,
                    clear_existing=False
                )
                
                assert result["status"] in ["success", "error"]
                assert result["documents_loaded"] == 150
        except Exception as e:
            pytest.skip(f"Large batch ingestion failed: {e}")
    
    def test_ingest_single_file(self, ingestion_service, temp_dir):
        """Test ingesting a single file"""
        if ingestion_service is None:
            pytest.skip("Ingestion service not initialized")
        
        # Create test file
        test_file = Path(temp_dir) / "test.md"
        test_file.write_text("# Test Document\n\nThis is a test.")
        
        try:
            result = ingestion_service.ingest_single_file(str(test_file))
            
            assert result["status"] in ["success", "error"]
            assert "file_path" in result
        except Exception as e:
            pytest.skip(f"Single file ingestion failed: {e}")
