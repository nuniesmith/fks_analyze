"""
RAG Ingestion Service

Service for ingesting documents into the RAG vector store.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path

from .vector_store import VectorStoreManager
from .loaders import FKSDocumentLoader
from .config import RAGConfig


class RAGIngestionService:
    """Service for ingesting documents into RAG system"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = VectorStoreManager(config)
        self.loader = FKSDocumentLoader(config)
    
    def ingest_documents(
        self,
        root_dir: str = "/home/jordan/Documents/code/fks",
        include_code: bool = False,
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """Ingest all FKS documentation
        
        Args:
            root_dir: Root directory of FKS repos
            include_code: Whether to include code files
            clear_existing: Whether to clear existing vector store
        
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info("Starting document ingestion...")
        
        if clear_existing:
            logger.warning("Clearing existing vector store...")
            self.vector_store.delete_collection()
            self.vector_store._initialize_vector_store()
        
        # Load documentation
        logger.info("Loading documentation files...")
        documents = self.loader.load_documents(root_dir=root_dir)
        
        if include_code:
            logger.info("Loading code files...")
            code_documents = self.loader.load_code_files(root_dir=root_dir)
            documents.extend(code_documents)
        
        if not documents:
            logger.warning("No documents loaded")
            return {
                "status": "error",
                "message": "No documents loaded",
                "documents_loaded": 0
            }
        
        # Add to vector store
        logger.info(f"Adding {len(documents)} document chunks to vector store...")
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        try:
            # Add in batches to avoid memory issues
            batch_size = 100
            total_added = 0
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                self.vector_store.add_documents(
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )
                total_added += len(batch_texts)
                logger.info(f"Added batch {i // batch_size + 1}: {total_added}/{len(texts)} documents")
            
            logger.info(f"Successfully ingested {total_added} document chunks")
            
            return {
                "status": "success",
                "documents_loaded": len(documents),
                "documents_added": total_added,
                "include_code": include_code
            }
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "documents_loaded": len(documents),
                "documents_added": 0
            }
    
    def ingest_single_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single file
        
        Args:
            file_path: Path to file to ingest
        
        Returns:
            Dictionary with ingestion status
        """
        logger.info(f"Ingesting single file: {file_path}")
        
        documents = self.loader.load_single_file(file_path)
        
        if not documents:
            return {
                "status": "error",
                "message": "No documents loaded from file",
                "file_path": file_path
            }
        
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        try:
            self.vector_store.add_documents(
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully ingested {len(documents)} chunks from {file_path}")
            
            return {
                "status": "success",
                "documents_added": len(documents),
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error ingesting file: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "file_path": file_path
            }
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        stats = self.vector_store.get_stats()
        return {
            **stats,
            "config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "vector_store_type": self.config.vector_store_type,
                "embedding_provider": self.config.embedding_provider
            }
        }

