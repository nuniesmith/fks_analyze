"""RAG (Retrieval-Augmented Generation) Module for FKS Analyze"""

from .config import RAGConfig
from .vector_store import VectorStoreManager
from .loaders import FKSDocumentLoader
from .ingestion_service import RAGIngestionService
from .query_service import RAGQueryService

# Optional advanced RAG imports
try:
    from .advanced.hyde import HyDERetriever
    __all__ = [
        "RAGConfig",
        "VectorStoreManager",
        "FKSDocumentLoader",
        "RAGIngestionService",
        "RAGQueryService",
        "HyDERetriever"
    ]
except ImportError:
    __all__ = [
        "RAGConfig",
        "VectorStoreManager",
        "FKSDocumentLoader",
        "RAGIngestionService",
        "RAGQueryService"
    ]

