"""
Vector Store Manager for RAG

Manages vector stores (Chroma/PGVector) with Gemini API and Ollama hybrid embeddings.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not available, install with: pip install chromadb")

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import OllamaEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available, install with: pip install langchain langchain-community")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available, install with: pip install sentence-transformers")

from .config import RAGConfig


class VectorStoreManager:
    """Manage vector store for RAG"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = None
        self.embeddings = self._create_embeddings()
        self._initialize_vector_store()
    
    def _create_embeddings(self):
        """Create embeddings model with Gemini API and Ollama hybrid support"""
        if self.config.embedding_provider == "gemini":
            if not self.config.gemini_api_key:
                logger.warning("Gemini API key not set, falling back to local embeddings")
                return self._create_local_embeddings()
            
            logger.info("Using Gemini API embeddings")
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                return GoogleGenerativeAIEmbeddings(
                    model=self.config.gemini_embedding_model,
                    google_api_key=self.config.gemini_api_key
                )
            except ImportError:
                logger.warning("langchain-google-genai not available, falling back to local embeddings")
                return self._create_local_embeddings()
        elif self.config.embedding_provider == "ollama":
            return self._create_ollama_embeddings()
        elif self.config.embedding_provider == "hybrid":
            # Hybrid: Use Gemini for complex, Ollama for simple
            # Default to Ollama, switch dynamically based on query
            logger.info("Using hybrid embeddings (Gemini + Ollama) - defaulting to Ollama")
            return self._create_ollama_embeddings()
        else:
            logger.info(f"Using local embeddings: {self.config.embedding_model}")
            return self._create_local_embeddings()
    
    def _create_ollama_embeddings(self):
        """Create Ollama embeddings"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, falling back to local embeddings")
            return self._create_local_embeddings()
        
        try:
            logger.info(f"Using Ollama embeddings from {self.config.ollama_endpoint}")
            return OllamaEmbeddings(
                model=self.config.ollama_model,
                base_url=self.config.ollama_endpoint
            )
        except Exception as e:
            logger.error(f"Failed to create Ollama embeddings: {e}, falling back to local")
            return self._create_local_embeddings()
    
    def _create_local_embeddings(self):
        """Create local sentence-transformers embeddings"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ValueError("No embedding provider available. Install sentence-transformers or configure Gemini/Ollama")
        
        logger.info(f"Using local embeddings: {self.config.embedding_model}")
        # Create a wrapper for sentence-transformers to work with LangChain
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=self.config.embedding_model
        )
    
    def _initialize_vector_store(self):
        """Initialize vector store (Chroma or PGVector)"""
        try:
            if self.config.vector_store_type == "chroma":
                self._initialize_chroma()
            elif self.config.vector_store_type == "pgvector":
                self._initialize_pgvector()
            else:
                raise ValueError(f"Unknown vector store type: {self.config.vector_store_type}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _initialize_chroma(self):
        """Initialize ChromaDB vector store"""
        if not CHROMA_AVAILABLE or not LANGCHAIN_AVAILABLE:
            raise ValueError("ChromaDB or LangChain not available. Install with: pip install chromadb langchain")
        
        # Create persist directory
        persist_dir = Path(self.config.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at {persist_dir}")
        try:
            self.vector_store = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=self.embeddings,
                collection_name="fks_documents"
            )
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            # Try creating a new collection
            try:
                self.vector_store = Chroma(
                    persist_directory=str(persist_dir),
                    embedding_function=self.embeddings,
                    collection_name="fks_documents_new"
                )
                logger.info("ChromaDB initialized with new collection")
            except Exception as e2:
                logger.error(f"Failed to initialize ChromaDB with new collection: {e2}")
                raise
    
    def _initialize_pgvector(self):
        """Initialize PGVector vector store"""
        if not self.config.pgvector_connection:
            raise ValueError("PGVector connection string not provided")
        
        logger.warning("PGVector initialization not yet implemented, falling back to Chroma")
        self.config.vector_store_type = "chroma"
        self._initialize_chroma()
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Add documents to vector store"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        try:
            if metadatas is None:
                metadatas = [{} for _ in documents]
            
            self.vector_store.add_texts(texts=documents, metadatas=metadatas)
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        k = k or self.config.top_k
        
        try:
            # Use hybrid routing if enabled
            if self.config.use_hybrid and self.config.embedding_provider == "hybrid":
                # Check if we should use Gemini for this query
                if self.config.should_use_gemini(query):
                    # Switch to Gemini embeddings temporarily
                    logger.debug(f"Using Gemini for query: {query[:50]}...")
                    self.config.track_gemini_usage(query)
                    # Note: In a full implementation, we'd create a temporary Gemini embedding
                    # For now, use the default embeddings
            
            # ChromaDB similarity_search_with_score returns (Document, score) tuples
            # Lower score = more similar in ChromaDB
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                # ChromaDB uses distance (lower is better), convert to similarity (higher is better)
                # For cosine similarity, similarity = 1 - distance
                similarity_score = 1.0 - abs(float(score)) if abs(float(score)) <= 1.0 else 0.0
                
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": similarity_score
                })
            
            # Filter by similarity threshold (higher is better)
            filtered_results = [
                r for r in formatted_results
                if r["score"] >= self.config.similarity_threshold
            ]
            
            logger.debug(f"Found {len(filtered_results)} documents above similarity threshold")
            return filtered_results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}", exc_info=True)
            return []
    
    def delete_collection(self):
        """Delete the entire collection (use with caution)"""
        if not self.vector_store:
            return
        
        try:
            if hasattr(self.vector_store, "delete_collection"):
                self.vector_store.delete_collection()
                logger.warning("Vector store collection deleted")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if not self.vector_store:
            return {"status": "not_initialized"}
        
        try:
            # Get collection info if available
            stats = {
                "vector_store_type": self.config.vector_store_type,
                "embedding_provider": self.config.embedding_provider,
                "status": "initialized"
            }
            
            # Try to get document count
            if hasattr(self.vector_store, "_collection"):
                collection = self.vector_store._collection
                if hasattr(collection, "count"):
                    stats["document_count"] = collection.count()
            
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e)}

