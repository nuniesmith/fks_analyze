"""
RAG Configuration with Gemini API and Ollama Hybrid Support
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from loguru import logger


class RAGConfig(BaseModel):
    """RAG configuration with Gemini API and Ollama hybrid support"""
    
    # Vector Store
    vector_store_type: str = os.getenv("RAG_VECTOR_STORE", "chroma")  # chroma or pgvector
    chroma_persist_dir: str = os.getenv("RAG_CHROMA_DIR", "./chroma_db")
    pgvector_connection: Optional[str] = os.getenv("RAG_PGVECTOR_CONNECTION")
    
    # Embeddings - Gemini API (Free Tier: 1,500-10,000 grounded prompts/day)
    gemini_api_key: Optional[str] = os.getenv("GOOGLE_AI_API_KEY")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
    gemini_llm_model: str = os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash-exp")
    gemini_free_tier_limit: int = int(os.getenv("GEMINI_FREE_TIER_LIMIT", "1500"))  # Daily limit
    
    # Embeddings - Local (Ollama)
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_provider: str = os.getenv("RAG_EMBEDDING_PROVIDER", "hybrid")  # gemini, ollama, local, or hybrid
    ollama_endpoint: str = os.getenv("OLLAMA_HOST", "http://fks_ai:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5")
    
    # Hybrid Routing
    use_hybrid: bool = os.getenv("RAG_USE_HYBRID", "true").lower() == "true"
    hybrid_threshold: int = int(os.getenv("RAG_HYBRID_THRESHOLD", "500"))  # Use Gemini for queries >500 chars
    gemini_usage_tracker: Dict[str, int] = {}  # Track daily usage
    
    # Chunking
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    
    # Retrieval
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    similarity_threshold: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7"))
    
    # Generation
    llm_temperature: float = float(os.getenv("RAG_LLM_TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("RAG_MAX_TOKENS", "1000"))
    
    # Advanced RAG
    use_hyde: bool = os.getenv("RAG_USE_HYDE", "true").lower() == "true"
    use_raptor: bool = os.getenv("RAG_USE_RAPTOR", "false").lower() == "true"
    use_self_rag: bool = os.getenv("RAG_USE_SELF_RAG", "false").lower() == "true"
    use_reranking: bool = os.getenv("RAG_USE_RERANKING", "false").lower() == "true"
    
    # Evaluation
    evaluation_enabled: bool = os.getenv("RAG_EVALUATION_ENABLED", "true").lower() == "true"
    ragas_threshold: float = float(os.getenv("RAGAS_THRESHOLD", "0.9"))
    
    def should_use_gemini(self, query: str) -> bool:
        """Determine if Gemini should be used based on query complexity and usage limits"""
        if not self.use_hybrid:
            return self.embedding_provider == "gemini"
        
        # Check daily usage limit
        today = datetime.now().strftime("%Y-%m-%d")
        daily_usage = self.gemini_usage_tracker.get(today, 0)
        if daily_usage >= self.gemini_free_tier_limit:
            logger.warning(f"Gemini daily limit reached ({daily_usage}/{self.gemini_free_tier_limit}), using Ollama")
            return False  # Use Ollama if limit reached
        
        # Use Gemini for complex queries
        if len(query) > self.hybrid_threshold:
            return True
        
        # Use Ollama for simple queries
        return False
    
    def track_gemini_usage(self, query: str):
        """Track Gemini API usage for daily limit management"""
        if not self.should_use_gemini(query):
            return
        
        today = datetime.now().strftime("%Y-%m-%d")
        self.gemini_usage_tracker[today] = self.gemini_usage_tracker.get(today, 0) + 1
        logger.debug(f"Gemini usage today: {self.gemini_usage_tracker[today]}/{self.gemini_free_tier_limit}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        today = datetime.now().strftime("%Y-%m-%d")
        usage_today = self.gemini_usage_tracker.get(today, 0)
        return {
            "today": today,
            "gemini_usage_today": usage_today,
            "gemini_free_tier_limit": self.gemini_free_tier_limit,
            "usage_percentage": (usage_today / self.gemini_free_tier_limit) * 100 if self.gemini_free_tier_limit > 0 else 0,
            "embedding_provider": self.embedding_provider,
            "use_hybrid": self.use_hybrid
        }
    
    class Config:
        arbitrary_types_allowed = True

