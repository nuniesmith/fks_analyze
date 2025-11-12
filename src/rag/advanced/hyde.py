"""
HyDE (Hypothetical Document Embeddings) Retriever

HyDE generates a hypothetical document that would answer the query,
then uses that document for retrieval instead of the original query.
This improves retrieval accuracy by finding documents similar to the
hypothetical answer rather than just the query.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from ..config import RAGConfig
from ..vector_store import VectorStoreManager


class HyDERetriever:
    """Hypothetical Document Embeddings (HyDE) retriever"""
    
    def __init__(self, config: RAGConfig, vector_store: VectorStoreManager):
        """
        Initialize HyDE retriever.
        
        Args:
            config: RAG configuration
            vector_store: Vector store manager
        """
        self.config = config
        self.vector_store = vector_store
        self.llm = None  # Will be created on demand
    
    def _get_llm(self):
        """Get LLM for generating hypothetical documents"""
        if self.llm is None:
            # For HyDE, we prefer Ollama for cost efficiency
            # Try to create Ollama LLM directly
            try:
                from langchain_community.llms import Ollama
                self.llm = Ollama(
                    model=self.config.ollama_model,
                    base_url=self.config.ollama_endpoint,
                    temperature=0.7
                )
                logger.debug("Using Ollama LLM for HyDE")
            except Exception as e:
                logger.warning(f"Failed to create Ollama LLM for HyDE: {e}, trying Gemini")
                # Fallback to Gemini if available
                try:
                    if self.config.gemini_api_key:
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        self.llm = ChatGoogleGenerativeAI(
                            model=self.config.gemini_llm_model,
                            google_api_key=self.config.gemini_api_key,
                            temperature=0.7
                        )
                        logger.debug("Using Gemini LLM for HyDE")
                    else:
                        raise ValueError("No LLM available for HyDE. Configure Ollama or Gemini.")
                except Exception as e2:
                    logger.error(f"Failed to create LLM for HyDE: {e2}")
                    raise ValueError("No LLM available for HyDE. Configure Ollama or Gemini.")
        
        return self.llm
    
    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.
        
        Args:
            query: Original query string
        
        Returns:
            Hypothetical document text
        """
        llm = self._get_llm()
        
        prompt = f"""Generate a hypothetical document that would answer this question: {query}

The document should be informative and detailed, written as if it were part of the FKS trading platform documentation.
Focus on technical details, architecture, implementation, and best practices.

Write the document in a clear, professional style similar to technical documentation.
Include relevant code examples, configuration details, and explanations.

Hypothetical document:"""
        
        try:
            response = llm.invoke(prompt)
            
            # Extract text from response
            if hasattr(response, 'content'):
                hyde_doc = response.content.strip()
            elif isinstance(response, str):
                hyde_doc = response.strip()
            else:
                hyde_doc = str(response).strip()
            
            logger.debug(f"HyDE: Generated hypothetical document (length: {len(hyde_doc)})")
            return hyde_doc
        except Exception as e:
            logger.error(f"HyDE generation error: {e}", exc_info=True)
            # Fallback to original query if generation fails
            logger.warning("HyDE generation failed, falling back to original query")
            return query
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using HyDE.
        
        Args:
            query: Original query string
            k: Number of documents to retrieve
            filter: Metadata filter for retrieval
        
        Returns:
            List of retrieved documents with scores
        """
        try:
            # Generate hypothetical document
            hyde_doc = self.generate_hypothetical_document(query)
            
            logger.info(f"HyDE: Using hypothetical document for retrieval (query length: {len(query)}, hyde length: {len(hyde_doc)})")
            
            # Use hypothetical document for retrieval
            results = self.vector_store.similarity_search(
                query=hyde_doc,
                k=k or self.config.top_k,
                filter=filter
            )
            
            # Add HyDE metadata to results
            for result in results:
                result["hyde_used"] = True
                result["original_query"] = query
                result["hyde_document_length"] = len(hyde_doc)
            
            logger.info(f"HyDE: Retrieved {len(results)} documents using hypothetical document")
            return results
        except Exception as e:
            logger.error(f"HyDE retrieval error: {e}", exc_info=True)
            # Fallback to standard retrieval
            logger.warning("HyDE retrieval failed, falling back to standard retrieval")
            return self.vector_store.similarity_search(
                query=query,
                k=k or self.config.top_k,
                filter=filter
            )
    
    def retrieve_hybrid(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        hyde_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using both original query and HyDE, then merge results.
        
        Args:
            query: Original query string
            k: Number of documents to retrieve
            filter: Metadata filter for retrieval
            hyde_weight: Weight for HyDE results (0.0-1.0)
        
        Returns:
            Merged list of retrieved documents
        """
        try:
            # Get results from both methods
            standard_results = self.vector_store.similarity_search(
                query=query,
                k=k or self.config.top_k,
                filter=filter
            )
            
            hyde_results = self.retrieve(query, k=k, filter=filter)
            
            # Merge results with weighted scores
            merged_results = {}
            
            # Add standard results
            for result in standard_results:
                doc_id = result.get("metadata", {}).get("source_file", "") + result.get("content", "")[:50]
                if doc_id not in merged_results:
                    merged_results[doc_id] = result.copy()
                    merged_results[doc_id]["score"] = result.get("score", 0.0) * (1.0 - hyde_weight)
                    merged_results[doc_id]["hyde_used"] = False
            
            # Add HyDE results with weighted scores
            for result in hyde_results:
                doc_id = result.get("metadata", {}).get("source_file", "") + result.get("content", "")[:50]
                if doc_id in merged_results:
                    # Merge scores if document already exists
                    merged_results[doc_id]["score"] = (
                        merged_results[doc_id]["score"] + 
                        result.get("score", 0.0) * hyde_weight
                    )
                    merged_results[doc_id]["hyde_used"] = True
                else:
                    merged_results[doc_id] = result.copy()
                    merged_results[doc_id]["score"] = result.get("score", 0.0) * hyde_weight
                    merged_results[doc_id]["hyde_used"] = True
            
            # Sort by score and return top k
            sorted_results = sorted(
                merged_results.values(),
                key=lambda x: x.get("score", 0.0),
                reverse=True
            )
            
            return sorted_results[:k or self.config.top_k]
        except Exception as e:
            logger.error(f"HyDE hybrid retrieval error: {e}", exc_info=True)
            # Fallback to standard retrieval
            return self.vector_store.similarity_search(
                query=query,
                k=k or self.config.top_k,
                filter=filter
            )

