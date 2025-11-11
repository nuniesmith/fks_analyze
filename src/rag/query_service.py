"""
RAG Query Service

Service for querying the RAG system and generating responses.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.llms import Ollama
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available, install with: pip install langchain langchain-community langchain-google-genai")

from .vector_store import VectorStoreManager
from .config import RAGConfig

# Optional advanced RAG imports
try:
    from .advanced.hyde import HyDERetriever
    HYDE_AVAILABLE = True
except ImportError:
    HYDE_AVAILABLE = False
    logger.warning("HyDE not available. Advanced RAG features may be limited.")

try:
    from .advanced.raptor import RAPTORRetriever
    RAPTOR_AVAILABLE = True
except ImportError:
    RAPTOR_AVAILABLE = False
    logger.warning("RAPTOR not available. Advanced RAG features may be limited.")


class RAGQueryService:
    """Service for querying RAG system"""
    
    def __init__(self, config: RAGConfig, vector_store: VectorStoreManager):
        if not LANGCHAIN_AVAILABLE:
            raise ValueError("LangChain not available. Install with: pip install langchain")
        
        self.config = config
        self.vector_store = vector_store
        self.llm = None  # Will be created on demand
        
        # Initialize advanced RAG retrievers if enabled
        self.hyde_retriever = None
        if self.config.use_hyde and HYDE_AVAILABLE:
            try:
                self.hyde_retriever = HyDERetriever(config, vector_store)
                logger.info("HyDE retriever initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize HyDE retriever: {e}")
                self.hyde_retriever = None
        
        self.raptor_retriever = None
        if self.config.use_raptor and RAPTOR_AVAILABLE:
            try:
                self.raptor_retriever = RAPTORRetriever(config, vector_store)
                logger.info("RAPTOR retriever initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize RAPTOR retriever: {e}")
                self.raptor_retriever = None
    
    def _get_llm(self, query: str = ""):
        """Get LLM (Gemini or Ollama) based on query and usage"""
        if self.config.should_use_gemini(query):
            if not self.config.gemini_api_key:
                logger.warning("Gemini API key not set, using Ollama")
                return self._get_ollama_llm()
            
            # Track Gemini usage
            self.config.track_gemini_usage(query)
            
            try:
                logger.debug(f"Using Gemini LLM: {self.config.gemini_llm_model}")
                return ChatGoogleGenerativeAI(
                    model=self.config.gemini_llm_model,
                    google_api_key=self.config.gemini_api_key,
                    temperature=self.config.llm_temperature,
                    max_output_tokens=self.config.max_tokens
                )
            except Exception as e:
                logger.error(f"Failed to create Gemini LLM: {e}, falling back to Ollama")
                return self._get_ollama_llm()
        else:
            return self._get_ollama_llm()
    
    def _get_ollama_llm(self):
        """Get Ollama LLM"""
        try:
            logger.debug(f"Using Ollama LLM: {self.config.ollama_model}")
            from langchain_community.llms import Ollama
            return Ollama(
                model=self.config.ollama_model,
                base_url=self.config.ollama_endpoint,
                temperature=self.config.llm_temperature
            )
        except Exception as e:
            logger.error(f"Failed to create Ollama LLM: {e}")
            raise ValueError("No LLM available. Configure Gemini API key or Ollama endpoint.")
    
    def query(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query the RAG system
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            filter: Metadata filter for retrieval
        
        Returns:
            Dictionary with query results and generated response
        """
        logger.info(f"Processing RAG query: {query[:50]}...")
        
        # Retrieve relevant documents (use advanced RAG if enabled)
        # Priority: RAPTOR > HyDE > Standard
        if self.raptor_retriever and self.config.use_raptor:
            logger.debug("Using RAPTOR for retrieval")
            retrieved_docs = self.raptor_retriever.retrieve(
                query=query,
                k=k or self.config.top_k,
                filter=filter
            )
        elif self.hyde_retriever and self.config.use_hyde:
            logger.debug("Using HyDE for retrieval")
            retrieved_docs = self.hyde_retriever.retrieve(
                query=query,
                k=k or self.config.top_k,
                filter=filter
            )
        else:
            retrieved_docs = self.vector_store.similarity_search(
                query=query,
                k=k or self.config.top_k,
                filter=filter
            )
        
        if not retrieved_docs:
            logger.warning("No relevant documents found")
            return {
                "query": query,
                "answer": "No relevant documents found in the knowledge base.",
                "sources": [],
                "retrieved_count": 0
            }
        
        # Get LLM
        llm = self._get_llm(query)
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant for the FKS trading platform.
Use the following context from the FKS documentation to answer the question.
If you don't know the answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Combine retrieved documents as context
        context = "\n\n".join([
            f"Source: {doc['metadata'].get('source_file', 'unknown')}\n{doc['content']}"
            for doc in retrieved_docs
        ])
        
        # Generate response
        try:
            prompt = prompt_template.format(context=context, question=query)
            answer = llm.invoke(prompt)
            
            # Extract answer text if it's a message object
            if hasattr(answer, 'content'):
                answer_text = answer.content
            else:
                answer_text = str(answer)
            
            logger.info("Generated RAG response successfully")
            
            return {
                "query": query,
                "answer": answer_text,
                "sources": [
                    {
                        "source": doc["metadata"].get("source_file", "unknown"),
                        "score": doc["score"],
                        "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
                    }
                    for doc in retrieved_docs
                ],
                "retrieved_count": len(retrieved_docs)
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return {
                "query": query,
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "retrieved_count": len(retrieved_docs),
                "error": str(e)
            }
    
    def suggest_optimizations(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Suggest optimizations based on query
        
        Args:
            query: Query about optimizations
            context: Optional additional context
        
        Returns:
            Dictionary with optimization suggestions
        """
        logger.info("Generating optimization suggestions...")
        
        # Retrieve relevant documents (use advanced RAG if enabled)
        # Priority: RAPTOR > HyDE > Standard
        if self.raptor_retriever and self.config.use_raptor:
            logger.debug("Using RAPTOR for optimization suggestions")
            retrieved_docs = self.raptor_retriever.retrieve(
                query=query,
                k=self.config.top_k
            )
        elif self.hyde_retriever and self.config.use_hyde:
            logger.debug("Using HyDE for optimization suggestions")
            retrieved_docs = self.hyde_retriever.retrieve(
                query=query,
                k=self.config.top_k
            )
        else:
            retrieved_docs = self.vector_store.similarity_search(
                query=query,
                k=self.config.top_k
            )
        
        # Get LLM
        llm = self._get_llm(query)
        
        # Create optimization prompt
        optimization_prompt = f"""Based on the following context from FKS documentation, suggest optimizations for: {query}

Context:
{context or ''}

Relevant Documentation:
{chr(10).join([doc['content'][:500] for doc in retrieved_docs[:3]])}

Please provide:
1. Specific optimization recommendations
2. Code examples if applicable
3. Potential risks or considerations

Suggestions:"""
        
        try:
            # Try invoke first (LangChain v0.1+), fall back to __call__ (older versions)
            try:
                response = llm.invoke(optimization_prompt)
            except AttributeError:
                # Older LangChain versions use __call__
                response = llm(optimization_prompt)
            
            # Extract response text
            if hasattr(response, 'content'):
                suggestions = response.content
            elif isinstance(response, str):
                suggestions = response
            else:
                suggestions = str(response)
            
            return {
                "query": query,
                "suggestions": suggestions,
                "sources": [
                    doc["metadata"].get("source_file", "unknown")
                    for doc in retrieved_docs
                ]
            }
        except Exception as e:
            logger.error(f"Error generating optimizations: {e}", exc_info=True)
            return {
                "query": query,
                "suggestions": f"Error generating suggestions: {str(e)}",
                "sources": [],
                "error": str(e)
            }

