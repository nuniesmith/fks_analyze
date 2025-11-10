"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

RAPTOR builds a hierarchical tree structure of documents by recursively
summarizing and clustering them. This improves retrieval for complex queries
by organizing documents at multiple levels of abstraction.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
from collections import defaultdict

from ..config import RAGConfig
from ..vector_store import VectorStoreManager


class RAPTORRetriever:
    """RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)"""
    
    def __init__(self, config: RAGConfig, vector_store: VectorStoreManager):
        """
        Initialize RAPTOR retriever.
        
        Args:
            config: RAG configuration
            vector_store: Vector store manager
        """
        self.config = config
        self.vector_store = vector_store
        self.llm = None  # Will be created on demand
        self.tree_structure = {}  # Store tree structure
    
    def _get_llm(self):
        """Get LLM for summarization"""
        if self.llm is None:
            # For RAPTOR, we prefer Ollama for cost efficiency
            try:
                from langchain_community.llms import Ollama
                self.llm = Ollama(
                    model=self.config.ollama_model,
                    base_url=self.config.ollama_endpoint,
                    temperature=0.3  # Lower temperature for summarization
                )
                logger.debug("Using Ollama LLM for RAPTOR")
            except Exception as e:
                logger.warning(f"Failed to create Ollama LLM for RAPTOR: {e}, trying Gemini")
                # Fallback to Gemini if available
                try:
                    if self.config.gemini_api_key:
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        self.llm = ChatGoogleGenerativeAI(
                            model=self.config.gemini_llm_model,
                            google_api_key=self.config.gemini_api_key,
                            temperature=0.3
                        )
                        logger.debug("Using Gemini LLM for RAPTOR")
                    else:
                        raise ValueError("No LLM available for RAPTOR. Configure Ollama or Gemini.")
                except Exception as e2:
                    logger.error(f"Failed to create LLM for RAPTOR: {e2}")
                    raise ValueError("No LLM available for RAPTOR. Configure Ollama or Gemini.")
        
        return self.llm
    
    def summarize_document(self, content: str, max_length: int = 200) -> str:
        """
        Summarize a document using LLM.
        
        Args:
            content: Document content to summarize
            max_length: Maximum length of summary
        
        Returns:
            Summarized text
        """
        llm = self._get_llm()
        
        prompt = f"""Summarize the following FKS documentation in {max_length} words or less.
Focus on key concepts, architecture, and implementation details.

Document:
{content[:2000]}  # Limit input to avoid token limits

Summary:"""
        
        try:
            response = llm.invoke(prompt)
            
            # Extract text from response
            if hasattr(response, 'content'):
                summary = response.content.strip()
            elif isinstance(response, str):
                summary = response.strip()
            else:
                summary = str(response).strip()
            
            # Truncate if too long
            if len(summary) > max_length * 2:
                summary = summary[:max_length * 2] + "..."
            
            return summary
        except Exception as e:
            logger.error(f"RAPTOR summarization error: {e}", exc_info=True)
            # Fallback to simple truncation
            return content[:max_length] + "..." if len(content) > max_length else content
    
    def build_tree(
        self,
        documents: List[Dict[str, Any]],
        max_depth: int = 3,
        cluster_size: int = 5
    ) -> Dict[str, Any]:
        """
        Build hierarchical tree structure from documents.
        
        Args:
            documents: List of documents with content and metadata
            max_depth: Maximum depth of tree
            cluster_size: Number of documents per cluster
        
        Returns:
            Tree structure dictionary
        """
        logger.info(f"Building RAPTOR tree from {len(documents)} documents (max_depth={max_depth})")
        
        tree = {
            "root": {
                "documents": documents,
                "depth": 0,
                "summaries": [],
                "children": []
            }
        }
        
        current_level = [tree["root"]]
        
        for depth in range(1, max_depth + 1):
            next_level = []
            
            for node in current_level:
                if len(node["documents"]) <= cluster_size:
                    # Too few documents, don't cluster further
                    continue
                
                # Cluster documents (simple approach: group by similarity)
                clusters = self._cluster_documents(node["documents"], cluster_size)
                
                # Create child nodes
                for cluster in clusters:
                    # Summarize cluster
                    cluster_content = "\n\n".join([doc.get("content", "")[:500] for doc in cluster])
                    summary = self.summarize_document(cluster_content, max_length=150)
                    
                    child_node = {
                        "documents": cluster,
                        "depth": depth,
                        "summary": summary,
                        "parent": node,
                        "children": []
                    }
                    
                    node["children"].append(child_node)
                    next_level.append(child_node)
            
            if not next_level:
                # No more clustering possible
                break
            
            current_level = next_level
        
        self.tree_structure = tree
        logger.info(f"RAPTOR tree built with depth {max_depth}")
        return tree
    
    def _cluster_documents(
        self,
        documents: List[Dict[str, Any]],
        cluster_size: int
    ) -> List[List[Dict[str, Any]]]:
        """
        Cluster documents by similarity (simple approach).
        
        Args:
            documents: List of documents
            cluster_size: Target cluster size
        
        Returns:
            List of document clusters
        """
        # Simple clustering: group documents by service/repo
        clusters = defaultdict(list)
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            service = metadata.get("service", "unknown")
            clusters[service].append(doc)
        
        # If clustering by service doesn't create enough clusters, use round-robin
        if len(clusters) < len(documents) // cluster_size:
            clusters = defaultdict(list)
            for i, doc in enumerate(documents):
                cluster_id = i // cluster_size
                clusters[cluster_id].append(doc)
        
        return list(clusters.values())
    
    def retrieve_from_tree(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents from RAPTOR tree structure.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            filter: Metadata filter
        
        Returns:
            List of retrieved documents
        """
        if not self.tree_structure:
            logger.warning("RAPTOR tree not built, falling back to standard retrieval")
            return self.vector_store.similarity_search(
                query=query,
                k=k or self.config.top_k,
                filter=filter
            )
        
        k = k or self.config.top_k
        
        # Search at each level of the tree
        all_results = []
        
        # Search root level
        root_docs = self.tree_structure["root"]["documents"]
        root_results = self._search_documents(query, root_docs, k)
        all_results.extend(root_results)
        
        # Search child levels
        def search_children(node, query, results, max_results):
            if len(results) >= max_results:
                return
            
            for child in node.get("children", []):
                # Search using summary
                if child.get("summary"):
                    summary_score = self._similarity_score(query, child["summary"])
                    if summary_score > 0.5:  # Threshold for relevance
                        child_results = self._search_documents(
                            query,
                            child["documents"],
                            max_results - len(results)
                        )
                        results.extend(child_results)
                        
                        # Recursively search deeper
                        search_children(child, query, results, max_results)
        
        search_children(self.tree_structure["root"], query, all_results, k * 2)
        
        # Deduplicate and rank
        seen = set()
        unique_results = []
        for result in all_results:
            content_hash = hash(result.get("content", "")[:100])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(result)
        
        # Sort by score and return top k
        unique_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        # Add RAPTOR metadata
        for result in unique_results[:k]:
            result["raptor_used"] = True
            result["tree_depth"] = result.get("metadata", {}).get("tree_depth", 0)
        
        logger.info(f"RAPTOR: Retrieved {len(unique_results[:k])} documents from tree")
        return unique_results[:k]
    
    def _search_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        k: int
    ) -> List[Dict[str, Any]]:
        """Search documents using vector store"""
        # Extract texts for search
        texts = [doc.get("content", "") for doc in documents]
        
        if not texts:
            return []
        
        # Use vector store for similarity search
        try:
            # Create temporary documents in vector store or use embeddings directly
            # For simplicity, use the vector store's similarity search
            results = self.vector_store.similarity_search(
                query=query,
                k=min(k, len(texts))
            )
            
            # Match results back to original documents
            matched_results = []
            for result in results:
                result_content = result.get("content", "")
                # Find matching document
                for doc in documents:
                    if result_content[:100] in doc.get("content", ""):
                        matched_results.append({
                            **result,
                            "metadata": {**result.get("metadata", {}), **doc.get("metadata", {})}
                        })
                        break
            
            return matched_results[:k]
        except Exception as e:
            logger.error(f"RAPTOR document search error: {e}", exc_info=True)
            return []
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Simple similarity score between two texts"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        build_tree_if_needed: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using RAPTOR.
        
        If tree is not built, falls back to standard retrieval.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            filter: Metadata filter
            build_tree_if_needed: Whether to build tree if not exists
        
        Returns:
            List of retrieved documents
        """
        if not self.tree_structure:
            return self.retrieve_from_tree(query, k=k, filter=filter)
        
        if build_tree_if_needed:
            logger.info("RAPTOR tree not built, building from vector store...")
            # Get all documents from vector store (this is a simplified approach)
            # In production, you'd want to cache the tree
            logger.warning("RAPTOR tree building from vector store not fully implemented")
            # Fall back to standard retrieval
            return self.vector_store.similarity_search(
                query=query,
                k=k or self.config.top_k,
                filter=filter
            )
        else:
            return self.vector_store.similarity_search(
                query=query,
                k=k or self.config.top_k,
                filter=filter
            )
