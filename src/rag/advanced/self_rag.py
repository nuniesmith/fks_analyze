"""
Self-RAG (Self-Retrieval Augmented Generation)

Self-RAG implements self-correction by:
1. Judging whether retrieval is needed
2. Generating answers with retrieval
3. Judging answer quality and faithfulness
4. Refining answers if needed
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from ..config import RAGConfig
from ..vector_store import VectorStoreManager


class SelfRAGNode:
    """Self-RAG node for self-correction workflow"""
    
    def __init__(self, config: RAGConfig, vector_store: VectorStoreManager):
        """
        Initialize Self-RAG node.
        
        Args:
            config: RAG configuration
            vector_store: Vector store manager
        """
        self.config = config
        self.vector_store = vector_store
        self.llm = None  # Will be created on demand
        self.query_service = RAGQueryService(config, vector_store)
    
    def _get_llm(self):
        """Get LLM for generation and judgment"""
        if self.llm is None:
            # For Self-RAG, we prefer Gemini for better judgment quality
            try:
                if self.config.gemini_api_key:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    self.llm = ChatGoogleGenerativeAI(
                        model=self.config.gemini_llm_model,
                        google_api_key=self.config.gemini_api_key,
                        temperature=0.3  # Lower temperature for judgment
                    )
                    logger.debug("Using Gemini LLM for Self-RAG")
                else:
                    raise ValueError("Gemini API key not available")
            except Exception as e:
                logger.warning(f"Failed to create Gemini LLM for Self-RAG: {e}, trying Ollama")
                # Fallback to Ollama
                try:
                    from langchain_community.llms import Ollama
                    self.llm = Ollama(
                        model=self.config.ollama_model,
                        base_url=self.config.ollama_endpoint,
                        temperature=0.3
                    )
                    logger.debug("Using Ollama LLM for Self-RAG")
                except Exception as e2:
                    logger.error(f"Failed to create LLM for Self-RAG: {e2}")
                    raise ValueError("No LLM available for Self-RAG. Configure Ollama or Gemini.")
        
        return self.llm
    
    def judge_retrieval_need(self, query: str) -> Dict[str, Any]:
        """
        Judge whether retrieval is needed for the query.
        
        Args:
            query: Query string
        
        Returns:
            Dictionary with judgment result
        """
        llm = self._get_llm()
        
        prompt = f"""Given the following query, determine if retrieval from documentation is needed to answer it accurately.

Query: {query}

Consider:
- Does the query require specific knowledge from documentation?
- Can the query be answered with general knowledge?
- Is the query about FKS platform architecture, implementation, or configuration?

Respond with only one word: "Yes" if retrieval is needed, "No" if not needed.

Judgment:"""
        
        try:
            response = llm.invoke(prompt)
            
            # Extract judgment
            if hasattr(response, 'content'):
                judgment_text = response.content.strip().lower()
            elif isinstance(response, str):
                judgment_text = response.strip().lower()
            else:
                judgment_text = str(response).strip().lower()
            
            needs_retrieval = "yes" in judgment_text or "retrieval" in judgment_text
            
            logger.debug(f"Self-RAG retrieval judgment: {needs_retrieval} for query: {query[:50]}")
            
            return {
                "needs_retrieval": needs_retrieval,
                "reason": judgment_text,
                "query": query
            }
        except Exception as e:
            logger.error(f"Self-RAG judgment error: {e}", exc_info=True)
            # Default to needing retrieval on error
            return {
                "needs_retrieval": True,
                "reason": f"Error in judgment: {str(e)}",
                "query": query
            }
    
    def generate_with_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answer using retrieved documents.
        
        Args:
            query: Query string
            retrieved_docs: Retrieved documents
        
        Returns:
            Dictionary with generated answer
        """
        llm = self._get_llm()
        
        # Format retrieved documents
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.get('content', '')[:500]}"
            for i, doc in enumerate(retrieved_docs[:5])  # Limit to 5 docs
        ])
        
        prompt = f"""Answer the following query using the provided documentation context.

Query: {query}

Context from documentation:
{context}

Instructions:
- Use only information from the provided context
- If the context doesn't contain enough information, say so
- Be specific and cite which document(s) you used
- If you're uncertain, indicate that

Answer:"""
        
        try:
            response = llm.invoke(prompt)
            
            # Extract answer
            if hasattr(response, 'content'):
                answer = response.content.strip()
            elif isinstance(response, str):
                answer = response.strip()
            else:
                answer = str(response).strip()
            
            logger.debug(f"Self-RAG generated answer (length: {len(answer)})")
            
            return {
                "answer": answer,
                "query": query,
                "sources": [doc.get("metadata", {}).get("source_file", "unknown") for doc in retrieved_docs],
                "retrieved_count": len(retrieved_docs)
            }
        except Exception as e:
            logger.error(f"Self-RAG generation error: {e}", exc_info=True)
            return {
                "answer": f"Error generating answer: {str(e)}",
                "query": query,
                "sources": [],
                "retrieved_count": 0,
                "error": str(e)
            }
    
    def judge_faithfulness(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Judge whether the answer is faithful to the retrieved documents.
        
        Args:
            query: Original query
            answer: Generated answer
            retrieved_docs: Retrieved documents used
        
        Returns:
            Dictionary with faithfulness judgment
        """
        llm = self._get_llm()
        
        # Format retrieved documents
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.get('content', '')[:500]}"
            for i, doc in enumerate(retrieved_docs[:5])
        ])
        
        prompt = f"""Judge whether the following answer is faithful to the provided documentation context.

Query: {query}

Answer: {answer}

Context from documentation:
{context}

Instructions:
- Check if the answer is supported by the context
- Check if the answer contains any information not in the context
- Check if the answer contradicts the context

Respond with:
- "Faithful" if the answer is fully supported by the context
- "Partially Faithful" if the answer is mostly supported but has some issues
- "Not Faithful" if the answer contradicts or contains unsupported information

Also provide a score from 0.0 to 1.0 where 1.0 is completely faithful.

Judgment:"""
        
        try:
            response = llm.invoke(prompt)
            
            # Extract judgment
            if hasattr(response, 'content'):
                judgment_text = response.content.strip().lower()
            elif isinstance(response, str):
                judgment_text = response.strip().lower()
            else:
                judgment_text = str(response).strip().lower()
            
            # Extract score (look for number between 0.0 and 1.0)
            import re
            score_match = re.search(r'(\d+\.?\d*)', judgment_text)
            if score_match:
                score = float(score_match.group(1))
                if score > 1.0:
                    score = score / 10.0  # Normalize if > 1.0
            else:
                # Estimate score from judgment text
                if "faithful" in judgment_text and "not" not in judgment_text:
                    score = 0.9
                elif "partially" in judgment_text:
                    score = 0.6
                else:
                    score = 0.3
            
            is_faithful = score >= self.config.ragas_threshold
            
            logger.debug(f"Self-RAG faithfulness judgment: {score:.2f} (threshold: {self.config.ragas_threshold})")
            
            return {
                "is_faithful": is_faithful,
                "score": score,
                "judgment": judgment_text,
                "threshold": self.config.ragas_threshold
            }
        except Exception as e:
            logger.error(f"Self-RAG faithfulness judgment error: {e}", exc_info=True)
            # Default to not faithful on error
            return {
                "is_faithful": False,
                "score": 0.0,
                "judgment": f"Error in judgment: {str(e)}",
                "threshold": self.config.ragas_threshold
            }
    
    def refine_answer(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        faithfulness_judgment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Refine answer based on faithfulness judgment.
        
        Args:
            query: Original query
            answer: Current answer
            retrieved_docs: Retrieved documents
            faithfulness_judgment: Faithfulness judgment result
        
        Returns:
            Dictionary with refined answer
        """
        if faithfulness_judgment.get("is_faithful", False):
            # Answer is faithful, no refinement needed
            logger.debug("Self-RAG: Answer is faithful, no refinement needed")
            return {
                "answer": answer,
                "refined": False,
                "reason": "Answer is faithful to context"
            }
        
        llm = self._get_llm()
        
        # Format retrieved documents
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.get('content', '')[:500]}"
            for i, doc in enumerate(retrieved_docs[:5])
        ])
        
        prompt = f"""Refine the following answer to be more faithful to the provided documentation context.

Query: {query}

Current Answer: {answer}

Issues with current answer:
- Faithfulness score: {faithfulness_judgment.get('score', 0.0):.2f}
- Judgment: {faithfulness_judgment.get('judgment', 'Unknown')}

Context from documentation:
{context}

Instructions:
- Rewrite the answer to be fully supported by the context
- Remove any information not in the context
- Correct any contradictions
- Be specific and cite which document(s) you used

Refined Answer:"""
        
        try:
            response = llm.invoke(prompt)
            
            # Extract refined answer
            if hasattr(response, 'content'):
                refined_answer = response.content.strip()
            elif isinstance(response, str):
                refined_answer = response.strip()
            else:
                refined_answer = str(response).strip()
            
            logger.info(f"Self-RAG: Refined answer (original length: {len(answer)}, refined length: {len(refined_answer)})")
            
            return {
                "answer": refined_answer,
                "refined": True,
                "reason": f"Answer refined to improve faithfulness (score: {faithfulness_judgment.get('score', 0.0):.2f} -> target: {self.config.ragas_threshold})",
                "original_answer": answer
            }
        except Exception as e:
            logger.error(f"Self-RAG refinement error: {e}", exc_info=True)
            # Return original answer on error
            return {
                "answer": answer,
                "refined": False,
                "reason": f"Error in refinement: {str(e)}",
                "error": str(e)
            }


class SelfRAGWorkflow:
    """Self-RAG workflow using LangGraph"""
    
    def __init__(self, config: RAGConfig, vector_store: VectorStoreManager):
        """
        Initialize Self-RAG workflow.
        
        Args:
            config: RAG configuration
            vector_store: Vector store manager
        """
        self.config = config
        self.vector_store = vector_store
        self.self_rag_node = SelfRAGNode(config, vector_store)
    
    def run(
        self,
        query: str,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run Self-RAG workflow.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
        
        Returns:
            Dictionary with final answer and workflow results
        """
        logger.info(f"Running Self-RAG workflow for query: {query[:50]}...")
        
        workflow_results = {
            "query": query,
            "steps": []
        }
        
        # Step 1: Judge retrieval need
        retrieval_judgment = self.self_rag_node.judge_retrieval_need(query)
        workflow_results["steps"].append({
            "step": "retrieval_judgment",
            "result": retrieval_judgment
        })
        
        if not retrieval_judgment.get("needs_retrieval", True):
            # No retrieval needed, generate answer without context
            logger.info("Self-RAG: Retrieval not needed, generating answer without context")
            answer_result = self.self_rag_node.generate_with_retrieval(query, [])
            workflow_results["steps"].append({
                "step": "generation",
                "result": answer_result
            })
            workflow_results["final_answer"] = answer_result["answer"]
            workflow_results["retrieval_used"] = False
            return workflow_results
        
        # Step 2: Retrieve documents
        retrieved_docs = self.vector_store.similarity_search(
            query=query,
            k=k or self.config.top_k
        )
        workflow_results["steps"].append({
            "step": "retrieval",
            "result": {
                "retrieved_count": len(retrieved_docs),
                "sources": [doc.get("metadata", {}).get("source_file", "unknown") for doc in retrieved_docs]
            }
        })
        
        if not retrieved_docs:
            logger.warning("Self-RAG: No documents retrieved")
            workflow_results["final_answer"] = "I couldn't find relevant documentation to answer your query."
            workflow_results["retrieval_used"] = True
            return workflow_results
        
        # Step 3: Generate answer with retrieval
        answer_result = self.self_rag_node.generate_with_retrieval(query, retrieved_docs)
        workflow_results["steps"].append({
            "step": "generation",
            "result": answer_result
        })
        
        # Step 4: Judge faithfulness
        faithfulness_judgment = self.self_rag_node.judge_faithfulness(
            query,
            answer_result["answer"],
            retrieved_docs
        )
        workflow_results["steps"].append({
            "step": "faithfulness_judgment",
            "result": faithfulness_judgment
        })
        
        # Step 5: Refine if needed
        if not faithfulness_judgment.get("is_faithful", False):
            logger.info("Self-RAG: Answer not faithful, refining...")
            refinement_result = self.self_rag_node.refine_answer(
                query,
                answer_result["answer"],
                retrieved_docs,
                faithfulness_judgment
            )
            workflow_results["steps"].append({
                "step": "refinement",
                "result": refinement_result
            })
            workflow_results["final_answer"] = refinement_result["answer"]
        else:
            workflow_results["final_answer"] = answer_result["answer"]
        
        workflow_results["retrieval_used"] = True
        workflow_results["faithfulness_score"] = faithfulness_judgment.get("score", 0.0)
        workflow_results["is_faithful"] = faithfulness_judgment.get("is_faithful", False)
        
        logger.info(f"Self-RAG workflow complete (faithfulness: {workflow_results['faithfulness_score']:.2f})")
        
        return workflow_results

