"""
RAGAS (Retrieval-Augmented Generation Assessment) Evaluation

RAGAS provides metrics for evaluating RAG system quality:
- Faithfulness: Answer is grounded in context
- Answer Correctness: Answer is correct
- Context Precision: Retrieved context is relevant
- Context Recall: Retrieved context covers all needed information
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from ..config import RAGConfig
from ..query_service import RAGQueryService


class RAGASEvaluator:
    """RAGAS evaluation for RAG system"""
    
    def __init__(self, config: RAGConfig, rag_service: RAGQueryService):
        """
        Initialize RAGAS evaluator.
        
        Args:
            config: RAG configuration
            rag_service: RAG query service
        """
        self.config = config
        self.rag_service = rag_service
        self.ragas_available = False
        
        # Try to import RAGAS
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_correctness,
                context_precision,
                context_recall
            )
            self.ragas_available = True
            self.evaluate = evaluate
            self.metrics = {
                "faithfulness": faithfulness,
                "answer_correctness": answer_correctness,
                "context_precision": context_precision,
                "context_recall": context_recall
            }
            logger.info("RAGAS evaluation framework available")
        except ImportError:
            logger.warning("RAGAS not available. Install with: pip install ragas")
            self.ragas_available = False
    
    def evaluate_rag(
        self,
        queries: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system using RAGAS metrics.
        
        Args:
            queries: List of queries to evaluate
            ground_truths: Optional ground truth answers
        
        Returns:
            Dictionary with evaluation results
        """
        if not self.ragas_available:
            logger.warning("RAGAS not available, using fallback evaluation")
            return self._fallback_evaluation(queries, ground_truths)
        
        logger.info(f"Evaluating RAG system with {len(queries)} queries")
        
        # Prepare evaluation dataset
        dataset = []
        for i, query in enumerate(queries):
            # Get RAG response
            rag_result = self.rag_service.query(query)
            
            # Extract context from retrieved documents
            contexts = [doc.get("content", "") for doc in rag_result.get("sources", [])]
            
            dataset_item = {
                "question": query,
                "answer": rag_result.get("answer", ""),
                "contexts": contexts,
                "ground_truth": ground_truths[i] if ground_truths and i < len(ground_truths) else ""
            }
            dataset.append(dataset_item)
        
        try:
            # Evaluate using RAGAS
            from datasets import Dataset
            
            eval_dataset = Dataset.from_list(dataset)
            
            # Select metrics based on availability of ground truth
            if ground_truths:
                metrics_to_use = [
                    self.metrics["faithfulness"],
                    self.metrics["answer_correctness"],
                    self.metrics["context_precision"],
                    self.metrics["context_recall"]
                ]
            else:
                # Without ground truth, we can only evaluate faithfulness and context precision
                metrics_to_use = [
                    self.metrics["faithfulness"],
                    self.metrics["context_precision"]
                ]
            
            scores = self.evaluate(
                dataset=eval_dataset,
                metrics=metrics_to_use
            )
            
            logger.info(f"RAGAS evaluation complete: {scores}")
            
            # Extract scores
            result = {
                "faithfulness": scores.get("faithfulness", 0.0),
                "context_precision": scores.get("context_precision", 0.0),
                "overall_score": scores.get("faithfulness", 0.0)  # Default to faithfulness
            }
            
            if ground_truths:
                result["answer_correctness"] = scores.get("answer_correctness", 0.0)
                result["context_recall"] = scores.get("context_recall", 0.0)
                # Overall score is average of all metrics
                result["overall_score"] = (
                    result["faithfulness"] +
                    result["answer_correctness"] +
                    result["context_precision"] +
                    result["context_recall"]
                ) / 4.0
            
            result["meets_threshold"] = result["overall_score"] >= self.config.ragas_threshold
            result["threshold"] = self.config.ragas_threshold
            result["queries_evaluated"] = len(queries)
            
            return result
        except Exception as e:
            logger.error(f"RAGAS evaluation error: {e}", exc_info=True)
            return self._fallback_evaluation(queries, ground_truths)
    
    def _fallback_evaluation(
        self,
        queries: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fallback evaluation when RAGAS is not available.
        
        Uses simple heuristics to evaluate RAG quality.
        
        Args:
            queries: List of queries
            ground_truths: Optional ground truth answers
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Using fallback evaluation (RAGAS not available)")
        
        total_faithfulness = 0.0
        total_precision = 0.0
        total_correctness = 0.0
        
        for i, query in enumerate(queries):
            # Get RAG response
            rag_result = self.rag_service.query(query)
            answer = rag_result.get("answer", "")
            sources = rag_result.get("sources", [])
            
            # Simple faithfulness check: answer mentions sources
            if sources:
                source_mentions = sum(1 for source in sources if source.get("source", "") in answer)
                faithfulness_score = min(1.0, source_mentions / len(sources)) if sources else 0.0
            else:
                faithfulness_score = 0.0
            
            # Simple precision check: sources have reasonable scores
            if sources:
                avg_score = sum(s.get("score", 0.0) for s in sources) / len(sources)
                precision_score = min(1.0, avg_score / 0.7)  # Normalize to 0.7 threshold
            else:
                precision_score = 0.0
            
            # Simple correctness check: compare with ground truth if available
            if ground_truths and i < len(ground_truths):
                ground_truth = ground_truths[i].lower()
                answer_lower = answer.lower()
                # Simple word overlap
                ground_truth_words = set(ground_truth.split())
                answer_words = set(answer_lower.split())
                if ground_truth_words:
                    overlap = len(ground_truth_words.intersection(answer_words)) / len(ground_truth_words)
                    correctness_score = overlap
                else:
                    correctness_score = 0.0
            else:
                correctness_score = 0.5  # Default when no ground truth
            
            total_faithfulness += faithfulness_score
            total_precision += precision_score
            total_correctness += correctness_score
        
        num_queries = len(queries)
        result = {
            "faithfulness": total_faithfulness / num_queries,
            "context_precision": total_precision / num_queries,
            "overall_score": (total_faithfulness + total_precision) / (2.0 * num_queries)
        }
        
        if ground_truths:
            result["answer_correctness"] = total_correctness / num_queries
            result["overall_score"] = (
                result["faithfulness"] +
                result["answer_correctness"] +
                result["context_precision"]
            ) / 3.0
        
        result["meets_threshold"] = result["overall_score"] >= self.config.ragas_threshold
        result["threshold"] = self.config.ragas_threshold
        result["queries_evaluated"] = num_queries
        result["evaluation_method"] = "fallback"
        
        return result
    
    def evaluate_single_query(
        self,
        query: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single query.
        
        Args:
            query: Query string
            ground_truth: Optional ground truth answer
        
        Returns:
            Dictionary with evaluation results
        """
        return self.evaluate_rag([query], [ground_truth] if ground_truth else None)
    
    def check_quality_threshold(self, scores: Dict[str, Any]) -> bool:
        """
        Check if scores meet quality threshold.
        
        Args:
            scores: Evaluation scores
        
        Returns:
            True if scores meet threshold
        """
        overall_score = scores.get("overall_score", 0.0)
        threshold = scores.get("threshold", self.config.ragas_threshold)
        
        return overall_score >= threshold

