"""
RAG Evaluation Module

Provides evaluation frameworks for RAG system quality.
"""

try:
    from .ragas_eval import RAGASEvaluator
    __all__ = ["RAGASEvaluator"]
except ImportError:
    __all__ = []

