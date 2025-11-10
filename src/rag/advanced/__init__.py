"""
Advanced RAG techniques for improved retrieval and generation.
"""

from .hyde import HyDERetriever

# Optional RAPTOR import
try:
    from .raptor import RAPTORRetriever
    RAPTOR_AVAILABLE = True
except ImportError:
    RAPTOR_AVAILABLE = False
    RAPTORRetriever = None

# Optional Self-RAG import
try:
    from .self_rag import SelfRAGNode, SelfRAGWorkflow
    SELF_RAG_AVAILABLE = True
except ImportError:
    SELF_RAG_AVAILABLE = False
    SelfRAGNode = None
    SelfRAGWorkflow = None

if RAPTOR_AVAILABLE and SELF_RAG_AVAILABLE:
    __all__ = [
        "HyDERetriever",
        "RAPTORRetriever",
        "SelfRAGNode",
        "SelfRAGWorkflow"
    ]
elif RAPTOR_AVAILABLE:
    __all__ = [
        "HyDERetriever",
        "RAPTORRetriever"
    ]
else:
    __all__ = [
        "HyDERetriever"
    ]

