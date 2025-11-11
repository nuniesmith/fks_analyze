"""
Document Loaders for RAG

Loads FKS documentation and code files for ingestion into vector store.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
import re

try:
    from langchain_community.document_loaders import (
        TextLoader,
        PyPDFLoader,
        DirectoryLoader,
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available, install with: pip install langchain langchain-community")

from .config import RAGConfig


class FKSDocumentLoader:
    """Load FKS documentation files"""
    
    def __init__(self, config: RAGConfig):
        if not LANGCHAIN_AVAILABLE:
            raise ValueError("LangChain not available. Install with: pip install langchain langchain-community")
        
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self, root_dir: str = "/home/jordan/Documents/code/fks") -> List[Dict[str, Any]]:
        """Load all documentation files from FKS repos"""
        documents = []
        root_path = Path(root_dir)
        
        # Define documentation patterns
        doc_patterns = [
            ("**/*.md", "markdown"),
            ("**/*.txt", "text"),
            ("**/*.rst", "text"),
        ]
        
        # Define documentation directories
        doc_dirs = [
            root_path / "repo" / "main" / "docs",
            root_path / "repo" / "main" / "docs" / "todo",
            root_path / "todo",
        ]
        
        # Also search in service directories
        repo_path = root_path / "repo"
        if repo_path.exists():
            for service_dir in repo_path.iterdir():
                if service_dir.is_dir():
                    # Check for README files
                    readme = service_dir / "README.md"
                    if readme.exists():
                        doc_dirs.append(service_dir)
                    # Check for docs directories
                    docs_dir = service_dir / "docs"
                    if docs_dir.exists():
                        doc_dirs.append(docs_dir)
        
        for doc_dir in doc_dirs:
            if not doc_dir.exists():
                continue
            
            for pattern, file_type in doc_patterns:
                try:
                    files = list(doc_dir.glob(pattern))
                    for file_path in files:
                        try:
                            # Skip hidden files and common exclusions
                            if any(excl in str(file_path) for excl in [".git", "__pycache__", ".pytest_cache"]):
                                continue
                            
                            # Load file
                            loader = TextLoader(str(file_path), encoding="utf-8")
                            docs = loader.load()
                            
                            for doc in docs:
                                # Split into chunks
                                chunks = self.text_splitter.split_text(doc.page_content)
                                
                                for i, chunk in enumerate(chunks):
                                    if chunk.strip():  # Skip empty chunks
                                        documents.append({
                                            "content": chunk,
                                            "metadata": {
                                                **doc.metadata,
                                                "chunk_index": i,
                                                "total_chunks": len(chunks),
                                                "source_file": str(file_path.relative_to(root_path)),
                                                "file_type": file_type,
                                                "repo": self._extract_repo(str(file_path)),
                                                "service": self._extract_service(str(file_path))
                                            }
                                        })
                        except Exception as e:
                            logger.warning(f"Failed to load {file_path}: {e}")
                            continue
                except Exception as e:
                    logger.warning(f"Failed to process pattern {pattern} in {doc_dir}: {e}")
                    continue
        
        logger.info(f"Loaded {len(documents)} document chunks from FKS repos")
        return documents
    
    def _extract_repo(self, file_path: str) -> str:
        """Extract repo name from file path"""
        parts = Path(file_path).parts
        if "repo" in parts:
            repo_index = parts.index("repo")
            if repo_index + 1 < len(parts):
                return parts[repo_index + 1]
        return "unknown"
    
    def _extract_service(self, file_path: str) -> str:
        """Extract service name from file path"""
        repo = self._extract_repo(file_path)
        # Map repo names to service names
        service_map = {
            "main": "fks_main",
            "data": "fks_data",
            "ai": "fks_ai",
            "web": "fks_web",
            "api": "fks_api",
            "app": "fks_app",
            "execution": "fks_execution",
            "portfolio": "fks_portfolio",
            "analyze": "fks_analyze",
            "monitor": "fks_monitor",
            "training": "fks_training",
            "auth": "fks_auth",
        }
        return service_map.get(repo, repo)
    
    def load_code_files(
        self,
        root_dir: str = "/home/jordan/Documents/code/fks",
        file_pattern: str = "**/*.py"
    ) -> List[Dict[str, Any]]:
        """Load Python code files for code analysis"""
        documents = []
        root_path = Path(root_dir)
        
        # Search in repo/*/src directories
        repo_path = root_path / "repo"
        if not repo_path.exists():
            logger.warning(f"Repo path {repo_path} does not exist")
            return documents
        
        for service_dir in repo_path.iterdir():
            if not service_dir.is_dir():
                continue
            
            src_dir = service_dir / "src"
            if not src_dir.exists():
                continue
            
            # Find all Python files
            python_files = list(src_dir.rglob("*.py"))
            
            for file_path in python_files:
                try:
                    # Skip test files and common exclusions
                    if any(excl in str(file_path) for excl in [
                        ".git", "__pycache__", ".pytest_cache", 
                        "test_", "_test.py", "tests/"
                    ]):
                        continue
                    
                    # Load file
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    docs = loader.load()
                    
                    for doc in docs:
                        # Split code into functions/classes
                        chunks = self._split_code(doc.page_content)
                        
                        for i, chunk in enumerate(chunks):
                            if chunk.strip():  # Skip empty chunks
                                documents.append({
                                    "content": chunk,
                                    "metadata": {
                                        **doc.metadata,
                                        "chunk_index": i,
                                        "total_chunks": len(chunks),
                                        "source_file": str(file_path.relative_to(root_path)),
                                        "type": "code",
                                        "language": "python",
                                        "repo": self._extract_repo(str(file_path)),
                                        "service": self._extract_service(str(file_path))
                                    }
                                })
                except Exception as e:
                    logger.warning(f"Failed to load code file {file_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(documents)} code chunks from FKS repos")
        return documents
    
    def _split_code(self, code: str) -> List[str]:
        """Split code into functions and classes"""
        # Simple splitting by function/class definitions
        chunks = []
        current_chunk = []
        lines = code.split("\n")
        
        for line in lines:
            # Check for function or class definition
            if re.match(r'^\s*(def |class |@)', line):
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        # If no functions/classes found, return the whole code as one chunk
        return chunks if chunks else [code]
    
    def load_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load a single file"""
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        try:
            loader = TextLoader(str(file_path_obj), encoding="utf-8")
            docs = loader.load()
            
            documents = []
            for doc in docs:
                chunks = self.text_splitter.split_text(doc.page_content)
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        documents.append({
                            "content": chunk,
                            "metadata": {
                                **doc.metadata,
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "source_file": str(file_path_obj.name),
                                "repo": self._extract_repo(str(file_path)),
                                "service": self._extract_service(str(file_path))
                            }
                        })
            
            return documents
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            return []

