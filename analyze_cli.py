#!/usr/bin/env python3
"""
FKS Analyze CLI Tool - Send documents to RAG system for analysis.

Usage:
    ./analyze document <file_path> [--type <analysis_type>] [--scope <scope>]
    ./analyze directory <dir_path> [--type <analysis_type>] [--scope <scope>]
    ./analyze query <query_text> [--type <analysis_type>] [--scope <scope>]
    ./analyze status <job_id>
    ./analyze results <job_id>
    ./analyze list

Examples:
    # Analyze a single document
    ./analyze document todo/01-core-architecture.md --type project_management

    # Analyze entire todo directory
    ./analyze directory todo/ --type documentation

    # Query the RAG system
    ./analyze query "What are the current priorities for the FKS project?"

    # Check job status
    ./analyze status abc123-def456-ghi789

    # Get results
    ./analyze results abc123-def456-ghi789
"""

import sys
import os
import json
import argparse
import requests
import uuid
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Default service URL
DEFAULT_SERVICE_URL = os.getenv("FKS_ANALYZE_URL", "http://localhost:8008")


def send_document_to_rag(
    file_path: str,
    analysis_type: str = "project_management",
    scope: str = "all",
    service_url: str = DEFAULT_SERVICE_URL
) -> str:
    """Send a single document to the RAG system for analysis."""
    job_id = str(uuid.uuid4())
    
    # Read document content
    doc_path = Path(file_path)
    if not doc_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    content = doc_path.read_text()
    
    # Prepare request
    request_data = {
        "job_id": job_id,
        "query": f"Analyze document: {doc_path.name}",
        "analysis_type": analysis_type,
        "scope": scope,
        "document_content": content,
        "document_path": str(doc_path.relative_to(Path.cwd())),
        "document_name": doc_path.name
    }
    
    # Send to RAG endpoint (if exists) or use analysis endpoint
    try:
        # Try RAG endpoint first
        response = requests.post(
            f"{service_url}/api/v1/rag/analyze",
            json=request_data,
            timeout=30
        )
        if response.status_code == 404:
            # Fallback to analysis endpoint
            response = requests.post(
                f"{service_url}/api/v1/analysis/run",
                json={
                    "repository_path": str(doc_path.parent),
                    "include_mermaid": False,
                    "include_lint": False
                },
                timeout=30
            )
        
        response.raise_for_status()
        result = response.json()
        return result.get("job_id", job_id)
    except requests.exceptions.RequestException as e:
        print(f"Error sending document to analyze service: {e}")
        print(f"Make sure the service is running at {service_url}")
        sys.exit(1)


def send_directory_to_rag(
    dir_path: str,
    analysis_type: str = "project_management",
    scope: str = "all",
    service_url: str = DEFAULT_SERVICE_URL
) -> List[str]:
    """Send all documents in a directory to the RAG system."""
    dir_path_obj = Path(dir_path)
    if not dir_path_obj.exists() or not dir_path_obj.is_dir():
        print(f"Error: Directory not found: {dir_path}")
        sys.exit(1)
    
    # Find all markdown and text files
    doc_extensions = {".md", ".txt", ".rst", ".py", ".yaml", ".yml", ".json"}
    job_ids = []
    
    for file_path in dir_path_obj.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in doc_extensions:
            # Skip hidden files and common exclusions
            if any(part.startswith(".") for part in file_path.parts):
                continue
            
            print(f"Processing: {file_path.relative_to(Path.cwd())}")
            job_id = send_document_to_rag(
                str(file_path),
                analysis_type=analysis_type,
                scope=scope,
                service_url=service_url
            )
            job_ids.append(job_id)
    
    return job_ids


def query_rag_system(
    query: str,
    analysis_type: str = "project_management",
    scope: str = "all",
    service_url: str = DEFAULT_SERVICE_URL
) -> str:
    """Query the RAG system with a natural language question."""
    job_id = str(uuid.uuid4())
    
    request_data = {
        "job_id": job_id,
        "query": query,
        "analysis_type": analysis_type,
        "scope": scope
    }
    
    try:
        # Try RAG query endpoint
        response = requests.post(
            f"{service_url}/api/v1/rag/query",
            json=request_data,
            timeout=30
        )
        if response.status_code == 404:
            # Fallback: use analysis endpoint with query
            response = requests.post(
                f"{service_url}/api/v1/analysis/run",
                json={
                    "repository_path": ".",
                    "query": query
                },
                timeout=30
            )
        
        response.raise_for_status()
        result = response.json()
        return result.get("job_id", job_id)
    except requests.exceptions.RequestException as e:
        print(f"Error querying RAG system: {e}")
        print(f"Make sure the service is running at {service_url}")
        sys.exit(1)


def get_job_status(job_id: str, service_url: str = DEFAULT_SERVICE_URL) -> dict:
    """Get the status of an analysis job."""
    try:
        response = requests.get(
            f"{service_url}/api/v1/analysis/status/{job_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting job status: {e}")
        sys.exit(1)


def get_job_results(job_id: str, service_url: str = DEFAULT_SERVICE_URL) -> dict:
    """Get the results of an analysis job."""
    try:
        response = requests.get(
            f"{service_url}/api/v1/analysis/results/{job_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting job results: {e}")
        sys.exit(1)


def list_jobs(service_url: str = DEFAULT_SERVICE_URL) -> List[dict]:
    """List all analysis jobs."""
    try:
        response = requests.get(
            f"{service_url}/api/v1/analysis/jobs",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error listing jobs: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="FKS Analyze CLI - Send documents to RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Document command
    doc_parser = subparsers.add_parser("document", help="Analyze a single document")
    doc_parser.add_argument("file_path", help="Path to document file")
    doc_parser.add_argument("--type", default="project_management",
                          choices=["project_management", "standardization", "documentation"],
                          help="Type of analysis")
    doc_parser.add_argument("--scope", default="all", help="Scope of analysis")
    doc_parser.add_argument("--url", default=DEFAULT_SERVICE_URL, help="Service URL")
    
    # Directory command
    dir_parser = subparsers.add_parser("directory", help="Analyze all documents in a directory")
    dir_parser.add_argument("dir_path", help="Path to directory")
    dir_parser.add_argument("--type", default="project_management",
                          choices=["project_management", "standardization", "documentation"],
                          help="Type of analysis")
    dir_parser.add_argument("--scope", default="all", help="Scope of analysis")
    dir_parser.add_argument("--url", default=DEFAULT_SERVICE_URL, help="Service URL")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query_text", help="Query text")
    query_parser.add_argument("--type", default="project_management",
                            choices=["project_management", "standardization", "documentation"],
                            help="Type of analysis")
    query_parser.add_argument("--scope", default="all", help="Scope of analysis")
    query_parser.add_argument("--url", default=DEFAULT_SERVICE_URL, help="Service URL")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get job status")
    status_parser.add_argument("job_id", help="Job ID")
    status_parser.add_argument("--url", default=DEFAULT_SERVICE_URL, help="Service URL")
    
    # Results command
    results_parser = subparsers.add_parser("results", help="Get job results")
    results_parser.add_argument("job_id", help="Job ID")
    results_parser.add_argument("--url", default=DEFAULT_SERVICE_URL, help="Service URL")
    results_parser.add_argument("--format", choices=["json", "text"], default="text",
                               help="Output format")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all jobs")
    list_parser.add_argument("--url", default=DEFAULT_SERVICE_URL, help="Service URL")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "document":
            job_id = send_document_to_rag(
                args.file_path,
                analysis_type=args.type,
                scope=args.scope,
                service_url=args.url
            )
            print(f"✅ Document sent for analysis. Job ID: {job_id}")
            print(f"   Check status: ./analyze status {job_id}")
            print(f"   Get results: ./analyze results {job_id}")
        
        elif args.command == "directory":
            job_ids = send_directory_to_rag(
                args.dir_path,
                analysis_type=args.type,
                scope=args.scope,
                service_url=args.url
            )
            print(f"\n✅ Processed {len(job_ids)} documents")
            print(f"   Job IDs: {', '.join(job_ids[:5])}")
            if len(job_ids) > 5:
                print(f"   ... and {len(job_ids) - 5} more")
        
        elif args.command == "query":
            job_id = query_rag_system(
                args.query_text,
                analysis_type=args.type,
                scope=args.scope,
                service_url=args.url
            )
            print(f"✅ Query submitted. Job ID: {job_id}")
            print(f"   Check status: ./analyze status {job_id}")
            print(f"   Get results: ./analyze results {job_id}")
        
        elif args.command == "status":
            status = get_job_status(args.job_id, args.url)
            print(f"Job ID: {status.get('job_id', args.job_id)}")
            print(f"Status: {status.get('status', 'unknown')}")
            if 'progress' in status:
                print(f"Progress: {status['progress'] * 100:.1f}%")
            if 'error' in status:
                print(f"Error: {status['error']}")
        
        elif args.command == "results":
            results = get_job_results(args.job_id, args.url)
            if args.format == "json":
                print(json.dumps(results, indent=2))
            else:
                print(f"\n📊 Analysis Results for Job: {args.job_id}")
                print("=" * 60)
                if 'ai_insights' in results:
                    insights = results['ai_insights']
                    if isinstance(insights, dict) and 'text' in insights:
                        print(insights['text'])
                    elif isinstance(insights, str):
                        print(insights)
                    else:
                        print(json.dumps(insights, indent=2))
                elif 'result' in results:
                    print(json.dumps(results['result'], indent=2))
                else:
                    print(json.dumps(results, indent=2))
        
        elif args.command == "list":
            jobs = list_jobs(args.url)
            print(f"\n📋 Analysis Jobs ({len(jobs)} total)")
            print("=" * 60)
            for job in jobs[:10]:  # Show first 10
                print(f"  {job.get('job_id', 'unknown')[:8]}... | {job.get('status', 'unknown'):10} | {job.get('query', 'N/A')[:40]}")
            if len(jobs) > 10:
                print(f"\n  ... and {len(jobs) - 10} more jobs")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

