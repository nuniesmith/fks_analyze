# FKS Analyze

Repository analysis and code quality service for FKS Trading Platform.

**Port**: 8008  
**Framework**: Python 3.12 + FastAPI  
**Role**: Repository analysis, code quality, RAG system for project management

## 🎯 Purpose

FKS Analyze provides repository analysis and code quality services for the FKS Trading Platform. It offers:

- **Repository Analysis**: File structure, code metrics, and quality checks
- **RAG System**: Google AI API integration for intelligent project management
- **Documentation Generation**: MkDocs and Mermaid diagram generation
- **Code Standardization**: Automated code consistency checks
- **AI-Powered Insights**: Context-aware responses for project management

## 🏗️ Architecture

```
┌─────────────┐
│  fks_web     │
│  (Django)    │
└──────┬───────┘
       │
       │ HTTP API
       ▼
┌─────────────┐     ┌─────────────┐
│ fks_analyze │────▶│ Google AI   │
│  (FastAPI)  │     │  (Gemini)   │
└──────┬───────┘     └─────────────┘
       │
       │ File System Access
       ▼
┌─────────────┐
│ FKS Repos   │
│ (Analysis)  │
└─────────────┘
```

## 🚀 Quick Start

### Development

```bash
# Install dependencies
pip install -r requirements.dev.txt

# Run service
uvicorn src.main:app --reload --host 0.0.0.0 --port 8008
```

### Docker

```bash
# Build image
docker build -t nuniesmith/fks:analyze-latest .

# Run container
docker run -p 8008:8008 \
  -e GOOGLE_AI_API_KEY=your-api-key \
  nuniesmith/fks:analyze-latest
```

### Kubernetes

```bash
# Deploy using Helm
cd repo/main/k8s/charts/fks-platform
helm install fks-platform . -n fks-trading

# Or using the unified start script
cd /home/jordan/Documents/code/fks
./start.sh --type k8s
```

## 📡 API Endpoints

### Health Checks

- `GET /health` - Health check
- `GET /health/ready` - Readiness check

### Analysis

- `POST /api/v1/analysis/run` - Run repository analysis
- `GET /api/v1/analysis/status/{job_id}` - Get job status
- `GET /api/v1/analysis/results/{job_id}` - Get job results
- `GET /api/v1/analysis/jobs` - List all jobs
- `DELETE /api/v1/analysis/jobs/{job_id}` - Delete job

## 🔄 RAG System Integration

### Overview

`fks_analyze` integrates Google AI API (free tier) as the LLM component of a RAG system. It retrieves relevant data from FKS repositories and applications to provide context-aware responses for project management, standardization of codebases, and generation of documentation (MkDocs) and visualizations (Mermaid diagrams).

### Setup

#### Google AI API Integration

- **API Key**: You'll need to obtain an API key for Google AI API (Vertex AI or Gemini API). The free tier has usage limits, so ensure you understand the constraints. Store the key securely as an environment variable `GOOGLE_AI_API_KEY`.
- **Project Configuration**: Set additional environment variables for Google Cloud project:
  - `GOOGLE_CLOUD_PROJECT`: Your project ID.
  - `GOOGLE_CLOUD_LOCATION`: The location for API calls (default is `us-central1`).
  - `GOOGLE_AI_MODEL`: The model to use (default is `gemini-1.0-pro`).

#### Dependencies

- Dependencies and installation steps will be listed here.

### Installation

1. **Clone the Repository**: If not already done, clone the FKS repository to your local machine.
2. **Navigate to Analyze Directory**: `cd /path/to/fks/repo/tools/analyze`
3. **Install Dependencies**: Run `pip install -r requirements.txt` to install necessary packages including `google-cloud-aiplatform`.
4. **Run the Service**: Start the analysis service with appropriate scripts or directly via Python modules.

### Usage

- **Project Management**: Use `fks_analyze` to track tasks, issues, and project status across FKS services. Run RAG analysis with `run_project_management_analysis` to get actionable insights.
- **Standardization**: Ensure consistent structure and coding standards across FKS repos using `run_standardization_analysis`.
- **Documentation**: Generate MkDocs sites and Mermaid diagrams for visual representation of workflows and architectures with `run_documentation_analysis`.
- **AI Agents**: Enable automated fixes and improvements using AI agents powered by the RAG system through `run_ai_agent_fixes`.

### Example Commands

- **Run Project Management Analysis**:
  ```bash
  python -m src.services.rag_pipeline run_project_management_analysis <job_id> --focus-area "tasks and issues" --scope "all"
  ```
- **Run Standardization Analysis**:
  ```bash
  python -m src.services.standardization run_standardization_analysis <job_id> --focus-area "code consistency" --scope "core"
  ```
- **Run Documentation Analysis**:
  ```bash
  python -m src.services.documentation run_documentation_analysis <job_id> --focus-area "mkdocs and diagrams" --scope "tools"
  ```
- **Run AI Agent Fixes**:
  ```bash
  python -m src.services.ai_agents run_ai_agent_fixes <job_id> --focus-area "code improvements" --scope "core"
  ```

## 🔧 Configuration

### Environment Variables

```bash
# Service Configuration
SERVICE_NAME=fks_analyze
SERVICE_PORT=8008

# Google AI API
GOOGLE_AI_API_KEY=your-google-ai-api-key
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_AI_MODEL=gemini-1.0-pro

# Analysis Configuration
ANALYSIS_MAX_FILES=1000
ANALYSIS_TIMEOUT_SECONDS=300
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 🐳 Docker

### Build

```bash
docker build -t nuniesmith/fks:analyze-latest .
```

### Run

```bash
docker run -p 8008:8008 \
  -e GOOGLE_AI_API_KEY=your-api-key \
  nuniesmith/fks:analyze-latest
```

## ☸️ Kubernetes

### Deployment

```bash
# Deploy using Helm
cd repo/main/k8s/charts/fks-platform
helm install fks-platform . -n fks-trading

# Or using the unified start script
cd /home/jordan/Documents/code/fks
./start.sh --type k8s
```

### Health Checks

Kubernetes probes:
- **Liveness**: `GET /live`
- **Readiness**: `GET /ready` (checks Google AI API connectivity)

## 📚 Documentation

- [API Documentation](docs/API.md)
- [RAG System Guide](docs/RAG_SYSTEM.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## 🔗 Integration

### Dependencies

- **Google AI API**: For RAG system LLM component
- **File System**: Access to FKS repositories for analysis

### Consumers

- **fks_web**: Consumes analyze API for project insights
- **CI/CD**: Automated code quality checks

## 📊 Monitoring

### Health Check Endpoints

- `GET /health` - Health check
- `GET /health/ready` - Readiness check

### Metrics

- Analysis job completion rates
- Google AI API usage and rate limits
- Analysis duration and file processing rates

### Logging

- Analysis job tracking
- RAG system interactions
- Error tracking and retries

## 🛠️ Development

### Setup

```bash
# Clone repository
git clone https://github.com/nuniesmith/fks_analyze.git
cd fks_analyze

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.dev.txt
```

### Code Structure

```
repo/analyze/
├── src/
│   ├── main.py              # FastAPI application
│   ├── services/
│   │   ├── analysis.py      # Analysis service
│   │   ├── rag_pipeline.py # RAG system
│   │   ├── retrieval.py     # Retrieval service
│   │   └── documentation.py # Documentation generation
│   └── api/
│       └── routes/          # API routes
├── tests/                   # Test suite
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Contributing

1. Follow Python best practices (PEP 8)
2. Write tests for new analysis features
3. Document RAG system changes
4. Update API documentation

---

**Repository**: [nuniesmith/fks_analyze](https://github.com/nuniesmith/fks_analyze)  
**Docker Image**: `nuniesmith/fks:analyze-latest`  
**Status**: Active
