# Multi-stage build for fks_analyze Python service
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies needed for chromadb, sentence-transformers, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    cmake \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel first (better caching)
# Using BuildKit cache mount for faster pip downloads (requires DOCKER_BUILDKIT=1)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel

# Copy dependency files
COPY requirements.txt ./

# Install Python dependencies
# The staged approach helps pip resolve dependencies more efficiently
# by installing compatible packages together and reducing backtracking

# Stage 1: Install core web framework packages first (stable, well-defined deps)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --user --no-warn-script-location \
    fastapi "uvicorn[standard]" pydantic pydantic-settings python-multipart \
    typer pyyaml tqdm prometheus-client "httpx>=0.26.0,<0.29.0" "redis>=5.0.0,<8.0.0" \
    json5==0.9.24 loguru markdown python-docx

# Stage 2: Install Google Cloud packages (pinned for compatibility)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --user --no-warn-script-location \
    google-cloud-aiplatform==1.43.0

# Stage 3: Install LangChain packages (complex dependency tree)
# Install core langchain packages together to allow pip to resolve dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --user --no-warn-script-location \
    "langchain>=0.3.0" \
    "langchain-community>=0.3.0" \
    "langchain-core>=0.3.0" \
    "langchain-text-splitters>=0.3.0"

# Stage 4: Install LangChain integrations
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --user --no-warn-script-location \
    "langchain-google-genai>=1.0.0,<4.0.0" \
    "google-generativeai>=0.3.0,<1.0.0" \
    "langchain-ollama>=0.1.0,<0.2.0" \
    "ollama>=0.1.0"

# Stage 5: Install heavy packages (chromadb, sentence-transformers, pypdf)
# These take the longest to build/install
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --user --no-warn-script-location \
    "chromadb>=0.4.0,<0.5.0" \
    "sentence-transformers>=2.2.0,<6.0.0" \
    "pypdf>=3.17.0,<7.0.0"

# Runtime stage
FROM python:3.12-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVICE_NAME=fks_analyze \
    SERVICE_PORT=8008 \
    PYTHONPATH=/app/src:/app \
    PATH=/home/appuser/.local/bin:$PATH \
    GOOGLE_CLOUD_PROJECT="" \
    GOOGLE_CLOUD_LOCATION="us-central1" \
    GOOGLE_AI_MODEL="gemini-1.0-pro"

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user FIRST (before copying files)
# This ensures proper ownership from the start
RUN useradd -u 1000 -m -s /bin/bash appuser

# Copy Python packages from builder with correct ownership
# Using --chown avoids the need for recursive chown later
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application source with correct ownership
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser entrypoint.sh ./

# Make entrypoint executable (already owned by appuser from COPY --chown)
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request,sys;port=os.getenv('SERVICE_PORT','8008');u=f'http://localhost:{port}/health';\
import urllib.error;\
try: urllib.request.urlopen(u,timeout=3);\
except Exception: sys.exit(1)" || exit 1

# Expose the service port
EXPOSE 8008

# Use entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
