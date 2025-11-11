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

# Install Python dependencies from requirements.txt
# Using requirements.txt with version constraints to avoid resolution issues
# Use --no-cache-dir to reduce disk usage in CI
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --user --no-warn-script-location --no-cache-dir -r requirements.txt

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
