# Multi-stage build for fks_analyze Python service
# Uses ML base image with LangChain, ChromaDB, and sentence-transformers pre-installed
FROM nuniesmith/fks:docker-ml AS builder

WORKDIR /app

# ML packages (langchain, chromadb, sentence-transformers, ollama) are already installed in base
# Just install service-specific packages (Google Cloud, document loaders, etc.)
COPY requirements.txt ./

# Install Python dependencies with BuildKit cache mount
# Use --no-cache-dir to reduce disk usage in CI
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --user --no-warn-script-location --no-cache-dir -r requirements.txt \
    && python -m pip cache purge || true \
    && rm -rf /root/.cache/pip/* /tmp/pip-* 2>/dev/null || true

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
