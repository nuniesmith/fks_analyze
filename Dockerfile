# Optimized Dockerfile for fks_analyze - Uses base image directly to reduce size
# Uses ML base image with all dependencies pre-installed
FROM nuniesmith/fks:docker-ml-latest

WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVICE_NAME=fks_analyze \
    SERVICE_PORT=8008 \
    PYTHONPATH=/app/src:/app \
    PATH=/usr/local/bin:$PATH \
    GOOGLE_CLOUD_PROJECT="" \
    GOOGLE_CLOUD_LOCATION="us-central1" \
    GOOGLE_AI_MODEL="gemini-1.0-pro"

# Install only service-specific packages
# Base image already has: langchain, chromadb, sentence-transformers, ollama, TA-Lib, numpy, pandas, httpx
COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip cache purge || true \
    && rm -rf /root/.cache/pip/* /tmp/pip-* 2>/dev/null || true

# Create non-root user if it doesn't exist
RUN id -u appuser 2>/dev/null || useradd -u 1000 -m -s /bin/bash appuser

# Copy application source
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser entrypoint.sh ./

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER appuser

# Verify uvicorn is accessible
RUN python3 -c "import uvicorn; print(f'✅ uvicorn found: {uvicorn.__file__}')" || echo "⚠️  uvicorn verification failed"

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
