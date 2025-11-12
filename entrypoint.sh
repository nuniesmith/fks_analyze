#!/bin/bash
# Entrypoint script for fks_analyze

set -e

# Default values
SERVICE_NAME=${SERVICE_NAME:-fks_analyze}
SERVICE_PORT=${SERVICE_PORT:-8008}
HOST=${HOST:-0.0.0.0}

echo "Starting ${SERVICE_NAME} on ${HOST}:${SERVICE_PORT}"

# Run the service
exec python -m uvicorn src.main:app \
    --host "${HOST}" \
    --port "${SERVICE_PORT}" \
    --no-access-log \
    --log-level info

