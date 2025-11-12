#!/bin/bash
# Start FKS Analyze Service

set -e

echo "Starting FKS Analyze Service..."

# Check if running in container
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
    exec uvicorn src.main:app --host 0.0.0.0 --port "${PORT:-8008}"
else
    echo "Running locally"
    
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Start with reload for development
    uvicorn src.main:app --reload --host 0.0.0.0 --port "${PORT:-8008}"
fi
