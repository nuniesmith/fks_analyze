#!/bin/bash
# Rebuild Docker image with proper settings
set -e

echo "=========================================="
echo "FKS Analyze - Docker Rebuild Script"
echo "=========================================="
echo ""

# Check if BuildKit is available
if docker buildx version > /dev/null 2>&1; then
    echo "✓ BuildKit is available"
    USE_BUILDKIT=1
else
    echo "⚠ BuildKit not detected, using standard build"
    USE_BUILDKIT=0
fi

echo ""
echo "Stopping any running containers..."
docker compose down 2>/dev/null || true

echo ""
echo "Cleaning up old build cache (optional, removes all unused cache)..."
read -p "Remove all unused build cache? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker builder prune -f
    echo "✓ Build cache cleaned"
else
    echo "Skipping cache cleanup"
fi

echo ""
echo "=========================================="
echo "Starting rebuild..."
echo "This will take 10-15 minutes on first build"
echo "=========================================="
echo ""

if [ "$USE_BUILDKIT" -eq 1 ]; then
    echo "Building with BuildKit (faster caching)..."
    DOCKER_BUILDKIT=1 docker compose build --no-cache --progress=plain
else
    echo "Building with standard Docker..."
    docker compose build --no-cache
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Build completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Build failed. Check errors above."
    echo "=========================================="
    exit 1
fi
