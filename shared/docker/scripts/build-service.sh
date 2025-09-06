#!/usr/bin/env bash
set -euo pipefail

# build-service.sh
# Unified builder for services using shared/shared_docker/Dockerfile
#
# Usage:
#   ./build-service.sh <service_path> [--runtime python] [--type api] [--gpu] [--tag myimage:dev] [--no-cache] [--arg KEY=VALUE]...
#
# Examples:
#   ./build-service.sh fks/fks_api --runtime python --type api --tag fks_api:dev
#   ./build-service.sh fks/fks_web --runtime node --type web --tag fks_web:dev
#   ./build-service.sh fks/fks_training --runtime python --type training --gpu --tag fks_training:gpu
#
# Notes:
#  * Resolves shared Dockerfile path relative to workspace root.
#  * Passes SERVICE_RUNTIME, SERVICE_TYPE and BUILD_TYPE (gpu when --gpu) as build args.
#  * Additional --arg KEY=VALUE flags are appended as build args.

if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -lt 1 ]]; then
  grep '^#' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

WORKSPACE_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
SHARED_DOCKERFILE="$WORKSPACE_ROOT/shared/shared_docker/Dockerfile"
if [[ ! -f "$SHARED_DOCKERFILE" ]]; then
  # If we are inside the shared/shared_docker repo itself, workspace root is its parent directory
  if [[ -f "$WORKSPACE_ROOT/Dockerfile" && $(basename "$WORKSPACE_ROOT") == "shared_docker" ]]; then
    WORKSPACE_ROOT=$(dirname "$WORKSPACE_ROOT")
  fi
  echo "[error] shared Dockerfile not found at $SHARED_DOCKERFILE" >&2
  exit 1
fi

SERVICE_PATH=$1; shift || true
if [[ ! -d "$SERVICE_PATH" ]]; then
  echo "[error] service path $SERVICE_PATH does not exist" >&2; exit 1;
fi
SHARED_DOCKERFILE="$WORKSPACE_ROOT/shared/shared_docker/Dockerfile"

RUNTIME="python"
TYPE="app"
GPU=false
TAG=""
NO_CACHE=false
EXTRA_BUILD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime) RUNTIME=$2; shift 2;;
    --type) TYPE=$2; shift 2;;
    --gpu) GPU=true; shift;;
    --tag) TAG=$2; shift 2;;
    --no-cache) NO_CACHE=true; shift;;
    --arg) EXTRA_BUILD_ARGS+=("$2"); shift 2;;
    *) echo "[warn] unknown arg: $1"; shift;;
  esac
done

if [[ -z "$TAG" ]]; then
  SANITIZED=$(echo "${SERVICE_PATH}" | tr '/' '_' )
  TAG="${SANITIZED}:local"
fi

BUILD_TYPE=cpu
[[ $GPU == true ]] && BUILD_TYPE=gpu

echo "[info] Building service: $SERVICE_PATH"
echo "       Runtime: $RUNTIME  Type: $TYPE  GPU: $GPU  Tag: $TAG"

BUILD_CMD=(docker build "$SERVICE_PATH" \
  -f "$SHARED_DOCKERFILE" \
  --build-arg SERVICE_RUNTIME="$RUNTIME" \
  --build-arg SERVICE_TYPE="$TYPE" \
  --build-arg BUILD_TYPE="$BUILD_TYPE" \
  -t "$TAG")

if [[ $NO_CACHE == true ]]; then
  BUILD_CMD+=(--no-cache)
fi

for kv in "${EXTRA_BUILD_ARGS[@]}"; do
  BUILD_CMD+=(--build-arg "$kv")
done

echo "[info] Running: ${BUILD_CMD[*]}"
"${BUILD_CMD[@]}"
echo "[info] Built image $TAG"
