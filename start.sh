#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log(){ local lvl="$1"; shift; local msg="$*"; case "$lvl" in INFO) echo -e "${GREEN}[INFO]${NC} $msg";; WARN) echo -e "${YELLOW}[WARN]${NC} $msg";; ERROR) echo -e "${RED}[ERROR]${NC} $msg";; DEBUG) echo -e "${BLUE}[DEBUG]${NC} $msg";; esac; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
[ -f "$ENV_FILE" ] && set -a && source "$ENV_FILE" && set +a || log WARN "Root .env not found; continuing with current env"
# Central compose now lives inside fks_master alongside this script
: "${COMPOSE_DIR:=$SCRIPT_DIR}"
COMPOSE_BASE="$COMPOSE_DIR/docker-compose.yml"
COMPOSE_DEV="$COMPOSE_DIR/docker-compose.dev.yml"
COMPOSE_PROD="$COMPOSE_DIR/docker-compose.prod.yml"

USE_DEV=false
USE_PROD=false
NO_BUILD=false
WAIT_SECS=12
HEALTH_RETRIES=15
HEALTH_INTERVAL=2
FOLLOW_LOGS=false
ADDITIONAL_SERVICES=()
STACK_NAME=""
LOCAL_MODE=false
WITH_TRANSFORMER=false

show_help(){ cat <<EOF
Usage: ./start.sh [options] [-- service1 service2 ...]

Options:
  --dev              Include development overrides (hot reload etc.)
  --prod             Include production overrides (resource limits, images)
  --no-build         Skip build/pull; just start existing images
  --wait SECONDS     Wait extra seconds after 'up' (default: $WAIT_SECS)
  --health-retries N Number of health probe retries (default: $HEALTH_RETRIES)
  --health-interval S Seconds between health probes (default: $HEALTH_INTERVAL)
  --logs             Follow logs after successful start
  --stack NAME       Use compose in <repo_root>/NAME (auto-detect fks_master if present)
  --local            Force use of this directory's docker-compose (skip auto central detection)
  --with-transformer Start sibling fks_transformer (for OLLAMA_BASE_URL)
  --help             Show this help

Service selection:
  By default all services in base compose are started. To restrict, append -- then names:
    ./start.sh --dev -- fks_api postgres redis
EOF
}

parse_args(){
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dev) USE_DEV=true; shift;;
      --prod) USE_PROD=true; shift;;
      --no-build) NO_BUILD=true; shift;;
      --wait) WAIT_SECS="${2:-}"; shift 2;;
      --health-retries) HEALTH_RETRIES="${2:-}"; shift 2;;
      --health-interval) HEALTH_INTERVAL="${2:-}"; shift 2;;
  --logs) FOLLOW_LOGS=true; shift;;
  --stack) STACK_NAME="${2:-}"; shift 2;;
  --local) LOCAL_MODE=true; shift;;
  --with-transformer) WITH_TRANSFORMER=true; shift;;
      --help|-h) show_help; exit 0;;
      --) shift; ADDITIONAL_SERVICES=("$@"); break;;
      *) log ERROR "Unknown option: $1"; show_help; exit 1;;
    esac
  done
}

check_tools(){
  if command -v docker-compose >/dev/null 2>&1; then COMPOSE_CMD="docker-compose"; elif docker compose version >/dev/null 2>&1; then COMPOSE_CMD="docker compose"; else log ERROR "Docker Compose not available"; exit 1; fi
  docker info >/dev/null 2>&1 || { log ERROR "Docker daemon not reachable"; exit 1; }
  command -v curl >/dev/null 2>&1 || log WARN "curl not installed; HTTP health checks may fail"
}

resolve_compose_dir(){
  if [ -n "$STACK_NAME" ]; then
    local candidate="$ROOT_DIR/$STACK_NAME"
    if [ -f "$candidate/docker-compose.yml" ]; then
      COMPOSE_DIR="$candidate"
      COMPOSE_BASE="$candidate/docker-compose.yml"
      COMPOSE_DEV="$candidate/docker-compose.dev.yml"
      COMPOSE_PROD="$candidate/docker-compose.prod.yml"
      log INFO "Using stack: $STACK_NAME"
    else
      log ERROR "Stack '$STACK_NAME' not found at $candidate"; exit 1
    fi
  elif ! $LOCAL_MODE && [ -d "$ROOT_DIR/fks_master" ] && [ -f "$ROOT_DIR/fks_master/docker-compose.yml" ] && [ "$COMPOSE_DIR" = "$SCRIPT_DIR" ]; then
    COMPOSE_DIR="$ROOT_DIR/fks_master"
    COMPOSE_BASE="$COMPOSE_DIR/docker-compose.yml"
    COMPOSE_DEV="$COMPOSE_DIR/docker-compose.dev.yml"
    COMPOSE_PROD="$COMPOSE_DIR/docker-compose.prod.yml"
    log INFO "Auto-detected central compose at $COMPOSE_DIR"
  fi
}

build_or_pull(){
  resolve_compose_dir
  local files=("-f" "$COMPOSE_BASE")
  $USE_DEV && [ -f "$COMPOSE_DEV" ] && files+=( -f "$COMPOSE_DEV")
  $USE_PROD && [ -f "$COMPOSE_PROD" ] && files+=( -f "$COMPOSE_PROD")
  if $NO_BUILD; then
    log INFO "Skipping build/pull as requested (--no-build)"
  else
    log INFO "Ensuring images are present (build if context present, pull otherwise)"
    $COMPOSE_CMD "${files[@]}" build --pull --parallel || log WARN "Build step had issues; continuing"
  fi
  FILE_ARGS=("${files[@]}")
}

bring_up(){
  local up_args=("${FILE_ARGS[@]}" up -d)
  if ((${#ADDITIONAL_SERVICES[@]})); then up_args+=("${ADDITIONAL_SERVICES[@]}"); fi
  log INFO "Starting services..."
  $COMPOSE_CMD "${up_args[@]}"
}

start_transformer(){
  $WITH_TRANSFORMER || return 0
  local tdir="$ROOT_DIR/fks_transformer"
  if [ ! -d "$tdir" ]; then
    log WARN "fks_transformer directory not found; skipping --with-transformer"
    return 0
  fi
  log INFO "Starting transformer (fks_transformer)"
  ( cd "$tdir" && docker compose up -d --build fks_transformer ) || log WARN "Failed to start transformer"
}

health_checks(){
  log INFO "Running health probes (retries=$HEALTH_RETRIES interval=${HEALTH_INTERVAL}s)"
  local attempts=0
  local ok_api=true ok_analyze=true ok_db=true ok_redis=true ok_transformer=true
  docker ps --format '{{.Names}}' | grep -q '^fks_api$' && ok_api=false
  docker ps --format '{{.Names}}' | grep -q '^fks_analyze$' && ok_analyze=false
  docker ps --format '{{.Names}}' | grep -q '^fks_postgres$' && ok_db=false
  docker ps --format '{{.Names}}' | grep -q '^fks_redis$' && ok_redis=false
  if $WITH_TRANSFORMER && docker ps --format '{{.Names}}' | grep -q '^fks_transformer'; then ok_transformer=false; fi
  while (( attempts < HEALTH_RETRIES )); do
    attempts=$((attempts+1))
    ! $ok_api && curl -s -f http://localhost:8000/health >/dev/null 2>&1 && { ok_api=true; log INFO "fks_api healthy"; }
    ! $ok_analyze && curl -s -f http://localhost:4802/health >/dev/null 2>&1 && { ok_analyze=true; log INFO "fks_analyze healthy"; }
    ! $ok_db && docker exec fks_postgres pg_isready -U fks_user >/dev/null 2>&1 && { ok_db=true; log INFO "Postgres ready"; }
    ! $ok_redis && docker exec fks_redis redis-cli ping 2>/dev/null | grep -q PONG && { ok_redis=true; log INFO "Redis ready"; }
  ! $ok_transformer && curl -s -f http://localhost:4500/health >/dev/null 2>&1 && { ok_transformer=true; log INFO "fks_transformer healthy"; }
  if $ok_api && $ok_analyze && $ok_db && $ok_redis && $ok_transformer; then break; fi
    sleep "$HEALTH_INTERVAL"
  done
  if ! ($ok_api && $ok_analyze && $ok_db && $ok_redis && $ok_transformer); then
    log WARN "Health summary: api=$ok_api analyze=$ok_analyze db=$ok_db redis=$ok_redis transformer=$ok_transformer"
  fi
}

summary(){
  log INFO "Compose ps:"; $COMPOSE_CMD "${FILE_ARGS[@]}" ps || true
  log INFO "Key endpoints:"
  echo "  API:        http://localhost:8000 (if running)"
  echo "  Analyze:    http://localhost:4802 (if running)"
  echo "  Auth:       http://localhost:9000 (if running)"
  echo "  Master:     http://localhost:3030 (if running)"
  $WITH_TRANSFORMER && echo "  Transformer:http://localhost:4500 (if running)"
}

follow_logs(){
  $FOLLOW_LOGS || return 0
  log INFO "Following logs (Ctrl+C to detach)"
  $COMPOSE_CMD "${FILE_ARGS[@]}" logs -f ${ADDITIONAL_SERVICES[*]:-}
}

main(){
  parse_args "$@"
  check_tools
  # Ensure shared_docker symlink(s) exist before building images
  if [ -f "$ROOT_DIR/ensure-shared_symlinks.sh" ]; then
    log INFO "Ensuring shared_docker symlinks"
    bash "$ROOT_DIR/ensure-shared_symlinks.sh" || log WARN "Symlink helper encountered an issue"
  fi
  start_transformer
  build_or_pull
  bring_up
  log INFO "Waiting ${WAIT_SECS}s for initial boot"; sleep "$WAIT_SECS"
  health_checks
  summary
  follow_logs
}

main "$@"
