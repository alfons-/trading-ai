#!/bin/bash
# Auto-deploy: pull changes from GitHub, install deps, restart bot.
# Designed to run periodically via launchd on the production Mac.
#
# Usage (manual):   ./scripts/deploy.sh
# Usage (launchd):  see com.tradedan.deploy.plist

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

BRANCH="${DEPLOY_BRANCH:-main}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/deploy.log"

log() { echo "[deploy $(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

CURRENT=$(git rev-parse HEAD)
git fetch origin "$BRANCH" --quiet

REMOTE=$(git rev-parse "origin/$BRANCH")
if [ "$CURRENT" = "$REMOTE" ]; then
    exit 0
fi

log "Cambios detectados: $CURRENT -> $REMOTE"
git pull origin "$BRANCH" --ff-only >> "$LOGFILE" 2>&1

if git diff "$CURRENT" "$REMOTE" --name-only | grep -q "requirements.txt"; then
    log "requirements.txt cambió, instalando dependencias..."
    source "$PROJECT_DIR/.venv/bin/activate"
    pip install -r requirements.txt -q >> "$LOGFILE" 2>&1
fi

CONFIG="${DEPLOY_CONFIG:-configs/execution.yaml}"

if pgrep -f "scripts.run_live" > /dev/null 2>&1; then
    log "Deteniendo bot anterior..."
    pkill -f "scripts.run_live" || true
    sleep 2
fi

log "Iniciando bot con config: $CONFIG"
source "$PROJECT_DIR/.venv/bin/activate"
nohup python -m scripts.run_live --config "$CONFIG" >> "$LOG_DIR/bot.log" 2>&1 &
BOT_PID=$!
log "Bot iniciado (PID: $BOT_PID)"
