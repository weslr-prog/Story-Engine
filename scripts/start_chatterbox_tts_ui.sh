#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHATTERBOX_DIR="$ROOT_DIR/chatterbox"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

PYTHON_BIN="${CHATTERBOX_PYTHON_BIN:-/opt/homebrew/bin/python3.11}"
HOST="${2:-${CHATTERBOX_HOST:-127.0.0.1}}"
PORT="${1:-${CHATTERBOX_PORT:-7865}}"
APP_FILE="${CHATTERBOX_GRADIO_APP:-gradio_tts_app.py}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "No usable Python interpreter found for Chatterbox UI."
    exit 1
  fi
fi

if [[ ! -f "$CHATTERBOX_DIR/$APP_FILE" ]]; then
  echo "Chatterbox app not found: $CHATTERBOX_DIR/$APP_FILE"
  exit 1
fi

export PYTHONPATH="$CHATTERBOX_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export GRADIO_SERVER_NAME="$HOST"
export GRADIO_SERVER_PORT="$PORT"
export GRADIO_SHARE="${GRADIO_SHARE:-false}"

cd "$CHATTERBOX_DIR"
echo "Starting repo-local Chatterbox TTS UI"
echo "  python: $PYTHON_BIN"
echo "  app:    $APP_FILE"
echo "  url:    http://$HOST:$PORT"

exec "$PYTHON_BIN" "$APP_FILE"