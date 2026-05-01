#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${HYPURA_SKIP_DOTENV:-}" && -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

PORT="${HYPURA_PORT:-11435}"
MODELS_DIR="${HYPURA_MODELS_DIR:-/Volumes/256 M.2/story-engine-models}"
MODEL_PATH="${1:-${HYPURA_MODEL_PATH:-}}"
CONTEXT="${HYPURA_CONTEXT:-${LLM_NUM_CTX:-8192}}"

if [[ -n "${HYPURA_BIN:-}" ]]; then
  HYPURA_EXEC="$HYPURA_BIN"
elif [[ -x "$ROOT_DIR/third_party/hypura/target/release/hypura" ]]; then
  HYPURA_EXEC="$ROOT_DIR/third_party/hypura/target/release/hypura"
elif command -v hypura >/dev/null 2>&1; then
  HYPURA_EXEC="$(command -v hypura)"
else
  echo "Hypura binary not found. Build from source first:"
  echo "  cd third_party/hypura && cargo build --release"
  exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
  if [[ -n "${HYPURA_MODEL:-}" && -f "$MODELS_DIR/${HYPURA_MODEL}.gguf" ]]; then
    MODEL_PATH="$MODELS_DIR/${HYPURA_MODEL}.gguf"
  else
    FIRST_GGUF="$(find "$MODELS_DIR" -type f -name "*.gguf" | head -n 1 || true)"
    if [[ -n "$FIRST_GGUF" ]]; then
      MODEL_PATH="$FIRST_GGUF"
    fi
  fi
fi

if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
  echo "No GGUF model found."
  echo "Usage: $0 /path/to/model.gguf"
  echo "Or set HYPURA_MODEL_PATH or place *.gguf in $MODELS_DIR"
  exit 1
fi

echo "Starting Hypura"
echo "  binary: $HYPURA_EXEC"
echo "  model:  $MODEL_PATH"
echo "  port:   $PORT"
echo "  context:$CONTEXT"
echo "  kv-note: kv compression is runtime auto-selection in Hypura scheduler"

"$HYPURA_EXEC" serve "$MODEL_PATH" --port "$PORT" --context "$CONTEXT"
