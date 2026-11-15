#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[Xyvora AI] Initializing Cognitive Engine & Neural Cortex on 127.0.0.1:8000..."
cd "$ROOT_DIR/backend"
"$ROOT_DIR/.venv/bin/uvicorn" main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 40); do
  if "$ROOT_DIR/.venv/bin/python" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=1)" 2>/dev/null; then
    echo "[Xyvora AI] Backend Cortex ready!"
    break
  fi
  sleep 0.4
done

echo "[Xyvora AI] Launching Interactive UI on 0.0.0.0:5173..."
cd "$ROOT_DIR/frontend"
exec npm run dev
