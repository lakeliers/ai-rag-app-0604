#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting Python Agent API on http://127.0.0.1:8000"
python3 -m uvicorn api_server:app --reload --port 8000 &
API_PID=$!

cleanup() {
  kill "$API_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting Next.js frontend on http://127.0.0.1:3000"
cd "$ROOT_DIR/frontend"
npm run dev -- --port 3000
