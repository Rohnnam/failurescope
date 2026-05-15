#!/bin/bash
set -e

echo "[FailureScope] Starting backend..."
cd /app
uvicorn backend.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level warning &

echo "[FailureScope] Waiting for backend..."
sleep 3

echo "[FailureScope] Starting nginx..."
nginx -g "daemon off;"
