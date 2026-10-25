#!/bin/sh
set -e

PORT="${PORT:-5000}"
echo "[AbiArt] Starting container on port $PORT..."
exec gunicorn app:app --bind 0.0.0.0:"$PORT" --workers 2 --timeout 300
