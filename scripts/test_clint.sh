#!/bin/bash

set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"

echo "[INFO] start test client"
echo "[INFO] host: ${HOST}"
echo "[INFO] port: ${PORT}"

python3 "${PROJECT_ROOT}/scripts/test_client.py" \
    --host "${HOST}" \
    --port "${PORT}"