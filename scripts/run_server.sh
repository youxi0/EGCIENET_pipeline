#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "${PROJECT_ROOT}/scripts/tensorrt_env.sh"

BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
EXECUTABLE="${BUILD_DIR}/bin/egcinet_pipeline"

ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcinet_352_fp16.engine}"
SOURCE="${SOURCE:-${PROJECT_ROOT}/datasets/images/val}"
TYPE="${TYPE:-folder}"
QUEUE_SIZE="${QUEUE_SIZE:-3}"
MAX_WIDTH="${MAX_WIDTH:-1920}"
MAX_HEIGHT="${MAX_HEIGHT:-1080}"
THRESHOLD="${THRESHOLD:-0.5}"
SAVE_DIR="${SAVE_DIR:-}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/results/logs}"

if [ ! -f "${EXECUTABLE}" ]; then
    echo "[ERROR] executable not found: ${EXECUTABLE}"
    echo "[HINT] run ./scripts/build.sh first"
    exit 1
fi

if [ ! -f "${ENGINE}" ]; then
    echo "[ERROR] engine not found: ${ENGINE}"
    exit 1
fi

if [ "${TYPE}" != "camera" ] && [ ! -e "${SOURCE}" ]; then
    echo "[ERROR] source not found: ${SOURCE}"
    exit 1
fi

configure_tensorrt_library_path

echo "[INFO] start EGCINET pipeline"
echo "[INFO] engine    : ${ENGINE}"
echo "[INFO] source    : ${SOURCE}"
echo "[INFO] type      : ${TYPE}"
echo "[INFO] queue size: ${QUEUE_SIZE}"
echo "[INFO] max source: ${MAX_WIDTH}x${MAX_HEIGHT}"
echo "[INFO] threshold : ${THRESHOLD}"
echo "[INFO] log dir   : ${LOG_DIR}"

ARGS=(
    --engine "${ENGINE}"
    --source "${SOURCE}"
    --type "${TYPE}"
    --queue_size "${QUEUE_SIZE}"
    --max_width "${MAX_WIDTH}"
    --max_height "${MAX_HEIGHT}"
    --threshold "${THRESHOLD}"
    --log_dir "${LOG_DIR}"
)

if [ -n "${SAVE_DIR}" ]; then
    ARGS+=(--save_dir "${SAVE_DIR}")
fi

"${EXECUTABLE}" "${ARGS[@]}"
