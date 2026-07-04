#!/bin/bash

set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXECUTABLE="${PROJECT_ROOT}/build/bin/blade_demo"

# ${变量名:-默认值}
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/best_fp16.engine}"
SOURCE="${SOURCE:-${PROJECT_ROOT}/data/blade_images}"
TYPE="folder"
PORT=9000
CLASSES="crack,corrosion,coating_loss,material_loss"

if [ ! -f "${EXECUTABLE}" ]; then
    echo "[ERROR] executable not found: ${EXECUTABLE}"
    echo "[HINT] run ./scripts/build.sh first"
    exit 1
fi

if [ ! -f "${ENGINE}" ]; then
    echo "[ERROR] engine not found: ${ENGINE}"
    exit 1
fi

if [ ! -e "${SOURCE}" ]; then
    echo "[ERROR] source not found: ${SOURCE}"
    exit 1
fi

echo "[INFO] start blade server"
echo "[INFO] engine : ${ENGINE}"
echo "[INFO] source : ${SOURCE}"
echo "[INFO] type   : ${TYPE}"
echo "[INFO] port   : ${PORT}"
echo "[INFO] classes: ${CLASSES}"

"${EXECUTABLE}" \
    --engine "${ENGINE}" \
    --source "${SOURCE}" \
    --type "${TYPE}" \
    --port "${PORT}" \
    --classes "${CLASSES}"