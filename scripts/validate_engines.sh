#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
APPLICATION="${APPLICATION:-${BUILD_DIR}/bin/egcinet_validate}"
IMAGES="${IMAGES:-${PROJECT_ROOT}/datasets/images/val}"
LABELS="${LABELS:-${PROJECT_ROOT}/datasets/labels/val}"
FP32_ENGINE="${FP32_ENGINE:-${PROJECT_ROOT}/models/egcinet_352_fp32.engine}"
FP16_ENGINE="${FP16_ENGINE:-${PROJECT_ROOT}/models/egcinet_352_fp16.engine}"
INT8_ENGINE="${INT8_ENGINE:-${PROJECT_ROOT}/models/egcinet_352_int8.engine}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/results/validation/summary.csv}"
WARMUP="${WARMUP:-20}"
ITERATIONS="${ITERATIONS:-200}"
MAX_IMAGES="${MAX_IMAGES:-0}"
THRESHOLD="${THRESHOLD:-0.5}"
VISUALIZE="${VISUALIZE:-1}"
MEAN="${MEAN:-140.505,157.845,135.660}"
STD="${STD:-61.455,60.180,62.220}"

if [ ! -x "${APPLICATION}" ]; then
    echo "[ERROR] validation executable not found: ${APPLICATION}" >&2
    echo "[INFO] run scripts/build.sh first" >&2
    exit 1
fi
if [ ! -d "${IMAGES}" ]; then
    echo "[ERROR] validation image directory not found: ${IMAGES}" >&2
    exit 1
fi
if [ ! -d "${LABELS}" ]; then
    echo "[ERROR] validation label directory not found: ${LABELS}" >&2
    exit 1
fi

ARGS=(
    --images "${IMAGES}"
    --labels "${LABELS}"
    --output "${OUTPUT}"
    --warmup "${WARMUP}"
    --iterations "${ITERATIONS}"
    --max-images "${MAX_IMAGES}"
    --threshold "${THRESHOLD}"
    --visualize "${VISUALIZE}"
    --mean "${MEAN}"
    --std "${STD}"
)

# 缺失的 engine 会跳过，便于先单独验证某一种精度。
ENGINE_COUNT=0
if [ -s "${FP32_ENGINE}" ]; then
    ARGS+=(--fp32 "${FP32_ENGINE}")
    ENGINE_COUNT=$((ENGINE_COUNT + 1))
else
    echo "[WARN] skip missing FP32 engine: ${FP32_ENGINE}" >&2
fi
if [ -s "${FP16_ENGINE}" ]; then
    ARGS+=(--fp16 "${FP16_ENGINE}")
    ENGINE_COUNT=$((ENGINE_COUNT + 1))
else
    echo "[WARN] skip missing FP16 engine: ${FP16_ENGINE}" >&2
fi
if [ -s "${INT8_ENGINE}" ]; then
    ARGS+=(--int8 "${INT8_ENGINE}")
    ENGINE_COUNT=$((ENGINE_COUNT + 1))
else
    echo "[WARN] skip missing INT8 engine: ${INT8_ENGINE}" >&2
fi

if [ "${ENGINE_COUNT}" -eq 0 ]; then
    echo "[ERROR] no engine is available" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"
echo "[INFO] images: ${IMAGES}"
echo "[INFO] labels: ${LABELS}"
echo "[INFO] CSV: ${OUTPUT}"
"${APPLICATION}" "${ARGS[@]}"

if [ ! -s "${OUTPUT}" ]; then
    echo "[ERROR] validation CSV was not generated: ${OUTPUT}" >&2
    exit 1
fi
echo "[PASS] validation CSV: ${OUTPUT}"
