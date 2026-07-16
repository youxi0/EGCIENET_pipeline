#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

TENSORRT_ROOT="${TENSORRT_ROOT:-/home/fulin/haiyanghuang/TensorRT-8.6.1.6}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
CALIBRATOR="${CALIBRATOR:-${BUILD_DIR}/bin/egcinet_calibrate_int8}"

ONNX="${ONNX:-${PROJECT_ROOT}/models/egcinet_352.onnx}"
CALIB_DIR="${CALIB_DIR:-${PROJECT_ROOT}/Dataset/AEBIS/cal}"
CALIB_CACHE="${CALIB_CACHE:-${PROJECT_ROOT}/models/egcinet_352_int8.cache}"
INPUT_NAME="${INPUT_NAME:-image}"
INPUT_W="${INPUT_W:-352}"
INPUT_H="${INPUT_H:-352}"
MEAN="${MEAN:-140.505,157.845,135.66}"
STD="${STD:-61.455,60.18,62.22}"
CALIB_BATCH="${CALIB_BATCH:-1}"
CALIB_MAX_IMAGES="${CALIB_MAX_IMAGES:-500}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"

if [ ! -x "${CALIBRATOR}" ]; then
    echo "[ERROR] INT8 calibrator not found: ${CALIBRATOR}" >&2
    echo "[INFO] run: bash scripts/build.sh" >&2
    exit 1
fi
if [ ! -s "${ONNX}" ]; then
    echo "[ERROR] ONNX model not found: ${ONNX}" >&2
    exit 1
fi
if [ ! -d "${CALIB_DIR}" ]; then
    echo "[ERROR] calibration image directory not found: ${CALIB_DIR}" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${TENSORRT_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$(dirname "${CALIB_CACHE}")"

echo "[INFO] generate INT8 calibration cache"
echo "[INFO] images: ${CALIB_DIR}"
echo "[INFO] cache: ${CALIB_CACHE}"
"${CALIBRATOR}" \
    --onnx "${ONNX}" \
    --calib_dir "${CALIB_DIR}" \
    --calib_cache "${CALIB_CACHE}" \
    --input_name "${INPUT_NAME}" \
    --input_w "${INPUT_W}" \
    --input_h "${INPUT_H}" \
    --batch "${CALIB_BATCH}" \
    --max_images "${CALIB_MAX_IMAGES}" \
    --mean "${MEAN}" \
    --std "${STD}" \
    --workspace_mib "${WORKSPACE_MIB}"

if [ ! -s "${CALIB_CACHE}" ]; then
    echo "[ERROR] calibration cache was not generated: ${CALIB_CACHE}" >&2
    exit 1
fi

echo "[PASS] INT8 calibration cache: ${CALIB_CACHE}"
