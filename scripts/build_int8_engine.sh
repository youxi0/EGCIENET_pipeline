#!/bin/bash

set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="${PROJECT_ROOT}/build"
BUILDER="${BUILD_DIR}/bin/egcinet_build_engine"

ONNX="${ONNX:-${PROJECT_ROOT}/models/egcinet_352.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcinet_352_int8.engine}"
CALIB_DIR="${CALIB_DIR:-${PROJECT_ROOT}/datasets/images/calibration}"
CALIB_CACHE="${CALIB_CACHE:-${PROJECT_ROOT}/models/egcinet_352_int8.cache}"
INPUT_NAME="${INPUT_NAME:-image}"
INPUT_W="${INPUT_W:-352}"
INPUT_H="${INPUT_H:-352}"
MEAN="${MEAN:-140.505,157.845,135.66}"
STD="${STD:-61.455,60.18,62.22}"
CALIB_BATCH="${CALIB_BATCH:-1}"
CALIB_MAX_IMAGES="${CALIB_MAX_IMAGES:-500}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"
FORCE_LAYERNORM_FP32="${FORCE_LAYERNORM_FP32:-1}"

if [ ! -x "${BUILDER}" ]; then
    echo "[ERROR] builder not found: ${BUILDER}"
    echo "[INFO] run scripts/build.sh first"
    exit 1
fi

echo "[INFO] build INT8 engine"
echo "[INFO] onnx: ${ONNX}"
echo "[INFO] engine: ${ENGINE}"
echo "[INFO] calib dir: ${CALIB_DIR}"
echo "[INFO] calib cache: ${CALIB_CACHE}"

"${BUILDER}" \
    --onnx "${ONNX}" \
    --engine "${ENGINE}" \
    --precision int8 \
    --input_name "${INPUT_NAME}" \
    --input_w "${INPUT_W}" \
    --input_h "${INPUT_H}" \
    --mean "${MEAN}" \
    --std "${STD}" \
    --calib_dir "${CALIB_DIR}" \
    --calib_cache "${CALIB_CACHE}" \
    --calib_batch "${CALIB_BATCH}" \
    --calib_max_images "${CALIB_MAX_IMAGES}" \
    --workspace_mib "${WORKSPACE_MIB}" \
    --force_layernorm_fp32 "${FORCE_LAYERNORM_FP32}"

echo "[INFO] INT8 engine finished: ${ENGINE}"
