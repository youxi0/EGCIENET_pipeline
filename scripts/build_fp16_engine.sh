#!/bin/bash

set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="${PROJECT_ROOT}/build"
BUILDER="${BUILD_DIR}/bin/egcinet_build_engine"

ONNX="${ONNX:-${PROJECT_ROOT}/models/egcinet_352.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcinet_352_fp16.engine}"
INPUT_NAME="${INPUT_NAME:-image}"
INPUT_W="${INPUT_W:-352}"
INPUT_H="${INPUT_H:-352}"
MEAN="${MEAN:-140.505,157.845,135.66}"
STD="${STD:-61.455,60.18,62.22}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"

if [ ! -x "${BUILDER}" ]; then
    echo "[ERROR] builder not found: ${BUILDER}"
    echo "[INFO] run scripts/build.sh first"
    exit 1
fi

echo "[INFO] build FP16 engine"
echo "[INFO] onnx: ${ONNX}"
echo "[INFO] engine: ${ENGINE}"

"${BUILDER}" \
    --onnx "${ONNX}" \
    --engine "${ENGINE}" \
    --precision fp16 \
    --input_name "${INPUT_NAME}" \
    --input_w "${INPUT_W}" \
    --input_h "${INPUT_H}" \
    --mean "${MEAN}" \
    --std "${STD}" \
    --workspace_mib "${WORKSPACE_MIB}"

echo "[INFO] FP16 engine finished: ${ENGINE}"
