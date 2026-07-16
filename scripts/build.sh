#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
TENSORRT_ROOT="${TENSORRT_ROOT:-/home/fulin/haiyanghuang/TensorRT-8.6.1.6}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

echo "[INFO] project root: ${PROJECT_ROOT}"
echo "[INFO] build dir: ${BUILD_DIR}"

cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
    -DEGCINET_BUILD_PIPELINE=ON \
    -DEGCINET_BUILD_INT8_CALIBRATOR=ON \
    -DTENSORRT_ROOT="${TENSORRT_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

echo "[PASS] build finished"
echo "[INFO] inference: ${BUILD_DIR}/bin/egcinet_infer_image"
echo "[INFO] validation: ${BUILD_DIR}/bin/egcinet_validate"
echo "[INFO] INT8 calibration: ${BUILD_DIR}/bin/egcinet_calibrate_int8"
