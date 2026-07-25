#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "${PROJECT_ROOT}/scripts/tensorrt_env.sh"

BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
BUILD_INT8_CALIBRATOR="${BUILD_INT8_CALIBRATOR:-ON}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

echo "[INFO] project root: ${PROJECT_ROOT}"
echo "[INFO] build dir: ${BUILD_DIR}"

TENSORRT_ARGS=()
if [ -n "${TENSORRT_ROOT:-}" ]; then
    TENSORRT_ARGS+=("-DTENSORRT_ROOT=${TENSORRT_ROOT}")
    echo "[INFO] TensorRT root: ${TENSORRT_ROOT}"
else
    echo "[INFO] TensorRT: use system multiarch installation"
fi
configure_tensorrt_library_path

cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
    -DEGCINET_BUILD_PIPELINE=ON \
    "-DEGCINET_BUILD_INT8_CALIBRATOR=${BUILD_INT8_CALIBRATOR}" \
    "${TENSORRT_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

echo "[PASS] build finished"
echo "[INFO] pipeline: ${BUILD_DIR}/bin/egcinet_pipeline"
echo "[INFO] inference: ${BUILD_DIR}/bin/egcinet_infer_image"
echo "[INFO] validation: ${BUILD_DIR}/bin/egcinet_validate"
echo "[INFO] INT8 calibration: ${BUILD_DIR}/bin/egcinet_calibrate_int8"
