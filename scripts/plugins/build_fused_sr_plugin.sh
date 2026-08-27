#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build/fused_sr_plugin}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-87}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

CMAKE_ARGS=(
    -S "${PROJECT_ROOT}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DEGCINET_BUILD_FUSED_SR_PLUGIN=ON
    -DEGCINET_BUILD_PACKED_DWCONV_PLUGIN=OFF
    -DEGCINET_BUILD_BLOCK1_FUSED_PLUGIN=OFF
    -DEGCINET_BUILD_INT8_CALIBRATOR=OFF
    -DEGCINET_BUILD_PIPELINE=OFF
    "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
)

if [ -n "${TENSORRT_ROOT:-}" ]; then
    CMAKE_ARGS+=("-DTENSORRT_ROOT=${TENSORRT_ROOT}")
fi

echo "[INFO] build fused attention spatial-reduction TensorRT plugin"
echo "[INFO] build directory: ${BUILD_DIR}"
echo "[INFO] CUDA architectures: ${CUDA_ARCHITECTURES}"

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" \
    --target egcinet_fused_sr_plugin \
    --parallel "${JOBS}"

PLUGIN_SO="${BUILD_DIR}/lib/libegcinet_fused_sr_plugin.so"
if [ ! -s "${PLUGIN_SO}" ]; then
    echo "[ERROR] plugin library was not generated: ${PLUGIN_SO}" >&2
    exit 1
fi

echo "[PASS] ${PLUGIN_SO}"
