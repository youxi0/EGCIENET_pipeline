#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# 步骤 1：V11/V12 共用独立构建目录和动态库，避免与 V10 插件混用。
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build/block1_packed_dwconv_plugin}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-87}"
JOBS="${JOBS:-$(nproc)}"

CMAKE_ARGS=(
    -S "${PROJECT_ROOT}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DEGCINET_BUILD_BLOCK1_PACKED_DWCONV_PLUGIN=ON
    -DEGCINET_BUILD_BLOCK1_FUSED_PLUGIN=OFF
    -DEGCINET_BUILD_INT8_CALIBRATOR=OFF
    -DEGCINET_BUILD_PIPELINE=OFF
    "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
)

# 步骤 2：非系统安装的 TensorRT 通过环境变量指定根目录。
if [ -n "${TENSORRT_ROOT:-}" ]; then
    CMAKE_ARGS+=("-DTENSORRT_ROOT=${TENSORRT_ROOT}")
fi

echo "[INFO] build block1 packed-weight DWConv TensorRT plugin"
echo "[INFO] build directory: ${BUILD_DIR}"
echo "[INFO] CUDA architectures: ${CUDA_ARCHITECTURES}"

# 步骤 3：只构建 packed DWConv 插件 target。
cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" \
    --target egcinet_block1_packed_dwconv_plugin \
    --parallel "${JOBS}"

PLUGIN_SO="${BUILD_DIR}/lib/libegcinet_block1_packed_dwconv_plugin.so"
if [ ! -s "${PLUGIN_SO}" ]; then
    echo "[ERROR] plugin library was not generated: ${PLUGIN_SO}" >&2
    exit 1
fi

echo "[PASS] ${PLUGIN_SO}"
