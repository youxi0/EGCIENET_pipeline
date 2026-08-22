#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

# 步骤 1：集中定义构建目录、Orin 架构和并行度，允许环境变量覆盖。
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build/block1_fused_plugin}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-87}"
JOBS="${JOBS:-$(nproc)}"

CMAKE_ARGS=(
    -S "${PROJECT_ROOT}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DEGCINET_BUILD_BLOCK1_FUSED_PLUGIN=ON
    -DEGCINET_BUILD_INT8_CALIBRATOR=OFF
    -DEGCINET_BUILD_PIPELINE=OFF
    "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
)

# 步骤 2：若 TensorRT 不是系统安装，显式把安装根目录交给 CMake。
if [ -n "${TENSORRT_ROOT:-}" ]; then
    CMAKE_ARGS+=("-DTENSORRT_ROOT=${TENSORRT_ROOT}")
fi

echo "[INFO] build experimental block1 fused TensorRT plugin"
echo "[INFO] build directory: ${BUILD_DIR}"
echo "[INFO] CUDA architectures: ${CUDA_ARCHITECTURES}"
# 步骤 3：只配置并编译插件 target，不构建校准器和完整流水线。
cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" \
    --target egcinet_block1_fused_plugin \
    --parallel "${JOBS}"

PLUGIN_SO="${BUILD_DIR}/lib/libegcinet_block1_fused_plugin.so"
# 步骤 4：检查最终 .so，避免后续 engine 构建误用空路径或旧产物。
if [ ! -s "${PLUGIN_SO}" ]; then
    echo "[ERROR] plugin library was not generated: ${PLUGIN_SO}" >&2
    exit 1
fi

echo "[PASS] ${PLUGIN_SO}"
