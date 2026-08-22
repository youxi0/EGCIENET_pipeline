#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "${PROJECT_ROOT}/scripts/common/tensorrt_env.sh"

BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_PIPELINE="${BUILD_PIPELINE:-ON}"
BUILD_INT8_CALIBRATOR="${BUILD_INT8_CALIBRATOR:-OFF}"
BUILD_BLOCK1_FUSED_PLUGIN="${BUILD_BLOCK1_FUSED_PLUGIN:-ON}"
BUILD_PACKED_DWCONV_PLUGIN="${BUILD_PACKED_DWCONV_PLUGIN:-ON}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-87}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

validate_switch() {
    local name="$1"
    local value="$2"
    case "${value}" in
        ON|OFF) ;;
        *)
            echo "[ERROR] ${name} must be ON or OFF, got: ${value}" >&2
            exit 2
            ;;
    esac
}

validate_switch BUILD_PIPELINE "${BUILD_PIPELINE}"
validate_switch BUILD_INT8_CALIBRATOR "${BUILD_INT8_CALIBRATOR}"
validate_switch BUILD_BLOCK1_FUSED_PLUGIN "${BUILD_BLOCK1_FUSED_PLUGIN}"
validate_switch BUILD_PACKED_DWCONV_PLUGIN "${BUILD_PACKED_DWCONV_PLUGIN}"

echo "[INFO] project root:         ${PROJECT_ROOT}"
echo "[INFO] build dir:            ${BUILD_DIR}"
echo "[INFO] build type:           ${BUILD_TYPE}"
echo "[INFO] CUDA architectures:   ${CUDA_ARCHITECTURES}"
echo "[INFO] jobs:                 ${JOBS}"
echo "[INFO] pipeline:             ${BUILD_PIPELINE}"
echo "[INFO] INT8 calibrator:      ${BUILD_INT8_CALIBRATOR}"
echo "[INFO] Block1 fused plugin:  ${BUILD_BLOCK1_FUSED_PLUGIN}"
echo "[INFO] packed DWConv plugin: ${BUILD_PACKED_DWCONV_PLUGIN}"

configure_tensorrt_library_path

CMAKE_ARGS=(
    -S "${PROJECT_ROOT}"
    -B "${BUILD_DIR}"
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
    "-DEGCINET_BUILD_PIPELINE=${BUILD_PIPELINE}"
    "-DEGCINET_BUILD_INT8_CALIBRATOR=${BUILD_INT8_CALIBRATOR}"
    "-DEGCINET_BUILD_BLOCK1_FUSED_PLUGIN=${BUILD_BLOCK1_FUSED_PLUGIN}"
    "-DEGCINET_BUILD_PACKED_DWCONV_PLUGIN=${BUILD_PACKED_DWCONV_PLUGIN}"
)

if [ -n "${TENSORRT_ROOT:-}" ]; then
    CMAKE_ARGS+=("-DTENSORRT_ROOT=${TENSORRT_ROOT}")
    echo "[INFO] TensorRT root: ${TENSORRT_ROOT}"
else
    echo "[INFO] TensorRT: use system multiarch installation"
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

require_output() {
    local output="$1"
    if [ ! -f "${output}" ]; then
        echo "[ERROR] expected build output is missing: ${output}" >&2
        exit 1
    fi
    echo "[INFO] built: ${output}"
}

if [ "${BUILD_PIPELINE}" = "ON" ]; then
    require_output "${BUILD_DIR}/bin/egcinet_pipeline"
    require_output "${BUILD_DIR}/bin/egcinet_infer_image"
    require_output "${BUILD_DIR}/bin/egcinet_validate"
fi

if [ "${BUILD_INT8_CALIBRATOR}" = "ON" ]; then
    require_output "${BUILD_DIR}/bin/egcinet_calibrate_int8"
fi

if [ "${BUILD_BLOCK1_FUSED_PLUGIN}" = "ON" ]; then
    require_output "${BUILD_DIR}/lib/libegcinet_block1_fused_plugin.so"
fi

if [ "${BUILD_PACKED_DWCONV_PLUGIN}" = "ON" ]; then
    require_output "${BUILD_DIR}/lib/libegcinet_packed_dwconv_plugin.so"
fi

echo "[PASS] full build finished"

if [ "${BUILD_PACKED_DWCONV_PLUGIN}" = "ON" ]; then
    echo "[INFO] load the fused packed DWConv plugin with:"
    echo "       export EGCINET_TRT_PLUGIN_LIBS=${BUILD_DIR}/lib/libegcinet_packed_dwconv_plugin.so"
fi
