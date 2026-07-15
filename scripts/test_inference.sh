#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

TENSORRT_ROOT="${TENSORRT_ROOT:-/home/fulin/haiyanghuang/TensorRT-8.6.1.6}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build-debug}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/results/debug_inference}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcinet_352_fp16.engine}"
IMAGE="${IMAGE:-/home/fulin/haiyanghuang/EGCIENET_pipeline/Dataset/AEBIS/Test/JPEGImages/0.jpg}"
SANITIZER="${SANITIZER:-none}"
SKIP_BUILD="${SKIP_BUILD:-0}"
LAUNCH_BLOCKING="${LAUNCH_BLOCKING:-0}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

APP="${BUILD_DIR}/bin/egcinet_infer_image"
BINARY_MASK="${RESULT_DIR}/binary_mask.png"
PROBABILITY_MASK="${RESULT_DIR}/probability_mask.png"
VISUALIZED_IMAGE="${RESULT_DIR}/visualized.png"
RUN_LOG="${RESULT_DIR}/run.log"

if [ -z "${IMAGE}" ]; then
    if [ "$#" -eq 1 ]; then
        IMAGE="$1"
    elif [ "$#" -ge 2 ]; then
        ENGINE="$1"
        IMAGE="$2"
    fi
fi

trap 'echo "[ERROR] test failed at line ${LINENO}" >&2' ERR

# 检查测试所需的命令是否存在。
require_command() {
    local command_name="$1"
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "[ERROR] command not found: ${command_name}" >&2
        exit 1
    fi
}

# 检查输入文件，错误时同时打印变量名和实际路径。
require_file() {
    local label="$1"
    local path="$2"
    if [ ! -s "${path}" ]; then
        echo "[ERROR] ${label} not found: ${path}" >&2
        exit 1
    fi
}

# 使用同一组参数运行指定的 Compute Sanitizer 工具。
run_sanitizer() {
    local tool_name="$1"
    echo "[INFO] compute-sanitizer tool=${tool_name}"
    compute-sanitizer --tool "${tool_name}" \
        "${APP}" \
        --engine "${ENGINE}" \
        --image "${IMAGE}" \
        --output "${BINARY_MASK}" \
        --probability "${PROBABILITY_MASK}" \
        --visualized "${VISUALIZED_IMAGE}"
}

if [ -z "${IMAGE}" ]; then
    echo "Usage:" >&2
    echo "  bash scripts/test_inference.sh <image>" >&2
    echo "  bash scripts/test_inference.sh <engine> <image>" >&2
    echo "  IMAGE=/path/test.jpg ENGINE=/path/model.engine bash scripts/test_inference.sh" >&2
    exit 1
fi

require_command tee
require_file "engine" "${ENGINE}"
require_file "image" "${IMAGE}"

export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${TENSORRT_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

if [ "${SKIP_BUILD}" != "1" ]; then
    require_command cmake
    require_file "TensorRT header" "${TENSORRT_ROOT}/include/NvInfer.h"

    echo "[INFO] configure debug build: ${BUILD_DIR}"
    cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
        -DEGCINET_BUILD_PIPELINE=ON \
        -DTENSORRT_ROOT="${TENSORRT_ROOT}" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CUDA_FLAGS_RELWITHDEBINFO="-O2 -g -lineinfo"

    echo "[INFO] build egcinet_infer_image"
    cmake --build "${BUILD_DIR}" \
        --target egcinet_infer_image \
        --parallel "${JOBS}"
fi

if [ ! -x "${APP}" ]; then
    echo "[ERROR] executable not found: ${APP}" >&2
    echo "[INFO] set SKIP_BUILD=0 or check the build output" >&2
    exit 1
fi

mkdir -p "${RESULT_DIR}"

echo "[INFO] engine: ${ENGINE}"
echo "[INFO] image: ${IMAGE}"
echo "[INFO] result dir: ${RESULT_DIR}"

APP_ARGS=(
    --engine "${ENGINE}"
    --image "${IMAGE}"
    --output "${BINARY_MASK}"
    --probability "${PROBABILITY_MASK}"
    --visualized "${VISUALIZED_IMAGE}"
)

echo "[INFO] run full GPU inference chain"
if [ "${LAUNCH_BLOCKING}" = "1" ]; then
    CUDA_LAUNCH_BLOCKING=1 "${APP}" "${APP_ARGS[@]}" 2>&1 | tee "${RUN_LOG}"
else
    "${APP}" "${APP_ARGS[@]}" 2>&1 | tee "${RUN_LOG}"
fi

require_file "binary mask" "${BINARY_MASK}"
require_file "probability mask" "${PROBABILITY_MASK}"
require_file "visualized image" "${VISUALIZED_IMAGE}"

case "${SANITIZER}" in
    none)
        ;;
    memcheck)
        require_command compute-sanitizer
        run_sanitizer memcheck
        ;;
    all)
        require_command compute-sanitizer
        run_sanitizer memcheck
        run_sanitizer initcheck
        run_sanitizer synccheck
        ;;
    *)
        echo "[ERROR] SANITIZER must be none, memcheck, or all: ${SANITIZER}" >&2
        exit 1
        ;;
esac

echo "[PASS] inference test finished"
echo "[PASS] log: ${RUN_LOG}"
echo "[PASS] binary mask: ${BINARY_MASK}"
echo "[PASS] probability mask: ${PROBABILITY_MASK}"
echo "[PASS] visualization: ${VISUALIZED_IMAGE}"
