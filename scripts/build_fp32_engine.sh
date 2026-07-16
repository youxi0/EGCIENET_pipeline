#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

TENSORRT_ROOT="${TENSORRT_ROOT:-/home/fulin/haiyanghuang/TensorRT-8.6.1.6}"
TRTEXEC="${TRTEXEC:-${TENSORRT_ROOT}/bin/trtexec}"
ONNX="${ONNX:-${PROJECT_ROOT}/models/egcinet_352.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcinet_352_fp32.engine}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"

# FP32 engine 只作为同一 TensorRT 后端下的精度基线，不启用 FP16 或 INT8。
if [ ! -x "${TRTEXEC}" ]; then
    if command -v trtexec >/dev/null 2>&1; then
        TRTEXEC=$(command -v trtexec)
    else
        echo "[ERROR] trtexec not found: ${TRTEXEC}" >&2
        exit 1
    fi
fi

if [ ! -s "${ONNX}" ]; then
    echo "[ERROR] ONNX model not found: ${ONNX}" >&2
    exit 1
fi

mkdir -p "$(dirname "${ENGINE}")"
export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${TENSORRT_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

echo "[INFO] build FP32 engine with trtexec"
echo "[INFO] onnx: ${ONNX}"
echo "[INFO] engine: ${ENGINE}"
"${TRTEXEC}" \
    "--onnx=${ONNX}" \
    "--saveEngine=${ENGINE}" \
    --noTF32 \
    "--memPoolSize=workspace:${WORKSPACE_MIB}" \
    --profilingVerbosity=detailed \
    --skipInference

if [ ! -s "${ENGINE}" ]; then
    echo "[ERROR] trtexec did not write engine: ${ENGINE}" >&2
    exit 1
fi

echo "[PASS] FP32 engine: ${ENGINE}"
