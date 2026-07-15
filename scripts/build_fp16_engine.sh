#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

TENSORRT_ROOT="${TENSORRT_ROOT:-/home/fulin/haiyanghuang/TensorRT-8.6.1.6}"
TRTEXEC="${TRTEXEC:-${TENSORRT_ROOT}/bin/trtexec}"
ONNX="${ONNX:-${PROJECT_ROOT}/models/egcinet_352.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcinet_352_fp16.engine}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"
LAYER_PRECISIONS="${LAYER_PRECISIONS:-}"
LAYER_OUTPUT_TYPES="${LAYER_OUTPUT_TYPES:-}"

# 优先使用指定 TensorRT 目录中的 trtexec，找不到时再查询 PATH。
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

TRTEXEC_ARGS=(
    "--onnx=${ONNX}"
    "--saveEngine=${ENGINE}"
    --fp16
    "--memPoolSize=workspace:${WORKSPACE_MIB}"
    --profilingVerbosity=detailed
    --skipInference
)

# 使用 ONNX opset 17 的标准 LayerNormalization 时无需手动指定；旧图可通过环境变量传入精度约束。
if [ -n "${LAYER_PRECISIONS}" ] || [ -n "${LAYER_OUTPUT_TYPES}" ]; then
    TRTEXEC_ARGS+=(--precisionConstraints=obey)
fi
if [ -n "${LAYER_PRECISIONS}" ]; then
    TRTEXEC_ARGS+=("--layerPrecisions=${LAYER_PRECISIONS}")
fi
if [ -n "${LAYER_OUTPUT_TYPES}" ]; then
    TRTEXEC_ARGS+=("--layerOutputTypes=${LAYER_OUTPUT_TYPES}")
fi

echo "[INFO] build FP16 engine with trtexec"
echo "[INFO] trtexec: ${TRTEXEC}"
echo "[INFO] onnx: ${ONNX}"
echo "[INFO] engine: ${ENGINE}"
"${TRTEXEC}" "${TRTEXEC_ARGS[@]}"

if [ ! -s "${ENGINE}" ]; then
    echo "[ERROR] trtexec did not write engine: ${ENGINE}" >&2
    exit 1
fi

echo "[PASS] FP16 engine: ${ENGINE}"
