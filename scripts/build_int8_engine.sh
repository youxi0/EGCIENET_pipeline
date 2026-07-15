#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

TENSORRT_ROOT="${TENSORRT_ROOT:-/home/fulin/haiyanghuang/TensorRT-8.6.1.6}"
TRTEXEC="${TRTEXEC:-${TENSORRT_ROOT}/bin/trtexec}"

ONNX="${ONNX:-${PROJECT_ROOT}/models/egcinet_352.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcinet_352_int8.engine}"
CALIB_CACHE="${CALIB_CACHE:-${PROJECT_ROOT}/models/egcinet_352_int8.cache}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"
LAYER_PRECISIONS="${LAYER_PRECISIONS:-}"
LAYER_OUTPUT_TYPES="${LAYER_OUTPUT_TYPES:-}"

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

export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${TENSORRT_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$(dirname "${ENGINE}")"

if [ ! -s "${CALIB_CACHE}" ]; then
    echo "[ERROR] calibration cache not found: ${CALIB_CACHE}" >&2
    echo "[INFO] run: bash scripts/calibrate_int8.sh" >&2
    exit 1
fi

TRTEXEC_ARGS=(
    "--onnx=${ONNX}"
    "--saveEngine=${ENGINE}"
    --int8
    --fp16
    "--calib=${CALIB_CACHE}"
    "--memPoolSize=workspace:${WORKSPACE_MIB}"
    --profilingVerbosity=detailed
    --skipInference
)

if [ -n "${LAYER_PRECISIONS}" ] || [ -n "${LAYER_OUTPUT_TYPES}" ]; then
    TRTEXEC_ARGS+=(--precisionConstraints=obey)
fi
if [ -n "${LAYER_PRECISIONS}" ]; then
    TRTEXEC_ARGS+=("--layerPrecisions=${LAYER_PRECISIONS}")
fi
if [ -n "${LAYER_OUTPUT_TYPES}" ]; then
    TRTEXEC_ARGS+=("--layerOutputTypes=${LAYER_OUTPUT_TYPES}")
fi

echo "[INFO] build INT8 engine with trtexec"
echo "[INFO] trtexec: ${TRTEXEC}"
echo "[INFO] onnx: ${ONNX}"
echo "[INFO] engine: ${ENGINE}"
echo "[INFO] calib cache: ${CALIB_CACHE}"
"${TRTEXEC}" "${TRTEXEC_ARGS[@]}"

if [ ! -s "${ENGINE}" ]; then
    echo "[ERROR] trtexec did not write engine: ${ENGINE}" >&2
    exit 1
fi

echo "[PASS] INT8 engine: ${ENGINE}"
