#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "${PROJECT_ROOT}/scripts/tensorrt_env.sh"

ONNX="${ONNX:-${PROJECT_ROOT}/models/egcienet_352_multiclass.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_int8.engine}"
CALIB_CACHE="${CALIB_CACHE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_int8.cache}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"
LAYER_PRECISIONS="${LAYER_PRECISIONS:-}"
LAYER_OUTPUT_TYPES="${LAYER_OUTPUT_TYPES:-}"

TRTEXEC=$(resolve_trtexec)

if [ ! -s "${ONNX}" ]; then
    echo "[ERROR] ONNX model not found: ${ONNX}" >&2
    exit 1
fi

configure_tensorrt_library_path
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
