#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "${PROJECT_ROOT}/scripts/common/tensorrt_env.sh"

# V13：18 个 Block3 22x22x1280 DWConv 节点使用 packed-weight 插件，
# GELU 仍由后续 TensorRT/Myelin 子图执行。
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
ONNX="${ONNX:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v13_block3_packed_dwconv.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v13_block3_packed_dwconv.engine}"
PLUGIN_SO="${PLUGIN_SO:-${BUILD_DIR}/lib/libegcinet_packed_dwconv_plugin.so}"
WORKSPACE_MIB="${WORKSPACE_MIB:-256}"
TIMING_CACHE="${TIMING_CACHE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v13_v14.timing.cache}"
TRT_TEMP_DIR="${TRT_TEMP_DIR:-${BUILD_DIR}/tensorrt_tmp}"

TRTEXEC=$(resolve_trtexec)

if [ ! -s "${ONNX}" ]; then
    echo "[ERROR] V13 ONNX model not found: ${ONNX}" >&2
    exit 1
fi
if [ ! -s "${PLUGIN_SO}" ]; then
    echo "[ERROR] packed DWConv plugin not found: ${PLUGIN_SO}" >&2
    echo "[INFO] run: bash scripts/pipeline/build.sh" >&2
    exit 1
fi

configure_tensorrt_library_path
mkdir -p "$(dirname "${ENGINE}")"
mkdir -p "$(dirname "${TIMING_CACHE}")"
mkdir -p "${TRT_TEMP_DIR}"

echo "[INFO] build V13 Block3 packed DWConv engine"
echo "[INFO] trtexec: ${TRTEXEC}"
echo "[INFO] onnx: ${ONNX}"
echo "[INFO] plugin: ${PLUGIN_SO}"
echo "[INFO] engine: ${ENGINE}"
echo "[INFO] timing cache: ${TIMING_CACHE}"

"${TRTEXEC}" \
    "--onnx=${ONNX}" \
    "--saveEngine=${ENGINE}" \
    "--staticPlugins=${PLUGIN_SO}" \
    --int8 \
    --fp16 \
    "--memPoolSize=workspace:${WORKSPACE_MIB}" \
    --maxAuxStreams=0 \
    --noCompilationCache \
    "--tempdir=${TRT_TEMP_DIR}" \
    --tempfileControls=in_memory:deny,temporary:allow \
    "--timingCacheFile=${TIMING_CACHE}" \
    --skipInference

if [ ! -s "${ENGINE}" ]; then
    echo "[ERROR] trtexec did not write engine: ${ENGINE}" >&2
    exit 1
fi

echo "[PASS] V13 engine: ${ENGINE}"
echo "[NOTE] Load ${PLUGIN_SO} before deserializing this engine."
