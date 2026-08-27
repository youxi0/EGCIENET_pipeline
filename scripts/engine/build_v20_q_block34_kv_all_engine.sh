#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "${PROJECT_ROOT}/scripts/common/tensorrt_env.sh"

# V20：Block3/4 的 attention Q projection 以及所有 Block 的 K/V
# projection 均已插入显式 Q/DQ。显式量化构建不需要 INT8 calibrator。
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
PLUGIN_BUILD_DIR="${PLUGIN_BUILD_DIR:-${PROJECT_ROOT}/build/packed_dwconv_plugin}"
ONNX="${ONNX:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v20_q_block34_kv_all.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v20_q_block34_kv_all.engine}"
PLUGIN_SO="${PLUGIN_SO:-${PLUGIN_BUILD_DIR}/lib/libegcinet_packed_dwconv_plugin.so}"
WORKSPACE_MIB="${WORKSPACE_MIB:-256}"
TIMING_CACHE="${TIMING_CACHE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v13_v14.timing.cache}"
TRT_TEMP_DIR="${TRT_TEMP_DIR:-${BUILD_DIR}/tensorrt_tmp}"
LAYER_INFO="${LAYER_INFO:-${ENGINE%.engine}_layers.json}"

TRTEXEC=$(resolve_trtexec)

if [ ! -s "${ONNX}" ]; then
    echo "[ERROR] V20 ONNX model not found: ${ONNX}" >&2
    exit 1
fi
if [ ! -s "${PLUGIN_SO}" ]; then
    echo "[ERROR] packed DWConv plugin not found: ${PLUGIN_SO}" >&2
    echo "[INFO] run: bash scripts/plugins/build_packed_dwconv_plugin.sh" >&2
    exit 1
fi

configure_tensorrt_library_path
mkdir -p "$(dirname "${ENGINE}")"
mkdir -p "$(dirname "${TIMING_CACHE}")"
mkdir -p "$(dirname "${LAYER_INFO}")"
mkdir -p "${TRT_TEMP_DIR}"

echo "[INFO] build V20 attention Q/K/V MatMul Q/DQ engine"
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
    --profilingVerbosity=detailed \
    "--exportLayerInfo=${LAYER_INFO}" \
    --skipInference

if [ ! -s "${ENGINE}" ]; then
    echo "[ERROR] trtexec did not write engine: ${ENGINE}" >&2
    exit 1
fi
if [ ! -s "${LAYER_INFO}" ]; then
    echo "[ERROR] trtexec did not write layer information: ${LAYER_INFO}" >&2
    exit 1
fi

echo "[PASS] V20 engine: ${ENGINE}"
echo "[PASS] layer info: ${LAYER_INFO}"
echo "[NOTE] Load ${PLUGIN_SO} before deserializing this engine."
