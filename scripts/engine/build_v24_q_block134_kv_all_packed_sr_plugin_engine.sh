#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "${PROJECT_ROOT}/scripts/common/tensorrt_env.sh"

BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
PACKED_DWCONV_BUILD_DIR="${PACKED_DWCONV_BUILD_DIR:-${PROJECT_ROOT}/build/packed_dwconv_plugin}"
FUSED_SR_BUILD_DIR="${FUSED_SR_BUILD_DIR:-${PROJECT_ROOT}/build/fused_sr_plugin}"
ONNX="${ONNX:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v24_q_block134_kv_all_packed_sr_plugin.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v24_q_block134_kv_all_packed_sr_plugin.engine}"
PACKED_DWCONV_SO="${PACKED_DWCONV_SO:-${PACKED_DWCONV_BUILD_DIR}/lib/libegcinet_packed_dwconv_plugin.so}"
FUSED_SR_SO="${FUSED_SR_SO:-${FUSED_SR_BUILD_DIR}/lib/libegcinet_fused_sr_plugin.so}"
WORKSPACE_MIB="${WORKSPACE_MIB:-256}"
TIMING_CACHE="${TIMING_CACHE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v13_v14.timing.cache}"
TRT_TEMP_DIR="${TRT_TEMP_DIR:-${BUILD_DIR}/tensorrt_tmp}"
LAYER_INFO="${LAYER_INFO:-${ENGINE%.engine}_layers.json}"

TRTEXEC=$(resolve_trtexec)

for required_file in "${ONNX}" "${PACKED_DWCONV_SO}" "${FUSED_SR_SO}"; do
    if [ ! -s "${required_file}" ]; then
        echo "[ERROR] required file not found: ${required_file}" >&2
        exit 1
    fi
done

configure_tensorrt_library_path
mkdir -p "$(dirname "${ENGINE}")"
mkdir -p "$(dirname "${TIMING_CACHE}")"
mkdir -p "$(dirname "${LAYER_INFO}")"
mkdir -p "${TRT_TEMP_DIR}"

echo "[INFO] build V24 packed spatial-reduction plugin engine"
echo "[INFO] trtexec: ${TRTEXEC}"
echo "[INFO] onnx: ${ONNX}"
echo "[INFO] packed DWConv plugin: ${PACKED_DWCONV_SO}"
echo "[INFO] fused SR plugin: ${FUSED_SR_SO}"
echo "[INFO] engine: ${ENGINE}"
echo "[INFO] timing cache: ${TIMING_CACHE}"

"${TRTEXEC}" \
    "--onnx=${ONNX}" \
    "--saveEngine=${ENGINE}" \
    "--staticPlugins=${PACKED_DWCONV_SO}" \
    "--staticPlugins=${FUSED_SR_SO}" \
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

echo "[PASS] V24 engine: ${ENGINE}"
echo "[PASS] layer info: ${LAYER_INFO}"
echo "[NOTE] Load both plugin libraries before deserializing this engine."
