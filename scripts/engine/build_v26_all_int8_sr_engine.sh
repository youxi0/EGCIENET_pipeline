#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "${PROJECT_ROOT}/scripts/common/tensorrt_env.sh"

BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
PACKED_DWCONV_BUILD_DIR="${PACKED_DWCONV_BUILD_DIR:-${PROJECT_ROOT}/build/packed_dwconv_plugin}"
FUSED_SR_BUILD_DIR="${FUSED_SR_BUILD_DIR:-${PROJECT_ROOT}/build/fused_sr_plugin}"
ONNX="${ONNX:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v26_q_block134_kv_all_packed_sr_all_int8.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v26_q_block134_kv_all_packed_sr_all_int8.engine}"
PACKED_DWCONV_SO="${PACKED_DWCONV_SO:-${PACKED_DWCONV_BUILD_DIR}/lib/libegcinet_packed_dwconv_plugin.so}"
FUSED_SR_SO="${FUSED_SR_SO:-${FUSED_SR_BUILD_DIR}/lib/libegcinet_fused_sr_plugin.so}"
WORKSPACE_MIB="${WORKSPACE_MIB:-256}"
TIMING_CACHE="${TIMING_CACHE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v13_v14.timing.cache}"
TRT_TEMP_DIR="${TRT_TEMP_DIR:-${BUILD_DIR}/tensorrt_tmp}"
LAYER_INFO="${LAYER_INFO:-${ENGINE%.engine}_layers.json}"
BUILD_LABEL="${BUILD_LABEL:-V26 all-stage INT8 spatial-reduction}"

TRTEXEC=$(resolve_trtexec)

for required_file in "${ONNX}" "${PACKED_DWCONV_SO}" "${FUSED_SR_SO}"; do
    if [ ! -s "${required_file}" ]; then
        echo "[ERROR] required file not found: ${required_file}" >&2
        exit 1
    fi
done

configure_tensorrt_library_path
mkdir -p "$(dirname "${ENGINE}")" "$(dirname "${LAYER_INFO}")"
mkdir -p "$(dirname "${TIMING_CACHE}")" "${TRT_TEMP_DIR}"

echo "[INFO] build ${BUILD_LABEL} engine"
echo "[INFO] onnx: ${ONNX}"
echo "[INFO] engine: ${ENGINE}"

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

for output_file in "${ENGINE}" "${LAYER_INFO}"; do
    if [ ! -s "${output_file}" ]; then
        echo "[ERROR] TensorRT did not write: ${output_file}" >&2
        exit 1
    fi
done

echo "[PASS] ${BUILD_LABEL} engine: ${ENGINE}"
echo "[PASS] layer info: ${LAYER_INFO}"
