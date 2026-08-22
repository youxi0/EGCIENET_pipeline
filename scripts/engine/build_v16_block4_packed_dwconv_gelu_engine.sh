#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "${PROJECT_ROOT}/scripts/common/tensorrt_env.sh"

# V16：三个 Block4 11x11x2048 节点直接消费 token-major fc1 输出，
# 在 packed kernel 内完成 DWConv + GELU，删除前后的 NCHW 布局转换。
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
PLUGIN_BUILD_DIR="${PLUGIN_BUILD_DIR:-${PROJECT_ROOT}/build/packed_dwconv_plugin}"
ONNX="${ONNX:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v16_block4_packed_dwconv_gelu.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v16_block4_packed_dwconv_gelu.engine}"
PLUGIN_SO="${PLUGIN_SO:-${PLUGIN_BUILD_DIR}/lib/libegcinet_packed_dwconv_plugin.so}"
WORKSPACE_MIB="${WORKSPACE_MIB:-256}"
TIMING_CACHE="${TIMING_CACHE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v13_v14.timing.cache}"
TRT_TEMP_DIR="${TRT_TEMP_DIR:-${BUILD_DIR}/tensorrt_tmp}"

TRTEXEC=$(resolve_trtexec)

if [ ! -s "${ONNX}" ]; then
    echo "[ERROR] V16 ONNX model not found: ${ONNX}" >&2
    echo "[INFO] generate it with scripts/onnx/fuse_block4_packed_dwconv_gelu.py" >&2
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
mkdir -p "${TRT_TEMP_DIR}"

echo "[INFO] build V16 Block4 packed DWConv + GELU engine"
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

echo "[PASS] V16 engine: ${ENGINE}"
echo "[NOTE] Load ${PLUGIN_SO} before deserializing this engine."
