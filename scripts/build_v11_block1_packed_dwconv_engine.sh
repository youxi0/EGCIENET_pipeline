#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "${PROJECT_ROOT}/scripts/tensorrt_env.sh"

# 步骤 1：V11 ONNX 中只有 DWConv 由插件执行，GELU 仍是 TensorRT 原生子图。
ONNX="${ONNX:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v11_block1_packed_dwconv.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v11_block1_packed_dwconv.engine}"
PLUGIN_SO="${PLUGIN_SO:-${PROJECT_ROOT}/build/block1_packed_dwconv_plugin/lib/libegcinet_block1_packed_dwconv_plugin.so}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"
LAYER_INFO="${LAYER_INFO:-${ENGINE%.engine}_layers.json}"

TRTEXEC=$(resolve_trtexec)

if [ ! -s "${ONNX}" ]; then
    echo "[ERROR] V11 ONNX model not found: ${ONNX}" >&2
    exit 1
fi
if [ ! -s "${PLUGIN_SO}" ]; then
    echo "[ERROR] plugin library not found: ${PLUGIN_SO}" >&2
    echo "[INFO] run: bash scripts/build_block1_packed_dwconv_plugin.sh" >&2
    exit 1
fi

configure_tensorrt_library_path
mkdir -p "$(dirname "${ENGINE}")"
mkdir -p "$(dirname "${LAYER_INFO}")"

# 步骤 2：加载 V11 Creator。packed weight/bias 会随 PluginField 序列化进 engine，
# 但插件代码仍在外部 .so 中，反序列化 engine 时必须加载同一动态库。
TRTEXEC_ARGS=(
    "--onnx=${ONNX}"
    "--saveEngine=${ENGINE}"
    "--dynamicPlugins=${PLUGIN_SO}"
    --int8
    --fp16
    "--memPoolSize=workspace:${WORKSPACE_MIB}"
    --profilingVerbosity=detailed
    --dumpLayerInfo
    "--exportLayerInfo=${LAYER_INFO}"
    --skipInference
)

echo "[INFO] build V11 block1 packed-weight DWConv engine"
echo "[INFO] trtexec: ${TRTEXEC}"
echo "[INFO] onnx: ${ONNX}"
echo "[INFO] plugin: ${PLUGIN_SO}"
echo "[INFO] engine: ${ENGINE}"
"${TRTEXEC}" "${TRTEXEC_ARGS[@]}"

if [ ! -s "${ENGINE}" ]; then
    echo "[ERROR] trtexec did not write engine: ${ENGINE}" >&2
    exit 1
fi

echo "[PASS] V11 engine: ${ENGINE}"
echo "[NOTE] Load ${PLUGIN_SO} before deserializing this engine."
