#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "${PROJECT_ROOT}/scripts/tensorrt_env.sh"

# 步骤 1：V10 使用已经插入插件节点的显式 Q/DQ ONNX，并保持独立输出文件名。
ONNX="${ONNX:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v10_block1_fused.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v10_block1_fused.engine}"
PLUGIN_SO="${PLUGIN_SO:-${PROJECT_ROOT}/build/block1_fused_plugin/lib/libegcinet_block1_fused_plugin.so}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"
LAYER_INFO="${LAYER_INFO:-${ENGINE%.engine}_layers.json}"

TRTEXEC=$(resolve_trtexec)

# 步骤 2：构建前同时确认 ONNX 和插件动态库存在。
if [ ! -s "${ONNX}" ]; then
    echo "[ERROR] V10 ONNX model not found: ${ONNX}" >&2
    exit 1
fi
if [ ! -s "${PLUGIN_SO}" ]; then
    echo "[ERROR] plugin library not found: ${PLUGIN_SO}" >&2
    echo "[INFO] run: bash scripts/build_block1_fused_plugin.sh" >&2
    exit 1
fi

configure_tensorrt_library_path
mkdir -p "$(dirname "${ENGINE}")"
mkdir -p "$(dirname "${LAYER_INFO}")"

# 步骤 3：通过 --dynamicPlugins 先注册自定义算子，再让 ONNX parser 构建 engine。
# 模型本身已有显式 Q/DQ，因此这里只打开 INT8/FP16 builder 能力，不再传校准缓存。
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

echo "[INFO] build V10 block1 fused engine"
echo "[INFO] trtexec: ${TRTEXEC}"
echo "[INFO] onnx: ${ONNX}"
echo "[INFO] plugin: ${PLUGIN_SO}"
echo "[INFO] engine: ${ENGINE}"
"${TRTEXEC}" "${TRTEXEC_ARGS[@]}"

# 步骤 4：确认 engine 和详细 layer 信息已经成功输出。
if [ ! -s "${ENGINE}" ]; then
    echo "[ERROR] trtexec did not write engine: ${ENGINE}" >&2
    exit 1
fi

echo "[PASS] V10 engine: ${ENGINE}"
echo "[NOTE] The plugin is external; load ${PLUGIN_SO} before deserializing this engine."
