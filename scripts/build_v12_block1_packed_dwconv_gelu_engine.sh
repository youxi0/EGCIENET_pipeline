#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "${PROJECT_ROOT}/scripts/tensorrt_env.sh"

# V12 在三个预排布 DWConv 插件节点中设置 fuse_gelu=1，插件直接输出
# FastGELU 结果，不再保留独立的 TensorRT/Myelin exact-erf GELU 子图。
ONNX="${ONNX:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v12_block1_packed_dwconv_gelu.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v12_block1_packed_dwconv_gelu.engine}"
PLUGIN_SO="${PLUGIN_SO:-${PROJECT_ROOT}/build/block1_packed_dwconv_plugin/lib/libegcinet_block1_packed_dwconv_plugin.so}"
# 4GB Orin 上 2048 MiB 会使 builder 在权重阶段触发 NvMap ENOMEM；
# 实测 512 MiB 可完成 tactic 搜索，最终 engine 的最大 scratch 仅约 2.84 MiB。
WORKSPACE_MIB="${WORKSPACE_MIB:-512}"
LAYER_INFO="${LAYER_INFO:-${ENGINE%.engine}_layers.json}"

TRTEXEC=$(resolve_trtexec)

if [ ! -s "${ONNX}" ]; then
    echo "[ERROR] V12 ONNX model not found: ${ONNX}" >&2
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

TRTEXEC_ARGS=(
    "--onnx=${ONNX}"
    "--saveEngine=${ENGINE}"
    "--staticPlugins=${PLUGIN_SO}"
    --int8
    --fp16
    "--memPoolSize=workspace:${WORKSPACE_MIB}"
    --profilingVerbosity=detailed
    --dumpLayerInfo
    "--exportLayerInfo=${LAYER_INFO}"
    --skipInference
)

echo "[INFO] build V12 packed DWConv + FastGELU engine"
echo "[INFO] trtexec: ${TRTEXEC}"
echo "[INFO] onnx: ${ONNX}"
echo "[INFO] plugin: ${PLUGIN_SO}"
echo "[INFO] engine: ${ENGINE}"
"${TRTEXEC}" "${TRTEXEC_ARGS[@]}"

if [ ! -s "${ENGINE}" ]; then
    echo "[ERROR] trtexec did not write engine: ${ENGINE}" >&2
    exit 1
fi

echo "[PASS] V12 engine: ${ENGINE}"
echo "[NOTE] Load ${PLUGIN_SO} before deserializing this engine."
