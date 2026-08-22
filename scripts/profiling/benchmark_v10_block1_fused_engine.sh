#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "${PROJECT_ROOT}/scripts/common/tensorrt_env.sh"

# 步骤 1：固定 engine、插件和 benchmark 条件，环境变量可覆盖默认值。
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v10_block1_fused.engine}"
PLUGIN_SO="${PLUGIN_SO:-${PROJECT_ROOT}/build/block1_fused_plugin/lib/libegcinet_block1_fused_plugin.so}"
WARMUP_MS="${WARMUP_MS:-1000}"
DURATION_SECONDS="${DURATION_SECONDS:-10}"
ITERATIONS="${ITERATIONS:-1000}"
PROFILE_JSON="${PROFILE_JSON:-${ENGINE%.engine}_profile.json}"

TRTEXEC=$(resolve_trtexec)

# 步骤 2：反序列化自定义插件 engine 前，仍需提供同一个插件动态库。
if [ ! -s "${ENGINE}" ]; then
    echo "[ERROR] engine not found: ${ENGINE}" >&2
    exit 1
fi
if [ ! -s "${PLUGIN_SO}" ]; then
    echo "[ERROR] plugin library not found: ${PLUGIN_SO}" >&2
    exit 1
fi

configure_tensorrt_library_path
mkdir -p "$(dirname "${PROFILE_JSON}")"

# 步骤 3：沿用无数据拷贝、CUDA Graph、相同预热和迭代次数，保证与 V7 可比。
echo "[INFO] benchmark V10 with the same no-transfer/CUDA-graph conditions"
"${TRTEXEC}" \
    "--loadEngine=${ENGINE}" \
    "--dynamicPlugins=${PLUGIN_SO}" \
    --noDataTransfers \
    --useCudaGraph \
    "--warmUp=${WARMUP_MS}" \
    "--duration=${DURATION_SECONDS}" \
    "--iterations=${ITERATIONS}" \
    --profilingVerbosity=detailed \
    --dumpProfile \
    "--exportProfile=${PROFILE_JSON}"

# 步骤 4：保留逐层 profile，后续检查三个插件层及整网耗时。
echo "[PASS] profile: ${PROFILE_JSON}"
