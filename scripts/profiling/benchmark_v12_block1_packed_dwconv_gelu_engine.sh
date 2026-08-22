#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "${PROJECT_ROOT}/scripts/common/tensorrt_env.sh"

ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v12_block1_packed_dwconv_gelu.engine}"
PLUGIN_SO="${PLUGIN_SO:-${PROJECT_ROOT}/build/packed_dwconv_plugin/lib/libegcinet_packed_dwconv_plugin.so}"
WARMUP_MS="${WARMUP_MS:-1000}"
DURATION_SECONDS="${DURATION_SECONDS:-10}"
ITERATIONS="${ITERATIONS:-10}"
PROFILE_JSON="${PROFILE_JSON:-${ENGINE%.engine}_profile.json}"

TRTEXEC=$(resolve_trtexec)

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

echo "[INFO] benchmark V12 packed DWConv + FastGELU"
"${TRTEXEC}" \
    "--loadEngine=${ENGINE}" \
    "--staticPlugins=${PLUGIN_SO}" \
    --noDataTransfers \
    --useCudaGraph \
    "--warmUp=${WARMUP_MS}" \
    "--duration=${DURATION_SECONDS}" \
    "--iterations=${ITERATIONS}" \
    --profilingVerbosity=detailed \
    --dumpProfile \
    "--exportProfile=${PROFILE_JSON}"

echo "[PASS] profile: ${PROFILE_JSON}"
