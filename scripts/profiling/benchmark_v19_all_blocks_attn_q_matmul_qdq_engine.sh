#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "${PROJECT_ROOT}/scripts/common/tensorrt_env.sh"

PLUGIN_BUILD_DIR="${PLUGIN_BUILD_DIR:-${PROJECT_ROOT}/build/packed_dwconv_plugin}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v19_all_blocks_attn_q_matmul_qdq.engine}"
PLUGIN_SO="${PLUGIN_SO:-${PLUGIN_BUILD_DIR}/lib/libegcinet_packed_dwconv_plugin.so}"
WARMUP_MS="${WARMUP_MS:-1000}"
DURATION_SECONDS="${DURATION_SECONDS:-10}"
ITERATIONS="${ITERATIONS:-10}"
PROFILE_JSON="${PROFILE_JSON:-${PROJECT_ROOT}/results/profile/egcienet_352_multiclass_qdq_v19_all_blocks_attn_q_matmul_qdq_profile.json}"

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

echo "[INFO] benchmark V19 all-block attention Q MatMul Q/DQ"
"${TRTEXEC}" \
    "--loadEngine=${ENGINE}" \
    "--staticPlugins=${PLUGIN_SO}" \
    --noDataTransfers \
    --useCudaGraph \
    "--warmUp=${WARMUP_MS}" \
    "--duration=${DURATION_SECONDS}" \
    "--iterations=${ITERATIONS}" \
    --separateProfileRun \
    --profilingVerbosity=detailed \
    --dumpProfile \
    "--exportProfile=${PROFILE_JSON}"

echo "[PASS] profile: ${PROFILE_JSON}"
