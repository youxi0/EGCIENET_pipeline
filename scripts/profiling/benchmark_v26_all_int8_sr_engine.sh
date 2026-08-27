#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "${PROJECT_ROOT}/scripts/common/tensorrt_env.sh"

PACKED_DWCONV_BUILD_DIR="${PACKED_DWCONV_BUILD_DIR:-${PROJECT_ROOT}/build/packed_dwconv_plugin}"
FUSED_SR_BUILD_DIR="${FUSED_SR_BUILD_DIR:-${PROJECT_ROOT}/build/fused_sr_plugin}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v26_q_block134_kv_all_packed_sr_all_int8.engine}"
PACKED_DWCONV_SO="${PACKED_DWCONV_SO:-${PACKED_DWCONV_BUILD_DIR}/lib/libegcinet_packed_dwconv_plugin.so}"
FUSED_SR_SO="${FUSED_SR_SO:-${FUSED_SR_BUILD_DIR}/lib/libegcinet_fused_sr_plugin.so}"
WARMUP_MS="${WARMUP_MS:-1000}"
DURATION_SECONDS="${DURATION_SECONDS:-10}"
ITERATIONS="${ITERATIONS:-10}"
PROFILE_JSON="${PROFILE_JSON:-${PROJECT_ROOT}/results/profile/v26_all_int8_sr_profile.json}"

TRTEXEC=$(resolve_trtexec)

for required_file in "${ENGINE}" "${PACKED_DWCONV_SO}" "${FUSED_SR_SO}"; do
    if [ ! -s "${required_file}" ]; then
        echo "[ERROR] required file not found: ${required_file}" >&2
        exit 1
    fi
done

configure_tensorrt_library_path
mkdir -p "$(dirname "${PROFILE_JSON}")"

echo "[INFO] benchmark V26 all-stage INT8 spatial-reduction engine"
"${TRTEXEC}" \
    "--loadEngine=${ENGINE}" \
    "--staticPlugins=${PACKED_DWCONV_SO}" \
    "--staticPlugins=${FUSED_SR_SO}" \
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
