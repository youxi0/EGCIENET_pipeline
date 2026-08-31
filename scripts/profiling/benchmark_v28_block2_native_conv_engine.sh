#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

export ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v28_q_all_kv_all_packed_sr_block13_int8_block2_native_conv.engine}"
export PROFILE_JSON="${PROFILE_JSON:-${PROJECT_ROOT}/results/profile/v28_block2_native_conv_profile.json}"
export BENCHMARK_LABEL="${BENCHMARK_LABEL:-V28 Block2 native INT8 spatial-reduction Conv}"

exec bash "${PROJECT_ROOT}/scripts/profiling/benchmark_v26_all_int8_sr_engine.sh"
