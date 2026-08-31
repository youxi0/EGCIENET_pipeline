#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

export ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v27_q_all_kv_all_packed_sr_all_int8_block2_shared_q.engine}"
export PROFILE_JSON="${PROFILE_JSON:-${PROJECT_ROOT}/results/profile/v27_block2_shared_q_sr_profile.json}"
export BENCHMARK_LABEL="${BENCHMARK_LABEL:-V27 Block2 shared-Q spatial-reduction}"

exec bash "${PROJECT_ROOT}/scripts/profiling/benchmark_v26_all_int8_sr_engine.sh"
