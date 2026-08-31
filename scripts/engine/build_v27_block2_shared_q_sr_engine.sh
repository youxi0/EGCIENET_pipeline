#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

export ONNX="${ONNX:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v27_q_all_kv_all_packed_sr_all_int8_block2_shared_q.onnx}"
export ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v27_q_all_kv_all_packed_sr_all_int8_block2_shared_q.engine}"
export LAYER_INFO="${LAYER_INFO:-${ENGINE%.engine}_layers.json}"
export BUILD_LABEL="${BUILD_LABEL:-V27 Block2 shared-Q spatial-reduction}"

exec bash "${PROJECT_ROOT}/scripts/engine/build_v26_all_int8_sr_engine.sh"
